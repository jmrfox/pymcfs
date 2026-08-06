"""Mean curvature flow skeletonization driver.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh as tm

from .laplacian import mcfs_cotangent_laplacian
from .medial import compute_voronoi_poles
from .remesh import (
    collapse_ok_for_edge,
    collapse_short_edges,
    mesh_adjacency,
    mesh_unique_edges,
    split_obtuse_faces,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from sksparse.cholmod import cholesky as _cholmod_cholesky
except ImportError:  # optional acceleration for SPD AtA
    _cholmod_cholesky = None


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _stitch_skeleton_components(G: nx.Graph, *, max_bridge: float) -> nx.Graph:
    """Connect fragmented skeleton CCs by nearest-node bridges under ``max_bridge``."""
    if G.number_of_nodes() == 0:
        return G
    G = G.copy()
    while True:
        comps = list(nx.connected_components(G))
        if len(comps) <= 1:
            break
        best = None  # (dist, u, v)
        for i, ci in enumerate(comps):
            for cj in comps[i + 1 :]:
                for u in ci:
                    pu = np.asarray(G.nodes[u]["pos"], dtype=float)
                    for v in cj:
                        pv = np.asarray(G.nodes[v]["pos"], dtype=float)
                        d = float(np.linalg.norm(pu - pv))
                        if d <= max_bridge and (best is None or d < best[0]):
                            best = (d, u, v)
        if best is None:
            break
        d, u, v = best
        G.add_edge(u, v, weight=d)
    return G


def meso_surface_to_curve_graph(
    V: np.ndarray,
    F: np.ndarray,
    *,
    max_steps: int | None = None,
    deadline: float | None = None,
) -> nx.Graph:
    """Convert a contracted meso-skeleton surface into a curve graph.

    Uses a sorted edge-collapse procedure compatible with Starlab ``mcfskel``.

    ``surfacemesh_filter_to_skeleton`` inserts all initial edges into a priority
    queue, visits them in their initial length order, and collapses an edge only
    while it still bounds a face. Geometry is not moved during this loop. Each
    surviving vertex is positioned afterward at the centroid of the original
    meso-skeleton vertices collapsed into it.
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int).copy()
    n0 = V.shape[0]
    if n0 == 0:
        return nx.Graph()

    initial_edges = mesh_unique_edges(F)
    if initial_edges.size == 0:
        return nx.Graph()
    initial_lengths = np.linalg.norm(
        V[initial_edges[:, 0]] - V[initial_edges[:, 1]], axis=1
    )
    # Starlab breaks equal-length ties by persistent edge index.
    order = np.lexsort((np.arange(initial_edges.shape[0]), initial_lengths))

    parent = np.arange(n0, dtype=int)
    position_sum = V.copy()
    member_count = np.ones(n0, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Maintain live faces with root vertex ids so has_faces(e) is an O(1) map
    # lookup instead of remapping all faces each step.
    face_alive = np.ones(F.shape[0], dtype=bool)
    v_to_faces: list[set[int]] = [set() for _ in range(n0)]
    edge_faces: dict[tuple[int, int], set[int]] = {}

    def _add_face(fi: int) -> None:
        i0, i1, i2 = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            edge_faces.setdefault(_edge_key(u, v), set()).add(fi)
        v_to_faces[i0].add(fi)
        v_to_faces[i1].add(fi)
        v_to_faces[i2].add(fi)

    def _remove_face(fi: int) -> None:
        i0, i1, i2 = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            key = _edge_key(u, v)
            bucket = edge_faces.get(key)
            if bucket is not None:
                bucket.discard(fi)
                if not bucket:
                    del edge_faces[key]
        v_to_faces[i0].discard(fi)
        v_to_faces[i1].discard(fi)
        v_to_faces[i2].discard(fi)

    for fi in range(F.shape[0]):
        _add_face(fi)

    if max_steps is None:
        max_steps = initial_edges.shape[0]

    steps = 0
    for edge_index in order:
        if steps >= max_steps:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        steps += 1
        a0, b0 = initial_edges[int(edge_index)]
        a, b = find(int(a0)), find(int(b0))
        if a == b:
            continue

        # Equivalent to WingedgeMesh::has_faces(e).
        if not edge_faces.get(_edge_key(a, b)):
            continue

        # Starlab deletes vertex(e, 0), keeps vertex(e, 1), and does not move
        # geometry until the collapse loop is complete.
        parent[a] = b
        position_sum[b] += position_sum[a]
        member_count[b] += member_count[a]

        for fi in list(v_to_faces[a]):
            if not face_alive[fi]:
                continue
            _remove_face(fi)
            for k in range(3):
                F[fi, k] = find(int(F[fi, k]))
            i0, i1, i2 = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
            if i0 == i1 or i1 == i2 or i2 == i0:
                face_alive[fi] = False
            else:
                _add_face(fi)

    edges_set: set[tuple[int, int]] = set()
    for a, b in initial_edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            edges_set.add(_edge_key(ra, rb))

    G = nx.Graph()
    root_to_node: dict[int, int] = {}
    roots = sorted({find(i) for i in range(n0)})
    for r in roots:
        nid = len(root_to_node)
        root_to_node[r] = nid
        pos = position_sum[r] / float(member_count[r])
        G.add_node(nid, pos=pos)

    for a, b in edges_set:
        u, v = root_to_node[a], root_to_node[b]
        pu = np.asarray(G.nodes[u]["pos"], dtype=float)
        pv = np.asarray(G.nodes[v]["pos"], dtype=float)
        G.add_edge(u, v, weight=float(np.linalg.norm(pu - pv)))
    return G


@dataclass
class MeanCurvatureFlowSkeletonization:
    """Mean-curvature-flow skeletonization driver for closed triangle meshes.

    Parameters
    ----------
    mesh :
        Input ``trimesh.Trimesh``.
    w_H :
        Quality/speed tradeoff (default 0.1). Larger → faster / coarser.
    w_M :
        Medial-centering tradeoff (default 0.2). ``0`` disables Voronoi poles.
    min_edge_length, max_triangle_angle :
        Remesh controls during contraction.
    area_variation_factor :
        Relative area change vs initial area for convergence.
    max_iterations, timeout_seconds :
        Hard stop criteria.
    validate, verbose, log :
        Validation and logging.

    Notes
    -----
    ``w_L`` is fixed at 1 (only weight ratios matter).
    """

    mesh: tm.Trimesh
    w_H: float = 0.1
    w_M: float = 0.2
    min_edge_length: float | None = None
    max_triangle_angle: float = 110.0
    area_variation_factor: float = 1e-4
    max_iterations: int = 500
    timeout_seconds: float | None = 120.0
    zero_TH: float = 1e-7
    validate: bool = True
    verbose: bool = False
    log: logging.Logger | None = None

    V: np.ndarray = field(init=False, repr=False)
    F: np.ndarray = field(init=False, repr=False)
    fixed: np.ndarray = field(init=False, repr=False)
    is_split: np.ndarray = field(init=False, repr=False)
    _constraint_fixed: np.ndarray = field(init=False, repr=False)
    _constraint_split: np.ndarray = field(init=False, repr=False)
    poles: np.ndarray = field(init=False, repr=False)
    pole_valid: np.ndarray = field(init=False, repr=False)
    _min_edge: float = field(init=False, repr=False)
    _area0: float = field(init=False, repr=False)
    _w_L: float = field(init=False, default=1.0, repr=False)
    _deadline: float | None = field(init=False, default=None, repr=False)
    _iter: int = field(init=False, default=0, repr=False)
    _bbox0: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, tm.Trimesh):
            raise TypeError("mesh must be a trimesh.Trimesh")
        self._log = self.log or logger
        if self.validate:
            from .validate import validate_mcfs_mesh

            validate_mcfs_mesh(self.mesh)
        self.V = np.asarray(self.mesh.vertices, dtype=float).copy()
        self.F = np.asarray(self.mesh.faces, dtype=int).copy()
        n = self.V.shape[0]
        self.fixed = np.zeros(n, dtype=bool)
        self.is_split = np.zeros(n, dtype=bool)
        self._constraint_fixed = np.zeros(n, dtype=bool)
        self._constraint_split = np.zeros(n, dtype=bool)
        bb = self.V.max(axis=0) - self.V.min(axis=0)
        self._bbox0 = bb.copy()
        diag = float(np.linalg.norm(bb))
        self._min_edge = (
            float(self.min_edge_length)
            if self.min_edge_length is not None
            else max(diag * 0.002, 1e-12)
        )
        self._area0 = self._surface_area()
        self._faces0 = int(self.F.shape[0])
        self._w_L = 1.0
        self._iter = 0
        # Starlab uses every pole returned by its bounding-box-filtered Voronoi
        # construction. Containment is logged as a diagnostic only.
        self.pole_valid = np.zeros(n, dtype=bool)
        if float(self.w_M) > 0.0:
            try:
                targets, _w = compute_voronoi_poles(self.mesh)
                self.poles = np.asarray(targets, dtype=float)
                self.pole_valid = self._poles_inside_mesh(self.poles)
                n_valid = int(self.pole_valid.sum())
                self._vinfo(
                    "Voronoi poles: %d/%d inside mesh (diagnostic only)",
                    n_valid,
                    n,
                )
            except Exception as e:
                self._log.warning("Voronoi poles failed (%s); setting w_M effective 0", e)
                self.poles = self.V.copy()
                self.w_M = 0.0
        else:
            self.poles = self.V.copy()
        self._deadline = None
        self._vinfo(
            "MCFS init: n=%d f=%d min_edge=%.4g area0=%.4g bbox0=%s w_H=%.3g w_M=%.3g",
            n,
            self.F.shape[0],
            self._min_edge,
            self._area0,
            np.array2string(self._bbox0, precision=4),
            self.w_H,
            self.w_M,
        )
        self._sanity_check_state(stage="init")

    def _vinfo(self, msg: str, *args) -> None:
        """Progress / diagnostic line: info when verbose, else debug."""
        if self.verbose:
            self._log.info(msg, *args)
        else:
            self._log.debug(msg, *args)

    def _surface_area(self) -> float:
        if self.F.size == 0:
            return 0.0
        v0 = self.V[self.F[:, 0]]
        v1 = self.V[self.F[:, 1]]
        v2 = self.V[self.F[:, 2]]
        return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())

    def _bbox_extents(self) -> np.ndarray:
        if self.V.shape[0] == 0:
            return np.zeros(3, dtype=float)
        return self.V.max(axis=0) - self.V.min(axis=0)

    def _poles_inside_mesh(self, poles: np.ndarray) -> np.ndarray:
        """Return boolean mask of poles that lie in the interior of ``self.mesh``.

        Diagnostic only — Starlab does not gate poles on containment. Skipped
        unless ``verbose`` is set so init stays cheap on large meshes.
        """
        poles = np.asarray(poles, dtype=float)
        n = poles.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool)
        if not self.verbose:
            return np.ones(n, dtype=bool)
        try:
            inside = np.asarray(self.mesh.contains(poles), dtype=bool)
            if inside.shape[0] != n:
                return np.zeros(n, dtype=bool)
            return inside
        except Exception as e:
            self._log.warning("pole inside-test failed (%s); treating all poles as valid", e)
            return np.ones(n, dtype=bool)

    def _sync_pole_valid(self) -> None:
        """Keep ``pole_valid`` aligned with ``poles`` after remesh."""
        n = self.V.shape[0]
        if self.pole_valid.shape[0] != n:
            # Remesh may have changed vertex count; containment is diagnostic only.
            if float(self.w_M) > 0.0 and self.poles.shape[0] == n:
                self.pole_valid = (
                    self._poles_inside_mesh(self.poles)
                    if self.verbose
                    else np.ones(n, dtype=bool)
                )
            else:
                self.pole_valid = np.zeros(n, dtype=bool)

    def _sanity_check_state(self, *, stage: str, prev_area: float | None = None) -> None:
        """Log geometric / numerical health of the current meso-skeleton."""
        n, f = int(self.V.shape[0]), int(self.F.shape[0])
        area = self._surface_area()
        bb = self._bbox_extents()
        fixed_n = int(self.fixed.sum()) if self.fixed.shape[0] == n else -1
        finite = bool(np.isfinite(self.V).all()) if n else True
        msg = (
            "sanity[%s] iter=%d n=%d f=%d area=%.4g (%.2f%% of area0) "
            "bbox=%s fixed=%d finite=%s"
            % (
                stage,
                self._iter,
                n,
                f,
                area,
                100.0 * area / max(self._area0, 1e-30),
                np.array2string(bb, precision=4),
                fixed_n,
                finite,
            )
        )
        if float(self.w_M) > 0.0 and self.poles.shape[0] == n and n > 0:
            d = np.linalg.norm(self.V - self.poles, axis=1)
            valid = self.pole_valid if self.pole_valid.shape[0] == n else np.zeros(n, dtype=bool)
            msg += " pole_dist(mean/max)=%.4g/%.4g valid=%d/%d" % (
                float(d.mean()),
                float(d.max()) if d.size else 0.0,
                int(valid.sum()),
                n,
            )
        self._vinfo(msg)

        if not finite:
            self._log.error("sanity[%s]: non-finite vertex coordinates detected", stage)
        if prev_area is not None and prev_area > 0 and area > 1.25 * prev_area:
            self._log.warning(
                "sanity[%s]: area increased sharply %.4g -> %.4g (possible medial overshoot)",
                stage,
                prev_area,
                area,
            )
        if n > 0 and self._bbox0 is not None:
            # Warn if the longest original axis collapsed much faster than others
            # while another axis grew (classic bad medial / remesh feedback).
            growth = bb / np.maximum(self._bbox0, 1e-12)
            if float(growth.max()) > 1.1 and float(area) < 0.5 * self._area0:
                self._log.warning(
                    "sanity[%s]: bbox axis growth=%s after area drop (check poles/remesh)",
                    stage,
                    np.array2string(growth, precision=3),
                )

    def _update_constraint_weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.V.shape[0]
        wL = np.full(n, float(self._w_L), dtype=float)
        wH = np.full(n, float(self.w_H), dtype=float)
        wM = np.full(n, float(self.w_M), dtype=float)
        wL[self._constraint_fixed] = 0.0
        wH[self._constraint_fixed] = 1.0 / max(self.zero_TH, 1e-16)
        wM[self._constraint_fixed] = 0.0
        wM[self._constraint_split] = 0.0
        return wL, wH, wM

    def contract_geometry(self) -> None:
        """Assemble and solve the stacked least-squares geometry update."""
        if self.V.shape[0] == 0 or self.F.shape[0] == 0:
            return
        self._sync_pole_valid()
        V_before = self.V.copy()
        wL, wH, wM = self._update_constraint_weights()

        # EigenContractionHelper::createLHS builds:
        #
        #     [ W_L L ]
        # A = [  W_H  ],  B = [0; W_H V; W_P P]
        #     [  W_P  ]
        #
        # and solves min ||A X - B||² through A.T @ A. Starlab multiplies
        # off-diagonal Laplacian entries by the source vertex's omega_L but
        # leaves the diagonal as the unweighted negative edge-weight sum.
        L = mcfs_cotangent_laplacian(self.V, self.F).tocsr()
        # Scale off-diagonals by omega_L; leave diagonal as the unweighted
        # negative edge-weight sum (Starlab / EigenContractionHelper). Avoid LIL.
        diag = np.asarray(L.diagonal()).ravel()
        L_off = L - sp.diags(diag, format="csr", shape=L.shape)
        L_weighted = (sp.diags(wL) @ L_off) + sp.diags(diag, format="csr")
        WH = sp.diags(wH, format="csr")
        WP = sp.diags(wM, format="csr")
        A = sp.vstack([L_weighted, WH, WP], format="csc")
        rhs = np.vstack(
            [
                np.zeros_like(self.V),
                wH[:, None] * self.V,
                wM[:, None] * self.poles,
            ]
        )
        AtA = (A.T @ A).tocsc()
        At_rhs = A.T @ rhs
        try:
            # Optional CHOLMOD (scikit-sparse) for SPD AtA; else SciPy SuperLU.
            if _cholmod_cholesky is not None:
                try:
                    factor = _cholmod_cholesky(AtA)
                    for c in range(3):
                        self.V[:, c] = np.asarray(
                            factor(np.asarray(At_rhs[:, c]).ravel())
                        ).ravel()
                except Exception:
                    solver = spla.factorized(AtA)
                    for c in range(3):
                        self.V[:, c] = solver(np.asarray(At_rhs[:, c]).ravel())
            else:
                solver = spla.factorized(AtA)
                for c in range(3):
                    self.V[:, c] = solver(np.asarray(At_rhs[:, c]).ravel())
        except Exception as e:
            self._log.warning("MCFS contract_geometry factorization failed: %s; using spsolve", e)
            for c in range(3):
                self.V[:, c] = spla.spsolve(AtA, np.asarray(At_rhs[:, c]).ravel())
        if not np.isfinite(self.V).all():
            self._log.error("contract_geometry produced non-finite vertices; reverting step")
            self.V = V_before
            return
        disp = np.linalg.norm(self.V - V_before, axis=1)
        self._log.debug(
            "contract_geometry: disp mean=%.4g max=%.4g wM_active=%d/%d",
            float(disp.mean()),
            float(disp.max()) if disp.size else 0.0,
            int((wM > 0).sum()),
            int(wM.shape[0]),
        )

    def collapse_edges(self) -> int:
        """Collapse edges shorter than ``min_edge_length``."""
        n_before = self.V.shape[0]
        flags = {"is_split": self.is_split}
        V2, F2, n, fixed2, poles2 = collapse_short_edges(
            self.V,
            self.F,
            min_edge_length=self._min_edge,
            fixed=self.fixed,
            poles=self.poles,
            vertex_flags=flags,
            deadline=self._deadline,
        )
        if n > 0 or V2.shape[0] != n_before:
            self.V, self.F = V2, F2
            self.fixed = fixed2 if fixed2 is not None else np.zeros(V2.shape[0], dtype=bool)
            self.poles = poles2 if poles2 is not None else self.V.copy()
            self.is_split = np.asarray(
                flags.get("is_split", np.zeros(V2.shape[0], dtype=bool)),
                dtype=bool,
            )
            self._sync_pole_valid()
        if n:
            self._vinfo(
                "collapse_edges: %d collapses -> n=%d f=%d", n, self.V.shape[0], self.F.shape[0]
            )
        return int(n)

    def split_faces(self) -> int:
        """Split faces with an angle larger than ``max_triangle_angle``."""
        V2, F2, n, fixed2, poles2, split2 = split_obtuse_faces(
            self.V,
            self.F,
            max_angle_deg=self.max_triangle_angle,
            short_edge=max(self.zero_TH, 1e-12),
            fixed=self.fixed,
            poles=self.poles,
            is_split=self.is_split,
            deadline=self._deadline,
        )
        if n > 0:
            self.V, self.F = V2, F2
            self.fixed = fixed2 if fixed2 is not None else np.zeros(V2.shape[0], dtype=bool)
            self.poles = poles2 if poles2 is not None else self.V.copy()
            self.is_split = split2 if split2 is not None else np.zeros(V2.shape[0], dtype=bool)
            self._sync_pole_valid()
        if n:
            self._vinfo(
                "split_faces: %d splits -> n=%d f=%d", n, self.V.shape[0], self.F.shape[0]
            )
        return int(n)

    def _timed_out(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    def detect_degeneracies(self) -> int:
        """Mark vertices on non-collapsible short edges as fixed (degeneracy test)."""
        if self.V.shape[0] == 0 or self.F.shape[0] == 0:
            return 0
        elength_fixed = self._min_edge / 10.0
        topo = mesh_adjacency(self.F, self.V.shape[0])
        newly = 0
        for v in range(self.V.shape[0]):
            if self._timed_out():
                break
            if self.fixed[v]:
                continue
            bad = 0
            for i in range(int(topo.nbr_count[v])):
                u = int(topo.nbr[v, i])
                a, b = (v, u) if v < u else (u, v)
                d = float(np.linalg.norm(self.V[a] - self.V[b]))
                if d < elength_fixed and not collapse_ok_for_edge(
                    a,
                    b,
                    self.V,
                    self.F,
                    check_connectivity=False,
                    topo=topo,
                ):
                    bad += 1
            if bad >= 2:
                self.fixed[v] = True
                newly += 1
        if newly:
            self._vinfo(
                "detect_degeneracies: pinned %d (total fixed=%d)", newly, int(self.fixed.sum())
            )
        return newly

    def contract(self) -> None:
        """One iteration: geometry + collapse + split + degeneracies."""
        self.contract_geometry()
        if self._timed_out():
            return
        self.collapse_edges()
        if self._timed_out():
            return
        new_splits = self.split_faces()
        if self._timed_out():
            return
        # Skelcollapse::updateConstraints runs after geometry but before topology.
        # These snapshots therefore exclude vertices fixed by the detection below,
        # and newly created split vertices retain default weights for one solve.
        self._constraint_fixed = self.fixed.copy()
        self._constraint_split = self.is_split.copy()
        if new_splits:
            self._constraint_split[-new_splits:] = False
        self.detect_degeneracies()

    def contract_until_convergence(self) -> int:
        """Iterate ``contract`` until area change is small, max iterations, or timeout."""
        if self.timeout_seconds is not None and self.timeout_seconds > 0:
            self._deadline = time.monotonic() + float(self.timeout_seconds)
        else:
            self._deadline = None
        prev_area = self._surface_area()
        last_it = 0
        for it in range(1, int(self.max_iterations) + 1):
            last_it = it
            self._iter = it
            if self._timed_out():
                self._vinfo(
                    "stopping: timeout after %.3gs at iter %d",
                    float(self.timeout_seconds or 0.0),
                    it - 1,
                )
                break
            self.contract()
            if self._timed_out():
                self._vinfo(
                    "stopping: timeout after %.3gs during iter %d",
                    float(self.timeout_seconds or 0.0),
                    it,
                )
                break
            area = self._surface_area()
            # Compact progress when verbose; detailed sanity every ~10% or on area jump.
            log_every = max(1, int(self.max_iterations) // 10)
            if self.verbose or it == 1 or it % log_every == 0:
                self._sanity_check_state(stage="iter", prev_area=prev_area)
            elif area > 1.25 * prev_area > 0:
                self._sanity_check_state(stage="iter-area-jump", prev_area=prev_area)

            if prev_area > 0 and abs(prev_area - area) < self.area_variation_factor * max(
                self._area0, 1e-30
            ):
                self._vinfo("converged at iter %d area=%.4g", it, area)
                break
            prev_area = area
            if self.F.shape[0] == 0:
                break
        self._sanity_check_state(stage="final", prev_area=prev_area)
        return last_it

    def meso_skeleton_mesh(self) -> tm.Trimesh:
        """Return the current contracted meso-skeleton as a ``Trimesh``."""
        return tm.Trimesh(vertices=self.V.copy(), faces=self.F.copy(), process=False)

    def convert_to_skeleton(
        self,
        *,
        refine: bool | str = False,
        refine_spacing: float | None = None,
        refine_spacing_frac: float | None = None,
        compress_chains: bool = False,
        resample_spacing: float | None = None,
        keep_largest_component: bool = False,
    ):
        """Convert the meso-skeleton surface into a 1D curve ``Skeleton``.

        Parameters
        ----------
        refine, refine_spacing, refine_spacing_frac, compress_chains, resample_spacing :
            Optional curve refinement (see :func:`pymcfs.skeleton.skeletonize`).
        keep_largest_component :
            If True, keep only the largest connected component of the curve graph.

        Returns
        -------
        Skeleton
        """
        from .refine import resolve_refine_options, refine_skeleton_graph
        from .skeleton import Skeleton

        self._vinfo(
            "convert_to_skeleton: meso n=%d f=%d bbox=%s",
            self.V.shape[0],
            self.F.shape[0],
            np.array2string(self._bbox_extents(), precision=4),
        )
        # Starlab's surface-to-skeleton conversion is a separate complete stage;
        # do not inherit an expired contraction deadline and return a partial
        # triangulated graph.
        G = meso_surface_to_curve_graph(self.V, self.F)
        self._vinfo(
            "convert_to_skeleton: raw curve nodes=%d edges=%d cc=%d",
            G.number_of_nodes(),
            G.number_of_edges(),
            nx.number_connected_components(G) if G.number_of_nodes() else 0,
        )
        if keep_largest_component and G.number_of_nodes() > 0:
            comps = list(nx.connected_components(G))
            if len(comps) > 1:
                largest = max(comps, key=len)
                self._vinfo(
                    "convert_to_skeleton: keeping largest of %d components (%d nodes)",
                    len(comps),
                    len(largest),
                )
                G = G.subgraph(largest).copy()

        mode, spacing, spacing_frac = resolve_refine_options(
            refine=refine,
            refine_spacing=refine_spacing,
            refine_spacing_frac=refine_spacing_frac,
            compress_chains=compress_chains,
            resample_spacing=resample_spacing,
        )
        if mode is not None and G.number_of_nodes() > 0:
            n_before = G.number_of_nodes()
            G = refine_skeleton_graph(
                G, mode=mode, spacing=spacing, spacing_frac=spacing_frac
            )
            self._vinfo(
                "convert_to_skeleton: refine mode=%s %d -> %d nodes",
                mode,
                n_before,
                G.number_of_nodes(),
            )

        # Dense relabel
        mapping = {n: i for i, n in enumerate(G.nodes)}
        if mapping:
            G = nx.relabel_nodes(G, mapping, copy=True)
        nodes_arr = (
            np.array([G.nodes[n]["pos"] for n in G.nodes], dtype=float)
            if G.number_of_nodes()
            else np.zeros((0, 3))
        )
        edges_arr = (
            np.array([[u, v] for u, v in G.edges], dtype=int)
            if G.number_of_edges()
            else np.zeros((0, 2), dtype=int)
        )
        if nodes_arr.shape[0] > 0 and edges_arr.shape[0] > 0:
            total_len = float(
                sum(
                    np.linalg.norm(nodes_arr[int(u)] - nodes_arr[int(v)])
                    for u, v in edges_arr
                )
            )
            self._vinfo(
                "convert_to_skeleton: final nodes=%d edges=%d total_length=%.4g",
                nodes_arr.shape[0],
                edges_arr.shape[0],
                total_len,
            )
        else:
            self._log.warning("convert_to_skeleton: empty skeleton")
        return Skeleton(nodes=nodes_arr, edges=edges_arr, graph=G)
