"""Mean-curvature-flow driver that contracts a closed mesh to a meso-skeleton."""
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
from .medial import compute_voronoi_poles, points_inside_mesh
from .remesh import (
    collapse_ok_for_edge,
    collapse_short_edges,
    mesh_adjacency,
    mesh_unique_edges,
    split_obtuse_faces,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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

    Parameters
    ----------
    V : (n, 3) float
        Meso-skeleton vertex positions.
    F : (m, 3) int
        Triangle indices.
    max_steps :
        Optional cap on collapse attempts (default: number of unique edges).
    deadline :
        Optional ``time.monotonic()`` deadline; abort early when exceeded.

    Returns
    -------
    networkx.Graph
        Undirected curve graph with node attribute ``pos``.
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
    """Step-through driver: contract a closed mesh, then convert to a curve skeleton.

    Each iteration solves a linear system that blends mean-curvature flow with
    attraction (stay put) and medial guidance (pull toward Voronoi poles —
    approximate medial-axis targets inside the volume). The result is a thin
    meso-skeleton surface; :meth:`convert_to_skeleton` collapses it to a 1D
    curve and applies the refine phase.

    Prefer :func:`pymcfs.skeletonize` / :func:`pymcfs.contract_mesh` for the
    usual one-shot paths; use this class when you need per-iteration control.

    Parameters
    ----------
    mesh :
        Input ``trimesh.Trimesh``.
    attraction_weight :
        Attraction weight (default 0.5). Larger → stronger pull toward current
        positions (stability).
    medial_weight :
        Medial-centering weight (default 5.0). ``0`` disables Voronoi poles.
    gate_exterior_poles :
        If True (default), apply medial weight only when a pole lies inside the
        input mesh. Set False for Starlab-style ungated medial pull.
    fast_gating :
        Use the mesh's own ray backend for pole containment (Embree when
        ``pymcfs[embree]`` is installed) instead of the exact float64
        traverser. Roughly 100x faster, but Embree traces in single precision:
        on meshes with large absolute coordinates that flips most gating
        decisions. Only enable for meshes at unit-ish scale near the origin.
    use_cholmod :
        If True, require scikit-sparse CHOLMOD for the ``AᵀA`` solve. If False,
        force SciPy SuperLU. If None (default), use CHOLMOD when importable.
    min_edge_length, max_triangle_angle :
        Remesh controls during contraction. If ``min_edge_length`` is None,
        the effective threshold is ``0.002 * bbox_diagonal``.
    area_variation_factor :
        Relative area change vs initial area for convergence.
    max_iterations, timeout_seconds :
        Hard stop criteria.
    max_vertex_growth :
        Abort contraction when ``n > max_vertex_growth * n0`` (remesh blow-up).
        ``None`` or ``<= 0`` disables the guard. Default ``4.0`` (successful
        runs often reach ~2×; catastrophic blow-ups are 10–100×).
    pinned_attraction_floor :
        Numerical floor used when pinning fixed vertices
        (``attraction_weight = 1 / pinned_attraction_floor``) and as a
        short-edge epsilon in remesh.
    validate, verbose, log :
        Validation and logging.

    Attributes
    ----------
    aborted_remesh_growth :
        Set True when :meth:`contract_until_convergence` stops because
        vertex count exceeded ``max_vertex_growth * n0``.
    area_overshoot_seen :
        Set True if a sanity check observed surface area growing far beyond
        the initial area (numerical / remesh failure signal).

    Notes
    -----
    Laplacian scale ``w_L`` is fixed at 1 (CGAL uses the same). Cotangent edge
    weights are recomputed each geometry step. When ``gate_exterior_poles`` is
    on, exterior poles get medial weight 0 so they cannot pull the surface outside.
    High-level :func:`pymcfs.skeletonize` / :func:`pymcfs.contract_mesh` construct
    this driver with the default ``max_vertex_growth=4.0``.
    """

    mesh: tm.Trimesh
    attraction_weight: float = 0.5
    medial_weight: float = 5.0
    gate_exterior_poles: bool = True
    fast_gating: bool = False
    use_cholmod: bool | None = None
    min_edge_length: float | None = None
    max_triangle_angle: float = 110.0
    area_variation_factor: float = 1e-4
    max_iterations: int = 500
    timeout_seconds: float | None = 120.0
    max_vertex_growth: float | None = 4.0
    pinned_attraction_floor: float = 1e-7
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
    _pole_valid_dirty: bool = field(init=False, default=True, repr=False)
    _n_contains: int = field(init=False, default=0, repr=False)
    _min_edge: float = field(init=False, repr=False)
    _area0: float = field(init=False, repr=False)
    _n0: int = field(init=False, repr=False)
    _n_max: int = field(init=False, repr=False)
    _w_L: float = field(init=False, default=1.0, repr=False)
    _deadline: float | None = field(init=False, default=None, repr=False)
    _iter: int = field(init=False, default=0, repr=False)
    _bbox0: np.ndarray = field(init=False, repr=False)
    aborted_remesh_growth: bool = field(init=False, default=False, repr=False)
    area_overshoot_seen: bool = field(init=False, default=False, repr=False)

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
        self._n0 = int(n)
        self._n_max = int(n)
        self._w_L = 1.0
        self._iter = 0
        self._n_contains = 0
        self._pole_valid_dirty = True
        self.aborted_remesh_growth = False
        self.area_overshoot_seen = False
        self.pole_valid = np.zeros(n, dtype=bool)
        if float(self.medial_weight) > 0.0:
            try:
                targets, _w = compute_voronoi_poles(self.mesh)
                self.poles = np.asarray(targets, dtype=float)
                self.pole_valid = self._compute_pole_valid(self.poles)
                self._pole_valid_dirty = False
                n_valid = int(self.pole_valid.sum())
                if self.gate_exterior_poles:
                    self._vinfo(
                        "Voronoi poles: %d/%d inside mesh (gated; exterior medial=0)",
                        n_valid,
                        n,
                    )
                else:
                    self._vinfo(
                        "Voronoi poles: %d/%d inside mesh (ungated Starlab-style)",
                        n_valid,
                        n,
                    )
            except Exception as e:
                self._log.warning("Voronoi poles failed (%s); setting medial_weight effective 0", e)
                self.poles = self.V.copy()
                self.medial_weight = 0.0
                self._pole_valid_dirty = False
        else:
            self.poles = self.V.copy()
            self._pole_valid_dirty = False
        self._deadline = None
        from .spd_solve import cholmod_available, resolve_use_cholmod

        self._use_cholmod = resolve_use_cholmod(self.use_cholmod)
        self._vinfo(
            "MCFS init: n=%d f=%d min_edge=%.4g area0=%.4g bbox0=%s "
            "attraction=%.3g medial=%.3g gate_poles=%s spd=%s",
            n,
            self.F.shape[0],
            self._min_edge,
            self._area0,
            np.array2string(self._bbox0, precision=4),
            self.attraction_weight,
            self.medial_weight,
            bool(self.gate_exterior_poles),
            "cholmod" if self._use_cholmod else "superlu",
        )
        if self.verbose and self.use_cholmod is None and not cholmod_available():
            self._log.info(
                "SPD solver: superlu (install pymcfs[cholmod] / scikit-sparse for CHOLMOD)"
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
        """Return boolean mask of poles that lie in the interior of ``self.mesh``."""
        poles = np.asarray(poles, dtype=float)
        n = poles.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool)
        try:
            self._n_contains += 1
            inside = points_inside_mesh(self.mesh, poles, fast=self.fast_gating)
            if inside.shape[0] != n:
                return np.zeros(n, dtype=bool)
            return inside
        except Exception as e:
            self._log.warning("pole inside-test failed (%s); treating all poles as valid", e)
            return np.ones(n, dtype=bool)

    def _compute_pole_valid(self, poles: np.ndarray) -> np.ndarray:
        """Return pole validity mask used for medial-weight gating / diagnostics.

        When ``gate_exterior_poles`` is True (CGAL-style), this is the true
        containment mask. When False (Starlab parity), all poles are treated as
        valid for weighting; containment is still computed only if ``verbose``.
        """
        poles = np.asarray(poles, dtype=float)
        n = poles.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool)
        if self.gate_exterior_poles:
            return self._poles_inside_mesh(poles)
        if self.verbose:
            # Diagnostic logging path only; does not affect weights when ungated.
            return self._poles_inside_mesh(poles)
        return np.ones(n, dtype=bool)

    def _mark_poles_dirty(self) -> None:
        """Force a full ``pole_valid`` recompute on the next sync."""
        self._pole_valid_dirty = True

    def _sync_pole_valid(self) -> None:
        """Keep ``pole_valid`` aligned with ``poles``.

        Containment is a property of a fixed pole position against the fixed
        input mesh. Collapse carries validity by index (it keeps one of two
        existing poles) and split refreshes only the poles it interpolates, so
        this is a safety net: it re-tests every pole only when the arrays fall
        out of sync or something explicitly marked them dirty.
        """
        n = self.V.shape[0]
        if float(self.medial_weight) <= 0.0 or self.poles.shape[0] != n:
            self.pole_valid = np.zeros(n, dtype=bool)
            self._pole_valid_dirty = False
            return
        if not self._pole_valid_dirty and self.pole_valid.shape[0] == n:
            return
        self.pole_valid = self._compute_pole_valid(self.poles)
        self._pole_valid_dirty = False

    def _refresh_new_pole_valid(self, n_new: int) -> None:
        """Test containment for the ``n_new`` poles appended by an edge split.

        Edge splits interpolate genuinely new pole positions and only ever
        append, so the trailing ``n_new`` rows are the sole poles whose
        containment is not already known by index. One small batched test here
        replaces a whole-mesh ``contains`` call per remesh.
        """
        if n_new <= 0 or float(self.medial_weight) <= 0.0:
            return
        n = self.V.shape[0]
        if self.poles.shape[0] != n or self.pole_valid.shape[0] != n or n_new > n:
            self._mark_poles_dirty()
            self._sync_pole_valid()
            return
        self.pole_valid[-n_new:] = self._compute_pole_valid(self.poles[-n_new:])

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
        if float(self.medial_weight) > 0.0 and self.poles.shape[0] == n and n > 0:
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
            self.area_overshoot_seen = True
            self._log.warning(
                "sanity[%s]: area increased sharply %.4g -> %.4g (possible medial overshoot)",
                stage,
                prev_area,
                area,
            )
        if n > 0 and self._bbox0 is not None:
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
        wH = np.full(n, float(self.attraction_weight), dtype=float)
        wM = np.full(n, float(self.medial_weight), dtype=float)
        wL[self._constraint_fixed] = 0.0
        wH[self._constraint_fixed] = 1.0 / max(self.pinned_attraction_floor, 1e-16)
        wM[self._constraint_fixed] = 0.0
        wM[self._constraint_split] = 0.0
        # CGAL: apply medial weight only when pole is inside the input mesh.
        if self.gate_exterior_poles and self.pole_valid.shape[0] == n:
            wM[~self.pole_valid] = 0.0
        elif self.gate_exterior_poles:
            wM[:] = 0.0
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
        #
        # The lower two blocks are diagonal, so A.T @ A is
        # L_w.T @ L_w + diag(attraction²) + diag(medial²) and A.T @ B collapses to a
        # dense scaling. Building those directly avoids materialising the
        # (3n, n) stack, its transpose and the big sparse product. The terms
        # are combined in the same order the stacked product would use, which
        # keeps the result bit-identical.
        L = mcfs_cotangent_laplacian(self.V, self.F).tocsr()
        # Scale off-diagonals by omega_L in the CSR data; leave the diagonal as
        # the unweighted negative edge-weight sum (EigenContractionHelper).
        row_of = np.repeat(np.arange(L.shape[0]), np.diff(L.indptr))
        row_scale = wL[row_of]
        row_scale[L.indices == row_of] = 1.0
        L_weighted = sp.csr_matrix(
            (L.data * row_scale, L.indices, L.indptr), shape=L.shape
        )
        AtA = (
            (L_weighted.T @ L_weighted)
            + sp.diags(wH * wH, format="csr")
            + sp.diags(wM * wM, format="csr")
        ).tocsc()
        At_rhs = wH[:, None] * (wH[:, None] * self.V) + wM[:, None] * (
            wM[:, None] * self.poles
        )
        try:
            from .spd_solve import solve_spd_ata

            X, _backend = solve_spd_ata(AtA, At_rhs, use_cholmod=self._use_cholmod)
            self.V = X
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
        valid_in = self.pole_valid if self.pole_valid.shape[0] == n_before else None
        V2, F2, n, fixed2, poles2, valid2 = collapse_short_edges(
            self.V,
            self.F,
            min_edge_length=self._min_edge,
            fixed=self.fixed,
            poles=self.poles,
            pole_valid=valid_in,
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
            if valid2 is not None:
                self.pole_valid = valid2
            else:
                self._mark_poles_dirty()
            self._sync_pole_valid()
        if n:
            self._vinfo(
                "collapse_edges: %d collapses -> n=%d f=%d", n, self.V.shape[0], self.F.shape[0]
            )
        return int(n)

    def split_faces(self) -> int:
        """Split faces with an angle larger than ``max_triangle_angle``."""
        valid_in = self.pole_valid if self.pole_valid.shape[0] == self.V.shape[0] else None
        V2, F2, n, fixed2, poles2, valid2, split2 = split_obtuse_faces(
            self.V,
            self.F,
            max_angle_deg=self.max_triangle_angle,
            short_edge=max(self.pinned_attraction_floor, 1e-12),
            fixed=self.fixed,
            poles=self.poles,
            pole_valid=valid_in,
            is_split=self.is_split,
            deadline=self._deadline,
        )
        if n > 0:
            self.V, self.F = V2, F2
            self.fixed = fixed2 if fixed2 is not None else np.zeros(V2.shape[0], dtype=bool)
            self.poles = poles2 if poles2 is not None else self.V.copy()
            self.is_split = split2 if split2 is not None else np.zeros(V2.shape[0], dtype=bool)
            if valid2 is not None:
                self.pole_valid = valid2
                self._refresh_new_pole_valid(n)
            else:
                self._mark_poles_dirty()
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
        elength_fixed_sq = elength_fixed * elength_fixed
        topo = mesh_adjacency(self.F, self.V.shape[0])
        bad_count = np.zeros(self.V.shape[0], dtype=np.int32)
        V = self.V
        # Only a tiny fraction of edges are short enough to matter, so screen
        # them all at once. Ascending index order preserves the visit order
        # among the short edges, which is all `bad_count` depends on.
        eu = topo.edge_u[: topo.n_edges]
        ev = topo.edge_v[: topo.n_edges]
        diff = V[eu] - V[ev]
        d2 = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]
        short = np.flatnonzero(d2 < elength_fixed_sq)
        for ei in short:
            if self._timed_out():
                break
            a = int(eu[ei])
            b = int(ev[ei])
            if not collapse_ok_for_edge(
                a,
                b,
                V,
                self.F,
                check_connectivity=False,
                topo=topo,
            ):
                bad_count[a] += 1
                bad_count[b] += 1
        pinned = (bad_count >= 2) & ~self.fixed
        newly = int(pinned.sum())
        if newly:
            self.fixed[pinned] = True
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

    def remesh_growth_ratio(self) -> float:
        """Peak vertex count during contraction divided by the input size."""
        n0 = max(int(self._n0), 1)
        return float(self._n_max) / float(n0)

    def contract_until_convergence(self) -> int:
        """Iterate ``contract`` until area change is small, max iterations, or timeout.

        Also aborts when vertex count exceeds ``max_vertex_growth * n0`` (remesh
        blow-up from aggressive medial pull / obtuse splits).
        """
        if self.timeout_seconds is not None and self.timeout_seconds > 0:
            self._deadline = time.monotonic() + float(self.timeout_seconds)
        else:
            self._deadline = None
        prev_area = self._surface_area()
        last_it = 0
        growth_cap = self.max_vertex_growth
        self._log.info(
            "contract_until_convergence: start (max_iters=%d, n0=%d)",
            int(self.max_iterations),
            int(self._n0),
        )
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
            n = int(self.V.shape[0])
            if n > self._n_max:
                self._n_max = n
            if (
                growth_cap is not None
                and float(growth_cap) > 0.0
                and n > float(growth_cap) * float(self._n0)
            ):
                self.aborted_remesh_growth = True
                self._log.warning(
                    "stopping: remesh growth n=%d > %.3g * n0=%d at iter %d",
                    n,
                    float(growth_cap),
                    int(self._n0),
                    it,
                )
                break
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
        self._log.info(
            "contract_until_convergence: done (iters=%d, n=%d, f=%d)",
            last_it,
            int(self.V.shape[0]),
            int(self.F.shape[0]),
        )
        return last_it

    def meso_skeleton_mesh(self) -> tm.Trimesh:
        """Return the current contracted meso-skeleton as a ``Trimesh``."""
        return tm.Trimesh(vertices=self.V.copy(), faces=self.F.copy(), process=False)

    def convert_to_skeleton(
        self,
        *,
        resample: bool | str = False,
        resample_spacing: float | None = None,
        resample_spacing_frac: float | None = None,
        keep_largest_component: bool = False,
        prune_exterior: bool = True,
        prune_short_leaves: bool = True,
        short_leaf_scale: float = 1.0,
        prune_thick_hubs: bool = True,
        keep_hub_branches: int = 2,
        hub_degree_min: int = 4,
        hub_radius_frac: float = 0.015,
        extend_tips: bool = False,
        tip_extend_scale: float = 1.0,
        tip_clearance_frac: float = 0.01,
        tip_cone_deg: float = 40.0,
    ):
        """Convert the meso-skeleton surface into a 1D curve ``Skeleton``.

        Parameters
        ----------
        resample, resample_spacing, resample_spacing_frac :
            Optional curve-density resampling (see :func:`pymcfs.skeleton.skeletonize`).
        keep_largest_component :
            If True, keep only the largest connected component of the curve graph.
        prune_exterior :
            If True (default), remove dangling curve tips that lie outside the
            original input mesh.
        prune_short_leaves :
            If True (default), mild micro-spur prune
            (``L < short_leaf_scale × junction thickness``).
        short_leaf_scale :
            Length threshold multiplier for short-leaf pruning (default 1.0).
        prune_thick_hubs :
            If True (default), at thick high-degree hubs keep only the longest
            ``keep_hub_branches`` leaf arms (volume-star refine step).
        keep_hub_branches, hub_degree_min, hub_radius_frac :
            Thick-hub prune controls (see :func:`pymcfs.refine.prune_thick_hubs`).
        extend_tips :
            If True, grow unfinished leaf tips toward lobe ends (default False).
        tip_extend_scale :
            Max tip travel as a multiple of bbox diagonal when ``extend_tips``
            is True (default 1.0).
        tip_clearance_frac, tip_cone_deg :
            Tip-extension stop clearance and cone search half-angle.

        Returns
        -------
        Skeleton
        """
        from .refine import (
            resolve_resample_options,
            resample_skeleton_graph,
            prune_exterior_graph,
        )
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

        if prune_exterior and G.number_of_nodes() > 0:
            G, n_pruned = prune_exterior_graph(
                G, self.mesh, fast=bool(self.fast_gating)
            )
            if n_pruned:
                self._vinfo(
                    "convert_to_skeleton: pruned %d exterior dangling node(s)",
                    n_pruned,
                )

        if prune_short_leaves and G.number_of_nodes() > 0:
            from .refine import prune_short_leaves_graph

            G, n_short = prune_short_leaves_graph(
                G, self.mesh, length_scale=float(short_leaf_scale)
            )
            if n_short:
                self._vinfo(
                    "convert_to_skeleton: pruned %d short-leaf node(s) "
                    "(scale=%.3g)",
                    n_short,
                    float(short_leaf_scale),
                )

        if prune_thick_hubs and G.number_of_nodes() > 0:
            from .refine import prune_thick_hubs_graph

            G, n_hub = prune_thick_hubs_graph(
                G,
                self.mesh,
                keep_hub_branches=int(keep_hub_branches),
                hub_degree_min=int(hub_degree_min),
                hub_radius_frac=float(hub_radius_frac),
            )
            if n_hub:
                self._vinfo(
                    "convert_to_skeleton: pruned %d thick-hub node(s) "
                    "(keep=%d deg_min=%d)",
                    n_hub,
                    int(keep_hub_branches),
                    int(hub_degree_min),
                )

        if extend_tips and G.number_of_nodes() > 0:
            from .refine import extend_tips_graph

            G, n_ext = extend_tips_graph(
                G,
                self.mesh,
                tip_extend_scale=float(tip_extend_scale),
                tip_clearance_frac=float(tip_clearance_frac),
                tip_cone_deg=float(tip_cone_deg),
            )
            if n_ext:
                self._vinfo(
                    "convert_to_skeleton: extended tips by %d node(s) "
                    "(tip_extend_scale=%.3g)",
                    n_ext,
                    float(tip_extend_scale),
                )

        mode, spacing, spacing_frac = resolve_resample_options(
            resample=resample,
            resample_spacing=resample_spacing,
            resample_spacing_frac=resample_spacing_frac,
        )
        if mode is not None and G.number_of_nodes() > 0:
            n_before = G.number_of_nodes()
            G = resample_skeleton_graph(
                G, mode=mode, spacing=spacing, spacing_frac=spacing_frac
            )
            self._vinfo(
                "convert_to_skeleton: resample mode=%s %d -> %d nodes",
                mode,
                n_before,
                G.number_of_nodes(),
            )

        skel = Skeleton.from_graph(G)
        if skel.nodes.shape[0] > 0 and skel.edges.shape[0] > 0:
            total_len = float(
                sum(
                    np.linalg.norm(skel.nodes[int(u)] - skel.nodes[int(v)])
                    for u, v in skel.edges
                )
            )
            self._vinfo(
                "convert_to_skeleton: final nodes=%d edges=%d total_length=%.4g",
                skel.nodes.shape[0],
                skel.edges.shape[0],
                total_len,
            )
        else:
            self._log.warning("convert_to_skeleton: empty skeleton")
        return skel
