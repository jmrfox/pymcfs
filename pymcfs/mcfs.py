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

from .laplacian import cotangent_laplacian
from .medial import compute_voronoi_poles
from .remesh import (
    collapse_ok_for_edge,
    collapse_short_edges,
    face_graph_components,
    is_vertex_degenerate,
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
    """Collapse face-bearing edges of a meso-skeleton into a 1D curve graph.

    Repeatedly collapse the shortest face-bearing edge. Final connectivity is the
    original mesh 1-skeleton remapped through the union-find of collapses, so
    curve topology is preserved even after all faces disappear.
    """
    V = np.asarray(V, dtype=float).copy()
    F = np.asarray(F, dtype=int).copy()
    n0 = V.shape[0]
    if n0 == 0:
        return nx.Graph()

    parent = np.arange(n0, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        a, b = find(a), find(b)
        if a == b:
            return a
        V[a] = 0.5 * (V[a] + V[b])
        parent[b] = a
        return a

    # Record the initial 1-skeleton (all mesh edges)
    initial_edges = mesh_unique_edges(F) if F.size else np.zeros((0, 2), dtype=int)

    guard = 0
    if max_steps is None:
        max_steps = max(20 * max(n0, 1), 2000)
    Fcur = F.copy()
    while Fcur.shape[0] > 0 and guard < max_steps:
        if deadline is not None and time.monotonic() >= deadline:
            break
        guard += 1
        # Rewrite faces through current parents
        Fr = np.vectorize(find, otypes=[int])(Fcur)
        keep = (Fr[:, 0] != Fr[:, 1]) & (Fr[:, 1] != Fr[:, 2]) & (Fr[:, 2] != Fr[:, 0])
        Fcur = Fr[keep]
        if Fcur.shape[0] == 0:
            break

        edges = mesh_unique_edges(Fcur)
        if edges.size == 0:
            break
        lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
        ei = int(np.argmin(lengths))
        a, b = int(edges[ei, 0]), int(edges[ei, 1])
        a, b = find(a), find(b)
        if a == b:
            # Degenerate residual; drop this face edge by forcing face rewrite next
            # round via a no-op face filter — break to avoid spinning.
            break
        union(a, b)

    # Build curve graph from initial edges remapped to surviving roots
    G = nx.Graph()
    root_to_node: dict[int, int] = {}
    for i in range(n0):
        r = find(i)
        if r not in root_to_node:
            nid = len(root_to_node)
            root_to_node[r] = nid
            G.add_node(nid, pos=V[r].copy())

    # Prefer remapped initial edges; also include any residual face edges
    edge_src = initial_edges
    if Fcur.shape[0] > 0:
        resid = mesh_unique_edges(Fcur)
        if resid.size:
            edge_src = resid if edge_src.size == 0 else np.vstack([edge_src, resid])

    if edge_src.size:
        for a, b in edge_src:
            ra, rb = find(int(a)), find(int(b))
            if ra == rb:
                continue
            u, v = root_to_node[ra], root_to_node[rb]
            w = float(np.linalg.norm(V[ra] - V[rb]))
            if G.has_edge(u, v):
                if w < G[u][v]["weight"]:
                    G[u][v]["weight"] = w
            else:
                G.add_edge(u, v, weight=w)

    isolates = [n for n, d in G.degree() if d == 0]
    G.remove_nodes_from(isolates)
    if G.number_of_nodes() > 0:
        mapping = {n: i for i, n in enumerate(G.nodes)}
        G = nx.relabel_nodes(G, mapping, copy=True)
    return G


@dataclass
class MeanCurvatureFlowSkeletonization:
    """Mean-curvature-flow skeletonization driver.

    Public contraction weights:
    - ``w_H`` = quality/speed tradeoff (default 0.1)
    - ``w_M`` = medial-centering tradeoff (default 0.2)
    - ``w_L`` is fixed at 1 (scale invariance / partition of unity over multiplication)
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
    laplacian_secure: bool = True
    validate: bool = True
    verbose: bool = False
    log: logging.Logger | None = None

    V: np.ndarray = field(init=False, repr=False)
    F: np.ndarray = field(init=False, repr=False)
    fixed: np.ndarray = field(init=False, repr=False)
    is_split: np.ndarray = field(init=False, repr=False)
    poles: np.ndarray = field(init=False, repr=False)
    _min_edge: float = field(init=False, repr=False)
    _area0: float = field(init=False, repr=False)
    _w_L: float = field(init=False, default=1.0, repr=False)
    _deadline: float | None = field(init=False, default=None, repr=False)

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
        bb = self.V.max(axis=0) - self.V.min(axis=0)
        diag = float(np.linalg.norm(bb))
        self._min_edge = (
            float(self.min_edge_length)
            if self.min_edge_length is not None
            else max(diag * 0.002, 1e-12)
        )
        self._area0 = self._surface_area()
        self._faces0 = int(self.F.shape[0])
        self._w_L = 1.0
        # Medial term on whenever w_M > 0
        if float(self.w_M) > 0.0:
            try:
                targets, _w = compute_voronoi_poles(self.mesh)
                self.poles = np.asarray(targets, dtype=float)
            except Exception as e:
                self._log.warning("Voronoi poles failed (%s); setting w_M effective 0", e)
                self.poles = self.V.copy()
                self.w_M = 0.0
        else:
            self.poles = self.V.copy()
        self._deadline = None
        if self.verbose:
            self._log.info(
                "MCFS init: n=%d f=%d min_edge=%.4g area0=%.4g w_H=%.3g w_M=%.3g",
                n,
                self.F.shape[0],
                self._min_edge,
                self._area0,
                self.w_H,
                self.w_M,
            )

    def _surface_area(self) -> float:
        if self.F.size == 0:
            return 0.0
        v0 = self.V[self.F[:, 0]]
        v1 = self.V[self.F[:, 1]]
        v2 = self.V[self.F[:, 2]]
        return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())

    def _update_constraint_weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.V.shape[0]
        wL = np.full(n, float(self._w_L), dtype=float)
        wH = np.full(n, float(self.w_H), dtype=float)
        wM = np.full(n, float(self.w_M), dtype=float)
        wL[self.fixed] = 0.0
        wH[self.fixed] = 1.0 / max(self.zero_TH, 1e-16)
        wM[self.fixed] = 0.0
        wM[self.is_split] = 0.0
        return wL, wH, wM

    def contract_geometry(self) -> None:
        """One contraction solve; reassembles Laplacian from current connectivity."""
        if self.V.shape[0] == 0 or self.F.shape[0] == 0:
            return
        L = cotangent_laplacian(self.V, self.F, secure=self.laplacian_secure)
        wL, wH, wM = self._update_constraint_weights()
        A = (sp.diags(wL) @ L + sp.diags(wH) + sp.diags(wM)).tocsc()
        rhs = sp.diags(wH) @ self.V + sp.diags(wM) @ self.poles
        try:
            solver = spla.factorized(A)
            for c in range(3):
                self.V[:, c] = solver(rhs[:, c])
        except Exception as e:
            self._log.warning("MCFS contract_geometry solve failed: %s; using lstsq", e)
            AtA = (A.T @ A).tocsc()
            for c in range(3):
                self.V[:, c] = spla.spsolve(AtA, A.T @ rhs[:, c])

    def collapse_edges(self) -> int:
        """Collapse edges shorter than ``min_edge_length``."""
        # Keep enough faces for a meaningful meso-skeleton; conversion handles the rest.
        if self.F.shape[0] <= max(4, self._faces0 // 100):
            return 0
        n_before = self.V.shape[0]
        per_iter_cap = max(50, self.V.shape[0] // 2)
        V2, F2, n, fixed2, poles2 = collapse_short_edges(
            self.V,
            self.F,
            min_edge_length=self._min_edge,
            fixed=self.fixed,
            poles=self.poles,
            max_collapses=per_iter_cap,
            max_passes=per_iter_cap,
            deadline=self._deadline,
        )
        if n > 0 or V2.shape[0] != n_before:
            self.V, self.F = V2, F2
            self.fixed = fixed2 if fixed2 is not None else np.zeros(V2.shape[0], dtype=bool)
            self.poles = poles2 if poles2 is not None else self.V.copy()
            # Resize is_split (collapsed verts removed; new count)
            if self.is_split.shape[0] != self.V.shape[0]:
                self.is_split = np.zeros(self.V.shape[0], dtype=bool)
        if self.verbose and n:
            self._log.info(
                "collapse_edges: %d collapses → n=%d f=%d", n, self.V.shape[0], self.F.shape[0]
            )
        return int(n)

    def split_faces(self) -> int:
        """Split faces with an angle larger than ``max_triangle_angle``."""
        per_iter_cap = max(50, self.V.shape[0] // 2)
        V2, F2, n, fixed2, poles2, split2 = split_obtuse_faces(
            self.V,
            self.F,
            max_angle_deg=self.max_triangle_angle,
            short_edge=max(self.zero_TH, 1e-12),
            fixed=self.fixed,
            poles=self.poles,
            is_split=self.is_split,
            max_passes=per_iter_cap,
            deadline=self._deadline,
        )
        if n > 0:
            self.V, self.F = V2, F2
            self.fixed = fixed2 if fixed2 is not None else np.zeros(V2.shape[0], dtype=bool)
            self.poles = poles2 if poles2 is not None else self.V.copy()
            self.is_split = split2 if split2 is not None else np.zeros(V2.shape[0], dtype=bool)
        if self.verbose and n:
            self._log.info(
                "split_faces: %d splits → n=%d f=%d", n, self.V.shape[0], self.F.shape[0]
            )
        return int(n)

    def _timed_out(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    def detect_degeneracies(self) -> int:
        """Pin vertices whose local neighborhood is no longer a disk."""
        if self.V.shape[0] == 0 or self.F.shape[0] == 0:
            return 0
        elength_fixed = self._min_edge / 10.0
        disk_radius = self._min_edge
        edges = mesh_unique_edges(self.F)
        from .remesh import _vertex_neighbors

        neighbors = _vertex_neighbors(self.F, self.V.shape[0])
        # Only consider vertices that touch at least one short edge (cheap filter).
        short_touch = np.zeros(self.V.shape[0], dtype=bool)
        incident: list[list[tuple[int, int]]] = [[] for _ in range(self.V.shape[0])]
        for a, b in edges:
            a, b = int(a), int(b)
            incident[a].append((a, b))
            incident[b].append((a, b))
            d = float(np.linalg.norm(self.V[a] - self.V[b]))
            if d < disk_radius:
                short_touch[a] = True
                short_touch[b] = True

        newly = 0
        for v in range(self.V.shape[0]):
            if self._timed_out():
                break
            if self.fixed[v] or not short_touch[v]:
                continue
            bad = 0
            for a, b in incident[v]:
                d = float(np.linalg.norm(self.V[a] - self.V[b]))
                # Link-only check (halfedge is_collapse_ok); skip connectivity scan.
                if d < elength_fixed and not collapse_ok_for_edge(
                    a, b, self.V, self.F, check_connectivity=False
                ):
                    bad += 1
            disk_bad = False
            if bad < 2 and neighbors[v]:
                try:
                    disk_bad = is_vertex_degenerate(
                        v, self.V, self.F, radius=disk_radius, neighbors=neighbors
                    )
                except Exception:
                    disk_bad = False
            if bad >= 2 or disk_bad:
                self.fixed[v] = True
                newly += 1
        if self.verbose and newly:
            self._log.info(
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
        self.split_faces()
        if self._timed_out():
            return
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
            if self._timed_out():
                if self.verbose:
                    self._log.info(
                        "stopping: timeout after %.3gs at iter %d",
                        float(self.timeout_seconds or 0.0),
                        it - 1,
                    )
                break
            self.contract()
            if self._timed_out():
                if self.verbose:
                    self._log.info(
                        "stopping: timeout after %.3gs during iter %d",
                        float(self.timeout_seconds or 0.0),
                        it,
                    )
                break
            area = self._surface_area()
            if prev_area > 0 and abs(prev_area - area) < self.area_variation_factor * max(
                self._area0, 1e-30
            ):
                if self.verbose:
                    self._log.info("converged at iter %d area=%.4g", it, area)
                break
            # Soft floor: only stop on tiny area once some vertices are pinned.
            if (
                self._area0 > 0
                and area < 1e-4 * self._area0
                and int(self.fixed.sum()) > 0
            ):
                if self.verbose:
                    self._log.info(
                        "stopping: area collapsed (%.4g) with %d fixed at iter %d",
                        area,
                        int(self.fixed.sum()),
                        it,
                    )
                break
            if self.F.shape[0] <= max(4, self._faces0 // 100):
                if self.verbose:
                    self._log.info("stopping: few faces left (%d) at iter %d", self.F.shape[0], it)
                break
            if face_graph_components(self.F) > 1:
                if self.verbose:
                    self._log.info(
                        "stopping: meso-skeleton fragmented (%d face components) at iter %d",
                        face_graph_components(self.F),
                        it,
                    )
                break
            prev_area = area
            if self.F.shape[0] == 0:
                break
            if self.verbose and (it % max(1, self.max_iterations // 10) == 0):
                self._log.info(
                    "iter %d: n=%d f=%d area=%.4g fixed=%d",
                    it,
                    self.V.shape[0],
                    self.F.shape[0],
                    area,
                    int(self.fixed.sum()),
                )
        return last_it

    def meso_skeleton_mesh(self) -> tm.Trimesh:
        return tm.Trimesh(vertices=self.V.copy(), faces=self.F.copy(), process=False)

    def convert_to_skeleton(
        self,
        *,
        compress_chains: bool = True,
        resample_spacing: float | None = None,
        keep_largest_component: bool = True,
    ):
        """Convert the meso-skeleton surface into a 1D curve ``Skeleton``."""
        from .skeleton import Skeleton, _compress_degree_two_chains, _resample_edges_uniform

        G = meso_surface_to_curve_graph(
            self.V, self.F, deadline=self._deadline
        )
        if G.number_of_nodes() > 1 and nx.number_connected_components(G) > 1:
            G = _stitch_skeleton_components(G, max_bridge=float(np.linalg.norm(
                self.V.max(axis=0) - self.V.min(axis=0)
            )) * 0.05)
            if self.verbose:
                self._log.info(
                    "convert_to_skeleton: after stitch cc=%d nodes=%d",
                    nx.number_connected_components(G) if G.number_of_nodes() else 0,
                    G.number_of_nodes(),
                )
        if keep_largest_component and G.number_of_nodes() > 0:
            comps = list(nx.connected_components(G))
            if len(comps) > 1:
                largest = max(comps, key=len)
                if self.verbose:
                    self._log.info(
                        "convert_to_skeleton: keeping largest of %d components (%d nodes)",
                        len(comps),
                        len(largest),
                    )
                G = G.subgraph(largest).copy()
        if compress_chains and G.number_of_nodes() > 0:
            G = _compress_degree_two_chains(G)
        if resample_spacing is not None and resample_spacing > 0 and G.number_of_edges() > 0:
            G = _resample_edges_uniform(G, float(resample_spacing))

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
        return Skeleton(nodes=nodes_arr, edges=edges_arr, graph=G)
