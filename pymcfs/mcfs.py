"""CGAL-style mean curvature flow skeletonization driver.

Mirrors ``CGAL::Mean_curvature_flow_skeletonization``:
contract_geometry → collapse_edges → split_faces → detect_degeneracies,
then convert_to_skeleton.
"""
from __future__ import annotations

import logging
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
    mesh_unique_edges,
    split_obtuse_faces,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def meso_surface_to_curve_graph(V: np.ndarray, F: np.ndarray) -> nx.Graph:
    """Collapse face-bearing edges of a meso-skeleton into a 1D curve graph.

    Follows Starlab ``surfacemesh_to_skeleton`` / CGAL ``convert_to_skeleton``:
    repeatedly collapse the shortest face-bearing edge. Final connectivity is the
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
    max_guard = max(40 * max(n0, 1), 4000)
    Fcur = F.copy()
    while Fcur.shape[0] > 0 and guard < max_guard:
        guard += 1
        # Rewrite faces through current parents
        Fr = np.vectorize(find, otypes=[int])(Fcur)
        keep = (Fr[:, 0] != Fr[:, 1]) & (Fr[:, 1] != Fr[:, 2]) & (Fr[:, 2] != Fr[:, 0])
        Fcur = Fr[keep]
        if Fcur.shape[0] == 0:
            break

        edges = mesh_unique_edges(Fcur)
        lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
        order = np.argsort(lengths)

        collapsed = False
        for ei in order:
            a, b = int(edges[ei, 0]), int(edges[ei, 1])
            a, b = find(a), find(b)
            if a == b:
                continue
            if collapse_ok_for_edge(a, b, V, Fcur):
                union(a, b)
                collapsed = True
                break
        if collapsed:
            continue
        # Force-collapse shortest face edge (sheet / pinch), as in Starlab to_skeleton
        a, b = int(edges[order[0], 0]), int(edges[order[0], 1])
        a, b = find(a), find(b)
        if a == b:
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

    if initial_edges.size:
        for a, b in initial_edges:
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

    # Drop isolated nodes that never participated in an edge
    isolates = [n for n, d in G.degree() if d == 0]
    G.remove_nodes_from(isolates)
    # Relabel densely
    if G.number_of_nodes() > 0:
        mapping = {n: i for i, n in enumerate(G.nodes)}
        G = nx.relabel_nodes(G, mapping, copy=True)
    return G


@dataclass
class MeanCurvatureFlowSkeletonization:
    """Python MCFS driver following CGAL / Starlab iteration structure."""

    mesh: tm.Trimesh
    omega_L: float = 1.0
    omega_H: float = 0.1
    omega_P: float = 0.2
    is_medially_centered: bool = True
    min_edge_length: float | None = None
    max_triangle_angle: float = 110.0
    area_variation_factor: float = 1e-4
    max_iterations: int = 500
    zero_TH: float = 1e-7
    laplacian_secure: bool = True
    verbose: bool = False
    log: logging.Logger | None = None

    V: np.ndarray = field(init=False, repr=False)
    F: np.ndarray = field(init=False, repr=False)
    fixed: np.ndarray = field(init=False, repr=False)
    is_split: np.ndarray = field(init=False, repr=False)
    poles: np.ndarray = field(init=False, repr=False)
    _min_edge: float = field(init=False, repr=False)
    _area0: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, tm.Trimesh):
            raise TypeError("mesh must be a trimesh.Trimesh")
        self._log = self.log or logger
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
        if self.is_medially_centered and self.omega_P > 0:
            try:
                targets, _w = compute_voronoi_poles(self.mesh)
                self.poles = np.asarray(targets, dtype=float)
            except Exception as e:
                self._log.warning("Voronoi poles failed (%s); disabling medial term", e)
                self.poles = self.V.copy()
                self.is_medially_centered = False
        else:
            self.poles = self.V.copy()
        if self.verbose:
            self._log.info(
                "MCFS init: n=%d f=%d min_edge=%.4g area0=%.4g medial=%s",
                n,
                self.F.shape[0],
                self._min_edge,
                self._area0,
                self.is_medially_centered,
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
        wL = np.full(n, float(self.omega_L), dtype=float)
        wH = np.full(n, float(self.omega_H), dtype=float)
        wP = np.full(n, float(self.omega_P) if self.is_medially_centered else 0.0, dtype=float)
        wL[self.fixed] = 0.0
        wH[self.fixed] = 1.0 / max(self.zero_TH, 1e-16)
        wP[self.fixed] = 0.0
        wP[self.is_split] = 0.0
        return wL, wH, wP

    def contract_geometry(self) -> None:
        """One contraction solve; reassembles Laplacian from current connectivity."""
        if self.V.shape[0] == 0 or self.F.shape[0] == 0:
            return
        L = cotangent_laplacian(self.V, self.F, secure=self.laplacian_secure)
        wL, wH, wP = self._update_constraint_weights()
        A = (sp.diags(wL) @ L + sp.diags(wH) + sp.diags(wP)).tocsc()
        rhs = sp.diags(wH) @ self.V + sp.diags(wP) @ self.poles
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
        V2, F2, n, fixed2, poles2 = collapse_short_edges(
            self.V,
            self.F,
            min_edge_length=self._min_edge,
            fixed=self.fixed,
            poles=self.poles,
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
        V2, F2, n, fixed2, poles2, split2 = split_obtuse_faces(
            self.V,
            self.F,
            max_angle_deg=self.max_triangle_angle,
            short_edge=max(self.zero_TH, 1e-12),
            fixed=self.fixed,
            poles=self.poles,
            is_split=self.is_split,
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

    def detect_degeneracies(self) -> int:
        """Pin vertices whose local neighborhood is no longer a disk (Starlab)."""
        if self.V.shape[0] == 0 or self.F.shape[0] == 0:
            return 0
        elength_fixed = self._min_edge / 10.0
        edges = mesh_unique_edges(self.F)
        incident: list[list[tuple[int, int]]] = [[] for _ in range(self.V.shape[0])]
        for a, b in edges:
            a, b = int(a), int(b)
            incident[a].append((a, b))
            incident[b].append((a, b))
        newly = 0
        for v in range(self.V.shape[0]):
            if self.fixed[v]:
                continue
            bad = 0
            for a, b in incident[v]:
                d = float(np.linalg.norm(self.V[a] - self.V[b]))
                if d < elength_fixed and not collapse_ok_for_edge(a, b, self.V, self.F):
                    bad += 1
            if bad >= 2:
                self.fixed[v] = True
                newly += 1
        if self.verbose and newly:
            self._log.info(
                "detect_degeneracies: pinned %d (total fixed=%d)", newly, int(self.fixed.sum())
            )
        return newly

    def contract(self) -> None:
        """One full CGAL iteration: geometry + collapse + split + degeneracies."""
        self.contract_geometry()
        self.collapse_edges()
        self.split_faces()
        self.detect_degeneracies()

    def contract_until_convergence(self) -> int:
        """Iterate ``contract`` until area change is small or max iterations."""
        prev_area = self._surface_area()
        last_it = 0
        for it in range(1, int(self.max_iterations) + 1):
            last_it = it
            self.contract()
            area = self._surface_area()
            if prev_area > 0 and abs(prev_area - area) < self.area_variation_factor * max(self._area0, 1e-30):
                if self.verbose:
                    self._log.info("converged at iter %d area=%.4g", it, area)
                break
            if self._area0 > 0 and area < 1e-3 * self._area0:
                if self.verbose:
                    self._log.info("stopping: area collapsed (%.4g) at iter %d", area, it)
                break
            if self.F.shape[0] <= max(4, self._faces0 // 100):
                if self.verbose:
                    self._log.info("stopping: few faces left (%d) at iter %d", self.F.shape[0], it)
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
    ):
        """Convert the meso-skeleton surface into a 1D curve ``Skeleton``."""
        from .skeleton import Skeleton, _compress_degree_two_chains, _resample_edges_uniform

        G = meso_surface_to_curve_graph(self.V, self.F)
        if compress_chains and G.number_of_nodes() > 0:
            G = _compress_degree_two_chains(G)
        if resample_spacing is not None and resample_spacing > 0 and G.number_of_edges() > 0:
            G = _resample_edges_uniform(G, float(resample_spacing))

        node_index = {n: i for i, n in enumerate(G.nodes)}
        nodes_arr = (
            np.array([G.nodes[n]["pos"] for n in G.nodes], dtype=float)
            if G.number_of_nodes()
            else np.zeros((0, 3))
        )
        edges_arr = (
            np.array([[node_index[u], node_index[v]] for u, v in G.edges], dtype=int)
            if G.number_of_edges()
            else np.zeros((0, 2), dtype=int)
        )
        return Skeleton(nodes=nodes_arr, edges=edges_arr, graph=G)
