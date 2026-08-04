"""Local remeshing helpers for MCF skeletonization (CGAL / Starlab TopologyJanitor).

Implements short-edge collapse and obtuse-triangle splits on (V, F) triangle meshes.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def mesh_unique_edges(F: np.ndarray) -> np.ndarray:
    """Return unique undirected edges as (e, 2) int array from triangle faces."""
    if F.size == 0:
        return np.zeros((0, 2), dtype=int)
    e01 = F[:, [0, 1]]
    e12 = F[:, [1, 2]]
    e20 = F[:, [2, 0]]
    E = np.vstack([e01, e12, e20])
    E.sort(axis=1)
    return np.unique(E, axis=0)


def _vertex_neighbors(F: np.ndarray, nv: int) -> list[set[int]]:
    nbrs: list[set[int]] = [set() for _ in range(nv)]
    for i0, i1, i2 in F:
        nbrs[i0].update((int(i1), int(i2)))
        nbrs[i1].update((int(i0), int(i2)))
        nbrs[i2].update((int(i0), int(i1)))
    return nbrs


def _edge_to_faces(F: np.ndarray) -> dict[tuple[int, int], list[int]]:
    m: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(u), int(v)) if u < v else (int(v), int(u))
            m[key].append(fi)
    return m


def _link_condition_ok(
    a: int,
    b: int,
    neighbors: list[set[int]],
    edge_faces: dict[tuple[int, int], list[int]],
    F: np.ndarray,
) -> bool:
    """Manifold link condition for collapsing edge (a, b).

    Common 1-ring neighbors must equal the opposite vertices of faces incident
    to the edge (at most two on a manifold interior edge).
    """
    key = (a, b) if a < b else (b, a)
    faces = edge_faces.get(key, [])
    if len(faces) == 0:
        return False
    thirds: set[int] = set()
    for fi in faces:
        verts = {int(x) for x in F[fi]}
        verts.discard(a)
        verts.discard(b)
        if len(verts) != 1:
            return False
        thirds.add(verts.pop())
    common = neighbors[a].intersection(neighbors[b])
    return common == thirds and len(thirds) <= 2


def compact_mesh(
    V: np.ndarray,
    F: np.ndarray,
    *,
    extra: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Drop unreferenced vertices and reindex faces; remap parallel vertex arrays."""
    if F.size == 0:
        out_extra = {k: np.zeros((0,) + v.shape[1:], dtype=v.dtype) for k, v in (extra or {}).items()}
        return V[:0].copy(), F.copy(), out_extra
    used = np.unique(F.reshape(-1))
    remap = -np.ones(V.shape[0], dtype=int)
    remap[used] = np.arange(used.shape[0], dtype=int)
    Vn = V[used]
    Fn = remap[F]
    out_extra: dict[str, np.ndarray] = {}
    if extra:
        for k, arr in extra.items():
            out_extra[k] = arr[used]
    return Vn, Fn, out_extra


def collapse_short_edges(
    V: np.ndarray,
    F: np.ndarray,
    *,
    min_edge_length: float,
    fixed: np.ndarray | None = None,
    poles: np.ndarray | None = None,
    max_collapses: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None, np.ndarray | None]:
    """Collapse edges shorter than ``min_edge_length`` when the link condition holds.

    Matches Starlab/CGAL local remeshing: midpoint merge (closest pole kept when
    ``poles`` is provided). Vertices with ``fixed[i]`` are never moved into a
    collapse that would delete them if both endpoints are fixed.

    Returns
    -------
    V, F, n_collapsed, fixed_out, poles_out
    """
    V = np.asarray(V, dtype=float).copy()
    F = np.asarray(F, dtype=int).copy()
    if F.size == 0 or V.size == 0:
        return V, F, 0, fixed, poles

    nv = V.shape[0]
    if fixed is None:
        fixed = np.zeros(nv, dtype=bool)
    else:
        fixed = np.asarray(fixed, dtype=bool).copy()
    if poles is not None:
        poles = np.asarray(poles, dtype=float).copy()

    total = 0
    # Iterate until no short edge can collapse (Starlab iteratively_coolapseShortEdges)
    while True:
        nv = V.shape[0]
        edges = mesh_unique_edges(F)
        if edges.size == 0:
            break
        lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
        order = np.argsort(lengths)
        neighbors = _vertex_neighbors(F, nv)
        edge_faces = _edge_to_faces(F)
        parent = np.arange(nv, dtype=int)
        alive = np.ones(nv, dtype=bool)
        collapsed_this = 0

        def find(x: int) -> int:
            while parent[x] != parent[parent[x]]:
                parent[x] = parent[parent[x]]
            return parent[x]

        for ei in order:
            if max_collapses is not None and total + collapsed_this >= max_collapses:
                break
            a0, b0 = int(edges[ei, 0]), int(edges[ei, 1])
            if not alive[a0] or not alive[b0]:
                continue
            if parent[a0] != a0 or parent[b0] != b0:
                continue
            a, b = a0, b0
            d = float(np.linalg.norm(V[a] - V[b]))
            if d >= min_edge_length:
                continue
            if fixed[a] and fixed[b]:
                continue
            if not _link_condition_ok(a, b, neighbors, edge_faces, F):
                continue
            # Midpoint (or keep fixed endpoint)
            if fixed[a] and not fixed[b]:
                mid = V[a].copy()
                keep, drop = a, b
            elif fixed[b] and not fixed[a]:
                mid = V[b].copy()
                keep, drop = b, a
            else:
                mid = 0.5 * (V[a] + V[b])
                keep, drop = a, b
            # Do not dissolve the last faces during local remesh — leave that to
            # convert_to_skeleton (keeps a meso-skeleton surface for conversion).
            Fm = F.copy()
            Fm[Fm == drop] = keep
            alive_f = (Fm[:, 0] != Fm[:, 1]) & (Fm[:, 1] != Fm[:, 2]) & (Fm[:, 2] != Fm[:, 0])
            if int(alive_f.sum()) == 0 and F.shape[0] > 0:
                continue
            if poles is not None:
                pa, pb = poles[keep], poles[drop]
                if np.linalg.norm(mid - pa) <= np.linalg.norm(mid - pb):
                    poles[keep] = pa
                else:
                    poles[keep] = pb
            V[keep] = mid
            parent[drop] = keep
            alive[drop] = False
            fixed[keep] = bool(fixed[keep] or fixed[drop])
            collapsed_this += 1
            # One topology-changing collapse per pass (Starlab updates halfedges
            # immediately; we rebuild adjacency on the next outer iteration).
            break

        if collapsed_this == 0:
            break

        # Compact
        for i in range(nv):
            while parent[i] != parent[parent[i]]:
                parent[i] = parent[parent[i]]
        Fm = parent[F]
        keep_f = (Fm[:, 0] != Fm[:, 1]) & (Fm[:, 1] != Fm[:, 2]) & (Fm[:, 2] != Fm[:, 0])
        Fm = Fm[keep_f]
        roots = np.unique(parent[alive])
        new_index = -np.ones(nv, dtype=int)
        new_index[roots] = np.arange(len(roots), dtype=int)
        V = V[roots]
        F = new_index[Fm]
        fixed = fixed[roots]
        if poles is not None:
            poles = poles[roots]
        total += collapsed_this

    return V, F, total, fixed, poles


def split_obtuse_faces(
    V: np.ndarray,
    F: np.ndarray,
    *,
    max_angle_deg: float = 110.0,
    short_edge: float = 1e-12,
    fixed: np.ndarray | None = None,
    poles: np.ndarray | None = None,
    is_split: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Split edges opposite angles larger than ``max_angle_deg`` (Starlab splitter).

    Projects the obtuse vertex onto the opposite edge and inserts a new vertex.
    """
    V = np.asarray(V, dtype=float).copy()
    F = np.asarray(F, dtype=int).copy()
    if F.size == 0:
        return V, F, 0, fixed, poles, is_split

    nv = V.shape[0]
    if fixed is None:
        fixed = np.zeros(nv, dtype=bool)
    else:
        fixed = np.asarray(fixed, dtype=bool).copy()
    if is_split is None:
        is_split = np.zeros(nv, dtype=bool)
    else:
        is_split = np.asarray(is_split, dtype=bool).copy()
    if poles is not None:
        poles = np.asarray(poles, dtype=float).copy()

    thr = float(max_angle_deg) * (np.pi / 180.0)
    total = 0

    while True:
        new_faces: list[np.ndarray] = []
        splits = 0
        # Track edges already split this pass
        split_edges: set[tuple[int, int]] = set()
        V_list = [V[i].copy() for i in range(V.shape[0])]
        fixed_list = list(fixed)
        split_list = list(is_split)
        poles_list = [poles[i].copy() for i in range(poles.shape[0])] if poles is not None else None

        for face in F:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            p0, p1, p2 = V_list[i0], V_list[i1], V_list[i2]
            a = float(np.linalg.norm(p1 - p2))
            b = float(np.linalg.norm(p2 - p0))
            c = float(np.linalg.norm(p0 - p1))
            if a < short_edge or b < short_edge or c < short_edge:
                new_faces.append(np.array([i0, i1, i2], dtype=int))
                continue
            # Angles at vertices
            cos0 = np.clip((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0)
            cos1 = np.clip((c * c + a * a - b * b) / (2.0 * c * a), -1.0, 1.0)
            cos2 = np.clip((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)
            ang = (float(np.arccos(cos0)), float(np.arccos(cos1)), float(np.arccos(cos2)))
            k = int(np.argmax(ang))
            if ang[k] <= thr:
                new_faces.append(np.array([i0, i1, i2], dtype=int))
                continue
            # Obtuse at vertex verts[k]; opposite edge is the other two
            verts = (i0, i1, i2)
            obtuse = verts[k]
            e0, e1 = verts[(k + 1) % 3], verts[(k + 2) % 3]
            if fixed_list[e0] and fixed_list[e1]:
                new_faces.append(np.array([i0, i1, i2], dtype=int))
                continue
            ek = (e0, e1) if e0 < e1 else (e1, e0)
            if ek in split_edges:
                new_faces.append(np.array([i0, i1, i2], dtype=int))
                continue
            # Project obtuse vertex onto opposite edge
            p_a, p_b = V_list[e0], V_list[e1]
            edge = p_b - p_a
            elen2 = float(np.dot(edge, edge))
            if elen2 <= 0:
                new_faces.append(np.array([i0, i1, i2], dtype=int))
                continue
            t = float(np.dot(V_list[obtuse] - p_a, edge) / elen2)
            t = float(np.clip(t, 1e-6, 1.0 - 1e-6))
            newpos = p_a + t * edge
            new_idx = len(V_list)
            V_list.append(newpos)
            fixed_list.append(False)
            split_list.append(True)
            if poles_list is not None:
                # Split verts get no medial pull (Starlab vissplit => omega_P=0);
                # store midpoint of endpoint poles as placeholder.
                poles_list.append(0.5 * (poles_list[e0] + poles_list[e1]))
            split_edges.add(ek)
            # Replace face with two triangles
            new_faces.append(np.array([obtuse, e0, new_idx], dtype=int))
            new_faces.append(np.array([obtuse, new_idx, e1], dtype=int))
            splits += 1

        V = np.asarray(V_list, dtype=float)
        F = np.asarray(new_faces, dtype=int) if new_faces else np.zeros((0, 3), dtype=int)
        fixed = np.asarray(fixed_list, dtype=bool)
        is_split = np.asarray(split_list, dtype=bool)
        if poles_list is not None:
            poles = np.asarray(poles_list, dtype=float)
        total += splits
        if splits == 0:
            break

    return V, F, total, fixed, poles, is_split


def collapse_ok_for_edge(
    a: int,
    b: int,
    V: np.ndarray,
    F: np.ndarray,
) -> bool:
    """Return True if collapsing edge (a,b) satisfies the link condition."""
    nv = V.shape[0]
    neighbors = _vertex_neighbors(F, nv)
    edge_faces = _edge_to_faces(F)
    return _link_condition_ok(int(a), int(b), neighbors, edge_faces, F)
