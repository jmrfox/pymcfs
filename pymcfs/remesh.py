"""Local remeshing helpers used by the MCFS driver (internal).

Implements short-edge collapse and obtuse-triangle splits on (V, F) triangle meshes.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from .topology import (
    MeshTopology,
    build_topology,
    link_condition_ok,
    pair_obtuse_edges,
    split_face_on_edge_numba,
    topology_collapse_buffers,
)

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


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def mesh_adjacency(F: np.ndarray, nv: int) -> MeshTopology:
    """Return array-backed mesh topology for ``F``."""
    return build_topology(np.asarray(F, dtype=np.int32), nv)


def _vertex_neighbors(F: np.ndarray, nv: int) -> list[set[int]]:
    nbrs: list[set[int]] = [set() for _ in range(nv)]
    for i0, i1, i2 in F:
        a, b, c = int(i0), int(i1), int(i2)
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    return nbrs


def _edge_to_faces(F: np.ndarray) -> dict[tuple[int, int], set[int]]:
    m: dict[tuple[int, int], set[int]] = defaultdict(set)
    for fi, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            m[_edge_key(int(u), int(v))].add(fi)
    return m


def _faces_incident_to_vertex(
    v: int,
    neighbors: list[set[int]],
    edge_faces: dict[tuple[int, int], set[int]],
) -> set[int]:
    out: set[int] = set()
    for nbr in neighbors[v]:
        out.update(edge_faces.get(_edge_key(v, nbr), ()))
    return out


def _link_condition_numba(
    a: int,
    b: int,
    F: np.ndarray,
    face_alive: np.ndarray,
    vert_alive: np.ndarray,
    nv: int,
) -> bool:
    """Numba link test on a read-only topology snapshot of ``F``."""
    topo = build_topology(
        np.asarray(F, dtype=np.int32),
        nv,
        face_alive=face_alive,
        vert_alive=vert_alive,
    )
    edge_u, edge_v, edge_f0, edge_f1, n_edges, edge_cap = topology_collapse_buffers(
        topo, face_count=F.shape[0]
    )
    _ = edge_u, edge_v, n_edges, edge_cap
    return bool(
        link_condition_ok(
            np.int32(a),
            np.int32(b),
            topo.F,
            topo.face_alive,
            topo.nbr,
            topo.nbr_count,
            edge_f0,
            edge_f1,
            topo.hash_key,
            topo.hash_val,
            topo.vface,
            topo.vface_count,
        )
    )


def _link_condition_ok_py(
    a: int,
    b: int,
    neighbors: list[set[int]],
    edge_faces: dict[tuple[int, int], set[int]],
    F: np.ndarray,
) -> bool:
    """Python link condition used by the sequential collapse pass."""
    key = _edge_key(a, b)
    faces = list(edge_faces.get(key, ()))
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
    if common != thirds or len(thirds) > 2:
        return False
    if len(neighbors[a]) <= 3 and len(neighbors[b]) <= 3 and len(thirds) == 2:
        return False
    keep, drop = b, a
    faces_drop = _faces_incident_to_vertex(drop, neighbors, edge_faces)
    edge_face_set = set(faces)
    remapped: list[tuple[int, int, int]] = []
    for fi in faces_drop:
        if fi in edge_face_set:
            continue
        v0, v1, v2 = (keep if int(x) == drop else int(x) for x in F[fi])
        if v0 == v1 or v1 == v2 or v2 == v0:
            continue
        remapped.append(tuple(sorted((v0, v1, v2))))  # type: ignore[arg-type]
    if len(remapped) != len(set(remapped)):
        return False
    faces_keep = _faces_incident_to_vertex(keep, neighbors, edge_faces) - faces_drop
    existing = {tuple(sorted(int(x) for x in F[fi])) for fi in faces_keep}
    for tri in remapped:
        if tri in existing:
            return False
    return True


def _unlink_face(fi: int, F: np.ndarray, edge_faces: dict[tuple[int, int], set[int]]) -> None:
    a, b, c = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
    for u, v in ((a, b), (b, c), (c, a)):
        key = _edge_key(u, v)
        bucket = edge_faces.get(key)
        if bucket is not None:
            bucket.discard(fi)
            if not bucket:
                del edge_faces[key]


def _link_face(fi: int, F: np.ndarray, edge_faces: dict[tuple[int, int], set[int]]) -> None:
    a, b, c = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
    for u, v in ((a, b), (b, c), (c, a)):
        edge_faces.setdefault(_edge_key(u, v), set()).add(fi)


def _apply_collapse_local(
    keep: int,
    drop: int,
    F: np.ndarray,
    face_alive: np.ndarray,
    neighbors: list[set[int]],
    edge_faces: dict[tuple[int, int], set[int]],
) -> None:
    faces_drop = _faces_incident_to_vertex(drop, neighbors, edge_faces)
    removed = set(edge_faces.get(_edge_key(keep, drop), ()))
    for fi in faces_drop:
        if not face_alive[fi]:
            continue
        _unlink_face(fi, F, edge_faces)
        if fi in removed:
            face_alive[fi] = False
            continue
        for k in range(3):
            if int(F[fi, k]) == drop:
                F[fi, k] = keep
        a, b, c = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        if a == b or b == c or c == a:
            face_alive[fi] = False
        else:
            _link_face(fi, F, edge_faces)
    drop_nbrs = set(neighbors[drop])
    drop_nbrs.discard(keep)
    for v in drop_nbrs:
        neighbors[v].discard(drop)
        neighbors[v].add(keep)
        neighbors[keep].add(v)
    neighbors[keep].discard(drop)
    neighbors[drop].clear()


def face_graph_components(F: np.ndarray) -> int:
    """Number of connected components in the face 1-skeleton (vertices linked by face edges)."""
    if F.size == 0:
        return 0
    used = np.unique(F.reshape(-1))
    parent = {int(i): int(i) for i in used}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i0, i1, i2 in F:
        a, b, c = int(i0), int(i1), int(i2)
        union(a, b)
        union(b, c)
        union(c, a)
    return len({find(int(i)) for i in used})


def faces_after_collapse(F: np.ndarray, keep: int, drop: int) -> np.ndarray:
    """Faces remaining after identifying ``drop`` with ``keep`` (degenerate faces removed)."""
    Fm = np.asarray(F, dtype=int).copy()
    Fm[Fm == drop] = keep
    alive = (Fm[:, 0] != Fm[:, 1]) & (Fm[:, 1] != Fm[:, 2]) & (Fm[:, 2] != Fm[:, 0])
    return Fm[alive]


def collapse_preserves_face_connectivity(F: np.ndarray, keep: int, drop: int) -> bool:
    """True if collapsing drop→keep does not increase face-graph component count."""
    before = face_graph_components(F)
    after_f = faces_after_collapse(F, keep, drop)
    after = face_graph_components(after_f)
    if after_f.shape[0] == 0:
        return before <= 1
    return after <= before


def is_vertex_degenerate(
    v: int,
    V: np.ndarray,
    F: np.ndarray,
    *,
    radius: float,
    neighbors: list[set[int]] | None = None,
) -> bool:
    """Local disk test: neighborhood within ``radius`` is not a topological disk."""
    v = int(v)
    if F.size == 0:
        return False
    nv = V.shape[0]
    if neighbors is None:
        topo = build_topology(np.asarray(F, dtype=np.int32), nv)
        neighbors = [
            {int(topo.nbr[i, j]) for j in range(int(topo.nbr_count[i]))}
            for i in range(nv)
        ]
    if v < 0 or v >= nv or not neighbors[v]:
        return False

    center = V[v]
    in_ball = np.linalg.norm(V - center, axis=1) <= radius
    in_ball[v] = True
    incident = [fi for fi, face in enumerate(F) if v in map(int, face)]
    if not incident:
        return False

    face_in: set[int] = set()
    vert_edge: dict[int, set[int]] = defaultdict(set)
    stack = list(incident)
    seen_f: set[int] = set(stack)
    v_to_f: list[list[int]] = [[] for _ in range(nv)]
    for fi, (a, b, c) in enumerate(F):
        v_to_f[int(a)].append(fi)
        v_to_f[int(b)].append(fi)
        v_to_f[int(c)].append(fi)

    while stack:
        fi = stack.pop()
        a, b, c = (int(x) for x in F[fi])
        if not (in_ball[a] and in_ball[b] and in_ball[c]):
            continue
        face_in.add(fi)
        for u, w in ((a, b), (b, c), (c, a)):
            vert_edge[u].add(w)
            vert_edge[w].add(u)
        for u in (a, b, c):
            for fj in v_to_f[u]:
                if fj not in seen_f:
                    seen_f.add(fj)
                    stack.append(fj)

    if not face_in:
        return False

    e_count = sum(len(nbrs) for nbrs in vert_edge.values()) // 2
    f_count = len(face_in)
    v_count = len(vert_edge)
    chi = v_count - e_count + f_count

    edge_face_count: dict[tuple[int, int], int] = defaultdict(int)
    for fi in face_in:
        a, b, c = (int(x) for x in F[fi])
        for u, w in ((a, b), (b, c), (c, a)):
            edge_face_count[_edge_key(u, w)] += 1
    border = [e for e, c in edge_face_count.items() if c == 1]
    if not border:
        return chi != 2

    bord_adj: dict[int, list[int]] = defaultdict(list)
    for u, w in border:
        bord_adj[u].append(w)
        bord_adj[w].append(u)
    visited: set[int] = set()
    cycles = 0
    for start in bord_adj:
        if start in visited:
            continue
        cycles += 1
        stack_b = [start]
        visited.add(start)
        while stack_b:
            u = stack_b.pop()
            for w in bord_adj[u]:
                if w not in visited:
                    visited.add(w)
                    stack_b.append(w)

    return not (chi == 1 and cycles == 1)


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
    vertex_flags: dict[str, np.ndarray] | None = None,
    max_collapses: int | None = None,
    max_passes: int | None = None,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None, np.ndarray | None]:
    """Collapse edges shorter than ``min_edge_length`` when the link condition holds.

    Midpoint merge with the closest pole retained, matching
    ``TopologyJanitor_ClosestPole::collapser``. Stops when no legal short edge
    remains, or when a configured safety limit/deadline is hit.

    Edges are visited in face-walk insertion order (legacy edge-map key order).
    Mid-pass link checks and remaps use incremental adjacency (O(degree) per
    collapse). Predicates match Numba ``link_condition_ok`` (golden-tested).

    Returns
    -------
    V, F, n_collapsed, fixed_out, poles_out
    """
    import time as _time

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
    flags = (
        {name: np.asarray(values).copy() for name, values in vertex_flags.items()}
        if vertex_flags is not None
        else {}
    )

    if max_collapses is None:
        max_collapses = max(nv, 200)
    if max_passes is None:
        max_passes = max(nv, 200)

    total = 0
    passes = 0
    vert_alive = np.ones(nv, dtype=bool)

    while passes < max_passes and total < max_collapses:
        if deadline is not None and _time.monotonic() >= deadline:
            break
        passes += 1
        if F.size == 0:
            break

        face_alive = np.ones(F.shape[0], dtype=bool)
        edges = _face_walk_undirected_edges(F, face_alive)
        if not edges:
            break

        neighbors = _vertex_neighbors(F, V.shape[0])
        edge_faces = _edge_to_faces(F)
        collapsed_this = 0

        for a, b in edges:
            if total + collapsed_this >= max_collapses:
                break
            if deadline is not None and _time.monotonic() >= deadline:
                break
            if not vert_alive[a] or not vert_alive[b]:
                continue
            if _edge_key(a, b) not in edge_faces:
                continue
            d = float(np.linalg.norm(V[a] - V[b]))
            if d >= min_edge_length:
                continue

            if not _link_condition_ok_py(a, b, neighbors, edge_faces, F):
                continue

            keep, drop = int(b), int(a)
            mid = 0.5 * (V[a] + V[b])
            if poles is not None:
                pa, pb = poles[keep], poles[drop]
                if np.linalg.norm(mid - pa) <= np.linalg.norm(mid - pb):
                    poles[keep] = pa
                else:
                    poles[keep] = pb
            V[keep] = mid
            vert_alive[drop] = False
            _apply_collapse_local(keep, drop, F, face_alive, neighbors, edge_faces)
            collapsed_this += 1

        if collapsed_this == 0:
            break

        F = F[face_alive]
        used = np.unique(F.reshape(-1)) if F.size else np.zeros(0, dtype=int)
        remap = -np.ones(V.shape[0], dtype=int)
        remap[used] = np.arange(used.shape[0], dtype=int)
        V = V[used]
        F = remap[F] if F.size else F
        fixed = fixed[used]
        if poles is not None:
            poles = poles[used]
        for name in flags:
            flags[name] = flags[name][used]
        vert_alive = np.ones(V.shape[0], dtype=bool)
        total += collapsed_this

    if vertex_flags is not None:
        vertex_flags.clear()
        vertex_flags.update(flags)
    return V, F, total, fixed, poles


def _face_walk_undirected_edges(
    F: np.ndarray, face_alive: np.ndarray
) -> list[tuple[int, int]]:
    """Undirected edges in first-seen face-walk order (matches legacy dict keys)."""
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for fi, (a, b, c) in enumerate(F):
        if not face_alive[fi]:
            continue
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = _edge_key(u, v)
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return edges


def _apply_collapse_faces(
    keep: int, drop: int, F: np.ndarray, face_alive: np.ndarray
) -> None:
    """Identify ``drop`` with ``keep`` and kill degenerate / edge-incident faces."""
    keep = int(keep)
    drop = int(drop)
    for fi in range(F.shape[0]):
        if not face_alive[fi]:
            continue
        a, b, c = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        verts = (a, b, c)
        if drop not in verts:
            continue
        if keep in verts:
            face_alive[fi] = False
            continue
        for k in range(3):
            if int(F[fi, k]) == drop:
                F[fi, k] = keep
        a, b, c = int(F[fi, 0]), int(F[fi, 1]), int(F[fi, 2])
        if a == b or b == c or c == a:
            face_alive[fi] = False


def _obtuse_split_candidates(
    V: np.ndarray,
    F: np.ndarray,
    *,
    thr: float,
    short_edge: float,
) -> np.ndarray:
    """Return ``(n, 5)`` int32 rows ``(u, v, face0, face1, split_side)`` for obtuse pairs."""
    m = F.shape[0]
    if m == 0:
        return np.zeros((0, 5), dtype=np.int32)

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    len01 = np.linalg.norm(v1 - v0, axis=1)
    len12 = np.linalg.norm(v2 - v1, axis=1)
    len20 = np.linalg.norm(v0 - v2, axis=1)
    short = (len01 < short_edge) | (len12 < short_edge) | (len20 < short_edge)

    def corner_angle(adj_a: np.ndarray, adj_b: np.ndarray, opp: np.ndarray) -> np.ndarray:
        denom = np.maximum(2.0 * adj_a * adj_b, 1e-30)
        cosine = np.clip((adj_a * adj_a + adj_b * adj_b - opp * opp) / denom, -1.0, 1.0)
        ang = np.arccos(cosine)
        ang[short] = -1.0
        return ang

    ang0 = corner_angle(len01, len20, len12)
    ang1 = corner_angle(len12, len01, len20)
    ang2 = corner_angle(len20, len12, len01)

    f01 = F[:, [0, 1]]
    f12 = F[:, [1, 2]]
    f20 = F[:, [2, 0]]
    e_min = np.minimum(
        np.concatenate([f01[:, 0], f12[:, 0], f20[:, 0]]),
        np.concatenate([f01[:, 1], f12[:, 1], f20[:, 1]]),
    ).astype(np.int32, copy=False)
    e_max = np.maximum(
        np.concatenate([f01[:, 0], f12[:, 0], f20[:, 0]]),
        np.concatenate([f01[:, 1], f12[:, 1], f20[:, 1]]),
    ).astype(np.int32, copy=False)
    face_ids = np.concatenate([np.arange(m, dtype=np.int32)] * 3)
    opp_ang = np.concatenate([ang2, ang0, ang1]).astype(np.float64, copy=False)
    opp_vert = np.concatenate([F[:, 2], F[:, 0], F[:, 1]]).astype(np.int32, copy=False)

    order = np.lexsort((e_max, e_min))
    rows = pair_obtuse_edges(
        e_min[order],
        e_max[order],
        face_ids[order],
        opp_ang[order],
        opp_vert[order],
        float(thr),
    )
    return rows


def split_obtuse_faces(
    V: np.ndarray,
    F: np.ndarray,
    *,
    max_angle_deg: float = 110.0,
    short_edge: float = 1e-12,
    fixed: np.ndarray | None = None,
    poles: np.ndarray | None = None,
    is_split: np.ndarray | None = None,
    max_passes: int | None = None,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Split flat triangle pairs at obtuse angles via edge splits.

    Procedure compatible with Starlab ``mcfskel`` remeshing.
    """
    import time as _time

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

    if max_passes is None:
        max_passes = max(nv, 200)

    thr = float(max_angle_deg) * (np.pi / 180.0)
    total = 0
    passes = 0
    tri0 = np.empty(3, dtype=np.int32)
    tri1 = np.empty(3, dtype=np.int32)

    while passes < max_passes:
        if deadline is not None and _time.monotonic() >= deadline:
            break
        passes += 1
        selected = _obtuse_split_candidates(V, F, thr=thr, short_edge=short_edge)
        if selected.shape[0] == 0:
            break

        used_verts: set[int] = set()
        batch_rows: list[np.ndarray] = []
        for row in selected:
            u, v, face0, face1, split_side = (int(row[i]) for i in range(5))
            if u in used_verts or v in used_verts or split_side in used_verts:
                continue
            used_verts.update((u, v, split_side))
            batch_rows.append(row)

        if not batch_rows:
            batch_rows = [selected[0]]

        face_used: set[int] = set()
        clean_batch: list[np.ndarray] = []
        for row in batch_rows:
            f0, f1 = int(row[2]), int(row[3])
            if f0 in face_used or f1 in face_used:
                continue
            face_used.update((f0, f1))
            clean_batch.append(row)
        batch = clean_batch or [selected[0]]

        planned: list[tuple[int, int, int, int, int, np.ndarray, np.ndarray | None]] = []
        for row in batch:
            u, v, face0, face1, split_side = (int(row[i]) for i in range(5))
            if deadline is not None and _time.monotonic() >= deadline:
                break
            edge = V[v] - V[u]
            edge_length = float(np.linalg.norm(edge))
            if edge_length <= 0:
                continue
            direction = edge / edge_length
            t = float(np.dot(V[split_side] - V[u], direction))
            new_position = V[u] + t * direction
            new_pole = None
            if poles is not None:
                pole_edge = poles[v] - poles[u]
                pole_length = float(np.linalg.norm(pole_edge))
                if pole_length > 0:
                    new_pole = poles[u] + t * (pole_edge / pole_length)
                else:
                    new_pole = poles[u].copy()
            planned.append((u, v, face0, face1, split_side, new_position, new_pole))

        if not planned:
            break

        n_old_v = V.shape[0]
        n_add_v = len(planned)
        n_old_f = F.shape[0]
        remove = {fi for _u, _v, f0, f1, _s, _p, _pp in planned for fi in (f0, f1)}
        n_keep_f = n_old_f - len(remove)
        n_new_f = 4 * n_add_v

        V_out = np.empty((n_old_v + n_add_v, 3), dtype=float)
        V_out[:n_old_v] = V
        fixed_out = np.empty(n_old_v + n_add_v, dtype=bool)
        fixed_out[:n_old_v] = fixed
        is_split_out = np.empty(n_old_v + n_add_v, dtype=bool)
        is_split_out[:n_old_v] = is_split
        if poles is not None:
            poles_out = np.empty((n_old_v + n_add_v, 3), dtype=float)
            poles_out[:n_old_v] = poles

        F_out = np.empty((n_keep_f + n_new_f, 3), dtype=int)
        keep_mask = np.ones(n_old_f, dtype=bool)
        keep_mask[list(remove)] = False
        F_out[:n_keep_f] = F[keep_mask]
        f_write = n_keep_f
        F_i32 = np.asarray(F, dtype=np.int32)

        for i, (u, v, face0, face1, _split_side, new_position, new_pole) in enumerate(planned):
            new_index = n_old_v + i
            V_out[new_index] = new_position
            fixed_out[new_index] = False
            is_split_out[new_index] = True
            if poles is not None and new_pole is not None:
                poles_out[new_index] = new_pole
            for src_face in (face0, face1):
                ok = split_face_on_edge_numba(
                    F_i32[src_face],
                    np.int32(u),
                    np.int32(v),
                    np.int32(new_index),
                    tri0,
                    tri1,
                )
                if not ok:
                    raise ValueError("face does not contain split edge")
                F_out[f_write] = tri0
                f_write += 1
                F_out[f_write] = tri1
                f_write += 1
            total += 1

        V = V_out
        fixed = fixed_out
        is_split = is_split_out
        if poles is not None:
            poles = poles_out
        F = F_out[:f_write]

    return V, F, total, fixed, poles, is_split


def collapse_ok_for_edge(
    a: int,
    b: int,
    V: np.ndarray,
    F: np.ndarray,
    *,
    check_connectivity: bool = True,
    topo: MeshTopology | None = None,
    neighbors: list[set[int]] | None = None,
    edge_faces: dict | None = None,
) -> bool:
    """Return True if collapsing edge (a,b) is link-legal (and optionally connectivity-safe).

    Pass a precomputed ``topo`` when checking many edges on the same mesh.
    Legacy ``neighbors`` / ``edge_faces`` kwargs are accepted but ignored.
    """
    _ = neighbors, edge_faces
    a, b = int(a), int(b)
    if a > b:
        a, b = b, a
    if topo is None:
        topo = build_topology(np.asarray(F, dtype=np.int32), V.shape[0])
    ok = bool(
        link_condition_ok(
            np.int32(a),
            np.int32(b),
            topo.F,
            topo.face_alive,
            topo.nbr,
            topo.nbr_count,
            topo.edge_f0,
            topo.edge_f1,
            topo.hash_key,
            topo.hash_val,
            topo.vface,
            topo.vface_count,
        )
    )
    if not ok:
        return False
    if not check_connectivity:
        return True
    return collapse_preserves_face_connectivity(F, b, a)
