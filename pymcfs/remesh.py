"""Local remeshing helpers for MCF skeletonization.

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


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _vertex_neighbors(F: np.ndarray, nv: int) -> list[set[int]]:
    nbrs: list[set[int]] = [set() for _ in range(nv)]
    for i0, i1, i2 in F:
        a, b, c = int(i0), int(i1), int(i2)
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    return nbrs


def _edge_to_faces(F: np.ndarray) -> dict[tuple[int, int], list[int]]:
    m: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(F):
        for u, v in ((a, b), (b, c), (c, a)):
            m[_edge_key(int(u), int(v))].append(fi)
    return m


def mesh_adjacency(
    F: np.ndarray, nv: int
) -> tuple[list[set[int]], dict[tuple[int, int], list[int]]]:
    """Return (vertex neighbors, edge→incident faces) for a triangle mesh."""
    return _vertex_neighbors(F, nv), _edge_to_faces(F)


def _faces_incident_to_vertex(
    v: int,
    neighbors: list[set[int]],
    edge_faces: dict[tuple[int, int], list[int]],
) -> set[int]:
    """Face indices incident to ``v`` via the edge→faces map."""
    out: set[int] = set()
    for nbr in neighbors[v]:
        out.update(edge_faces.get(_edge_key(v, nbr), ()))
    return out


def _link_condition_ok(
    a: int,
    b: int,
    neighbors: list[set[int]],
    edge_faces: dict[tuple[int, int], list[int]],
    F: np.ndarray,
) -> bool:
    """Manifold link condition for collapsing edge (a, b).

    Common 1-ring neighbors must equal the opposite vertices of faces incident
    to the edge (at most two on a manifold interior edge). Additional local
    checks reject tetrahedron collapses and duplicate-face outcomes without
    scanning the full mesh (O(degree) rather than O(|F|)).
    """
    key = _edge_key(a, b)
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
    if common != thirds or len(thirds) > 2:
        return False

    # Tetrahedron (and similar valence-3 diamonds): link equality holds but the
    # collapse leaves duplicate faces / a non-manifold. Matches the intent of
    # Surface_mesh::is_collapse_ok without a global post-collapse scan.
    if len(neighbors[a]) <= 3 and len(neighbors[b]) <= 3 and len(thirds) == 2:
        return False

    # Local duplicate-face check: remapped faces incident to drop must not
    # collide with each other or with untouched faces around keep.
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
    # Allow dissolving toward a curve (0 components) but never fragment.
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
    """Local disk test: neighborhood within ``radius`` is not a topological disk.

    Grows the set of faces whose vertices all lie inside the Euclidean ball of
    ``radius`` around ``V[v]``, then checks Euler characteristic and border cycle
    count. A proper disk has χ = 1 and a single boundary component.
    """
    v = int(v)
    if F.size == 0:
        return False
    nv = V.shape[0]
    if neighbors is None:
        neighbors = _vertex_neighbors(F, nv)
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


def _apply_collapse_topology(
    keep: int,
    drop: int,
    F: np.ndarray,
    neighbors: list[set[int]],
    edge_faces: dict[tuple[int, int], list[int]],
) -> np.ndarray:
    """Identify ``drop`` with ``keep`` and patch adjacency in-place. Returns new F."""
    # Remap / drop faces in the face array.
    Fm = F.copy()
    Fm[Fm == drop] = keep
    alive = (Fm[:, 0] != Fm[:, 1]) & (Fm[:, 1] != Fm[:, 2]) & (Fm[:, 2] != Fm[:, 0])
    new_faces = Fm[alive]

    # Rebuild edge_faces from surviving faces (O(|F|) per collapse; link checks
    # stay O(degree)). Neighbor sets are patched locally.
    edge_faces.clear()
    for fi, (a, b, c) in enumerate(new_faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[_edge_key(int(u), int(v))].append(fi)

    drop_nbrs = set(neighbors[drop])
    drop_nbrs.discard(keep)
    for v in drop_nbrs:
        neighbors[v].discard(drop)
        neighbors[v].add(keep)
        neighbors[keep].add(v)
    neighbors[keep].discard(drop)
    neighbors[drop].clear()

    return new_faces


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
    alive = np.ones(nv, dtype=bool)

    while passes < max_passes and total < max_collapses:
        if deadline is not None and _time.monotonic() >= deadline:
            break
        passes += 1
        if F.size == 0:
            break

        edges = mesh_unique_edges(F)
        if edges.size == 0:
            break
        # Starlab scans mesh edge order; it does not sort local-remesh collapses.
        neighbors = _vertex_neighbors(F, V.shape[0])
        edge_faces = _edge_to_faces(F)
        collapsed_this = 0

        for ei in range(edges.shape[0]):
            if total + collapsed_this >= max_collapses:
                break
            if deadline is not None and _time.monotonic() >= deadline:
                break
            a, b = int(edges[ei, 0]), int(edges[ei, 1])
            if not alive[a] or not alive[b]:
                continue
            d = float(np.linalg.norm(V[a] - V[b]))
            if d >= min_edge_length:
                continue
            if not _link_condition_ok(a, b, neighbors, edge_faces, F):
                continue

            # halfedge(e, 0) collapses from v0 into v1 in Starlab.
            keep, drop = b, a
            mid = 0.5 * (V[a] + V[b])
            if poles is not None:
                pa, pb = poles[keep], poles[drop]
                if np.linalg.norm(mid - pa) <= np.linalg.norm(mid - pb):
                    poles[keep] = pa
                else:
                    poles[keep] = pb
            V[keep] = mid
            alive[drop] = False
            F = _apply_collapse_topology(keep, drop, F, neighbors, edge_faces)
            collapsed_this += 1

        if collapsed_this == 0:
            break

        # Compact dead vertices after a productive scan.
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
        alive = np.ones(V.shape[0], dtype=bool)
        total += collapsed_this

    if vertex_flags is not None:
        vertex_flags.clear()
        vertex_flags.update(flags)
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
    max_passes: int | None = None,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Split flat triangle pairs using Starlab's edge-split procedure.

    ``TopologyJanitor_ClosestPole::splitter`` splits an interior edge only when
    the angles opposite that edge in *both* incident triangles exceed the
    threshold. Splitting the edge updates both triangles and preserves the
    closed manifold connectivity.
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

    def opposite_angle(face: np.ndarray, u: int, v: int) -> tuple[float, int]:
        verts = [int(x) for x in face]
        w = next(x for x in verts if x != u and x != v)
        lengths = [
            float(np.linalg.norm(V[verts[(k + 1) % 3]] - V[verts[(k + 2) % 3]]))
            for k in range(3)
        ]
        if min(lengths) < short_edge:
            return -1.0, w
        wi = verts.index(w)
        a = lengths[wi]
        b = lengths[(wi + 1) % 3]
        c = lengths[(wi + 2) % 3]
        cosine = float(np.clip((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0))
        return float(np.arccos(cosine)), w

    def split_face_on_edge(face: np.ndarray, u: int, v: int, new: int) -> list[np.ndarray]:
        a, b, c = (int(x) for x in face)
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
            if x == u and y == v:
                return [
                    np.array([x, new, z], dtype=int),
                    np.array([new, y, z], dtype=int),
                ]
            if x == v and y == u:
                return [
                    np.array([x, new, z], dtype=int),
                    np.array([new, y, z], dtype=int),
                ]
        raise ValueError("face does not contain split edge")

    while passes < max_passes:
        if deadline is not None and _time.monotonic() >= deadline:
            break
        passes += 1
        edge_faces = _edge_to_faces(F)
        # Collect all currently legal split edges, then apply an independent
        # set (no shared vertices) so one adjacency rebuild serves many splits.
        selected: list[tuple[int, int, int, int, int]] = []
        for (u, v), incident in edge_faces.items():
            if len(incident) != 2:
                continue
            angle0, opposite0 = opposite_angle(F[incident[0]], u, v)
            angle1, opposite1 = opposite_angle(F[incident[1]], u, v)
            if angle0 < thr or angle1 < thr:
                continue
            split_side = opposite0 if angle0 > angle1 else opposite1
            selected.append((u, v, incident[0], incident[1], split_side))

        if not selected:
            break

        used_verts: set[int] = set()
        # Apply in scan order (Starlab takes the first); skip conflicts.
        batch = []
        for u, v, face0, face1, split_side in selected:
            if u in used_verts or v in used_verts or split_side in used_verts:
                continue
            used_verts.update((u, v, split_side))
            batch.append((u, v, face0, face1, split_side))

        if not batch:
            # Degenerate conflict set — fall back to single first candidate.
            batch = [selected[0]]

        # Apply from high face index to low so removals stay valid within batch
        # only when faces don't overlap — guaranteed by used_verts on face verts
        # but face indices themselves must not collide.
        face_used: set[int] = set()
        clean_batch = []
        for item in batch:
            _u, _v, f0, f1, _s = item
            if f0 in face_used or f1 in face_used:
                continue
            face_used.update((f0, f1))
            clean_batch.append(item)
        batch = clean_batch
        if not batch:
            batch = [selected[0]]

        # Remove faces from highest index first, append all new faces after.
        remove = set()
        new_rows: list[np.ndarray] = []
        for u, v, face0, face1, split_side in batch:
            if deadline is not None and _time.monotonic() >= deadline:
                break
            edge = V[v] - V[u]
            edge_length = float(np.linalg.norm(edge))
            if edge_length <= 0:
                continue
            direction = edge / edge_length
            t = float(np.dot(V[split_side] - V[u], direction))
            new_position = V[u] + t * direction
            new_index = V.shape[0]
            V = np.vstack([V, new_position])
            fixed = np.append(fixed, False)
            is_split = np.append(is_split, True)
            if poles is not None:
                pole_edge = poles[v] - poles[u]
                pole_length = float(np.linalg.norm(pole_edge))
                if pole_length > 0:
                    new_pole = poles[u] + t * (pole_edge / pole_length)
                else:
                    new_pole = poles[u].copy()
                poles = np.vstack([poles, new_pole])

            new_rows.extend(split_face_on_edge(F[face0], u, v, new_index))
            new_rows.extend(split_face_on_edge(F[face1], u, v, new_index))
            remove.update((face0, face1))
            total += 1

        if not remove:
            break
        keep_faces = np.ones(F.shape[0], dtype=bool)
        keep_faces[list(remove)] = False
        F = np.vstack([F[keep_faces], *new_rows]) if new_rows else F[keep_faces]

    return V, F, total, fixed, poles, is_split


def collapse_ok_for_edge(
    a: int,
    b: int,
    V: np.ndarray,
    F: np.ndarray,
    *,
    check_connectivity: bool = True,
    neighbors: list[set[int]] | None = None,
    edge_faces: dict[tuple[int, int], list[int]] | None = None,
) -> bool:
    """Return True if collapsing edge (a,b) is link-legal (and optionally connectivity-safe).

    Pass precomputed ``neighbors`` / ``edge_faces`` when checking many edges on
    the same mesh (e.g. degeneracy detection) to avoid O(|F|) rebuilds per call.
    """
    a, b = int(a), int(b)
    nv = V.shape[0]
    if neighbors is None:
        neighbors = _vertex_neighbors(F, nv)
    if edge_faces is None:
        edge_faces = _edge_to_faces(F)
    if not _link_condition_ok(a, b, neighbors, edge_faces, F):
        return False
    if not check_connectivity:
        return True
    keep, drop = (a, b) if a < b else (b, a)
    return collapse_preserves_face_connectivity(F, keep, drop)
