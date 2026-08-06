"""Array-backed mesh topology with Numba kernels (internal; used by remesh)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

# Per-vertex neighbor / incident-face capacity (manifold meshes are << this).
MAX_NBR = 64
MAX_VF = 64


@dataclass
class MeshTopology:
    """Int32 array topology for a triangle mesh (possibly with face tombstones)."""

    F: np.ndarray  # (m, 3) int32
    face_alive: np.ndarray  # (m,) bool
    nv: int
    nbr: np.ndarray  # (nv, MAX_NBR) int32
    nbr_count: np.ndarray  # (nv,) int32
    edge_u: np.ndarray  # (ne,) int32
    edge_v: np.ndarray  # (ne,) int32
    edge_f0: np.ndarray  # (ne,) int32
    edge_f1: np.ndarray  # (ne,) int32
    n_edges: int
    hash_key: np.ndarray  # (cap,) int64
    hash_val: np.ndarray  # (cap,) int32  (-1 empty)
    vface: np.ndarray  # (nv, MAX_VF) int32
    vface_count: np.ndarray  # (nv,) int32
    vert_alive: np.ndarray  # (nv,) bool


@njit(cache=True)
def _pack_edge_key(u: np.int32, v: np.int32) -> np.int64:
    if u < v:
        return (np.int64(u) << np.int64(32)) | np.int64(v)
    return (np.int64(v) << np.int64(32)) | np.int64(u)


HASH_EMPTY = np.int64(-1)


@njit(cache=True)
def _next_pow2(x: np.int64) -> np.int64:
    if x < 2:
        return np.int64(2)
    n = np.int64(1)
    while n < x:
        n <<= np.int64(1)
    return n


@njit(cache=True)
def _hash_lookup(hash_key: np.ndarray, hash_val: np.ndarray, u: np.int32, v: np.int32) -> np.int32:
    key = _pack_edge_key(u, v)
    cap = hash_key.shape[0]
    h = key % np.int64(cap)
    if h < 0:
        h = -h
    for _ in range(cap):
        k = hash_key[h]
        if k == HASH_EMPTY:
            return np.int32(-1)
        if k == key:
            return hash_val[h]
        h += np.int64(1)
        if h >= cap:
            h = np.int64(0)
    return np.int32(-1)


@njit(cache=True)
def _hash_insert(
    hash_key: np.ndarray,
    hash_val: np.ndarray,
    u: np.int32,
    v: np.int32,
    edge_idx: np.int32,
) -> None:
    key = _pack_edge_key(u, v)
    cap = hash_key.shape[0]
    h = key % np.int64(cap)
    if h < 0:
        h = -h
    for _ in range(cap):
        k = hash_key[h]
        if k == key or k == HASH_EMPTY:
            hash_key[h] = key
            hash_val[h] = edge_idx
            return
        h += np.int64(1)
        if h >= cap:
            h = np.int64(0)


@njit(cache=True)
def _add_nbr(nbr: np.ndarray, nbr_count: np.ndarray, a: np.int32, b: np.int32) -> None:
    if a == b:
        return
    c = nbr_count[a]
    for i in range(c):
        if nbr[a, i] == b:
            return
    if c >= MAX_NBR:
        return
    nbr[a, c] = b
    nbr_count[a] = c + 1


@njit(cache=True)
def _add_vface(vface: np.ndarray, vface_count: np.ndarray, v: np.int32, fi: np.int32) -> None:
    c = vface_count[v]
    for i in range(c):
        if vface[v, i] == fi:
            return
    if c >= MAX_VF:
        return
    vface[v, c] = fi
    vface_count[v] = c + 1


@njit(cache=True)
def _build_topology_kernel(
    F: np.ndarray,
    face_alive: np.ndarray,
    nv: np.int32,
):
    m = F.shape[0]
    nbr = np.full((nv, MAX_NBR), -1, dtype=np.int32)
    nbr_count = np.zeros(nv, dtype=np.int32)
    vface = np.full((nv, MAX_VF), -1, dtype=np.int32)
    vface_count = np.zeros(nv, dtype=np.int32)

    # Upper bound on edges: 3 * alive faces.
    max_edges = 3 * m + 1
    edge_u = np.empty(max_edges, dtype=np.int32)
    edge_v = np.empty(max_edges, dtype=np.int32)
    edge_f0 = np.empty(max_edges, dtype=np.int32)
    edge_f1 = np.empty(max_edges, dtype=np.int32)
    n_edges = np.int32(0)

    cap = int(_next_pow2(np.int64(max(8, 2 * max_edges))))
    hash_key = np.full(cap, np.int64(-1), dtype=np.int64)
    hash_val = np.full(cap, np.int32(-1), dtype=np.int32)

    for fi in range(m):
        if not face_alive[fi]:
            continue
        i0 = F[fi, 0]
        i1 = F[fi, 1]
        i2 = F[fi, 2]
        _add_nbr(nbr, nbr_count, i0, i1)
        _add_nbr(nbr, nbr_count, i0, i2)
        _add_nbr(nbr, nbr_count, i1, i0)
        _add_nbr(nbr, nbr_count, i1, i2)
        _add_nbr(nbr, nbr_count, i2, i0)
        _add_nbr(nbr, nbr_count, i2, i1)
        _add_vface(vface, vface_count, i0, np.int32(fi))
        _add_vface(vface, vface_count, i1, np.int32(fi))
        _add_vface(vface, vface_count, i2, np.int32(fi))

        for ua, va in ((i0, i1), (i1, i2), (i2, i0)):
            ei = _hash_lookup(hash_key, hash_val, ua, va)
            if ei < 0:
                if ua < va:
                    eu, ev = ua, va
                else:
                    eu, ev = va, ua
                edge_u[n_edges] = eu
                edge_v[n_edges] = ev
                edge_f0[n_edges] = np.int32(fi)
                edge_f1[n_edges] = np.int32(-1)
                _hash_insert(hash_key, hash_val, eu, ev, n_edges)
                n_edges += np.int32(1)
            else:
                if edge_f1[ei] < 0:
                    edge_f1[ei] = np.int32(fi)

    return (
        nbr,
        nbr_count,
        edge_u[:n_edges].copy(),
        edge_v[:n_edges].copy(),
        edge_f0[:n_edges].copy(),
        edge_f1[:n_edges].copy(),
        n_edges,
        hash_key,
        hash_val,
        vface,
        vface_count,
    )


def build_topology(
    F: np.ndarray,
    nv: int | None = None,
    *,
    face_alive: np.ndarray | None = None,
    vert_alive: np.ndarray | None = None,
) -> MeshTopology:
    """Build array topology from triangle faces."""
    F = np.asarray(F, dtype=np.int32)
    if F.size == 0:
        n = int(nv or 0)
        return MeshTopology(
            F=F.reshape(0, 3),
            face_alive=np.zeros(0, dtype=np.bool_),
            nv=n,
            nbr=np.full((n, MAX_NBR), -1, dtype=np.int32),
            nbr_count=np.zeros(n, dtype=np.int32),
            edge_u=np.zeros(0, dtype=np.int32),
            edge_v=np.zeros(0, dtype=np.int32),
            edge_f0=np.zeros(0, dtype=np.int32),
            edge_f1=np.zeros(0, dtype=np.int32),
            n_edges=0,
            hash_key=np.full(2, np.int64(-1), dtype=np.int64),
            hash_val=np.full(2, -1, dtype=np.int32),
            vface=np.full((n, MAX_VF), -1, dtype=np.int32),
            vface_count=np.zeros(n, dtype=np.int32),
            vert_alive=np.ones(n, dtype=np.bool_) if vert_alive is None else vert_alive,
        )
    if nv is None:
        nv = int(F.max()) + 1 if F.size else 0
    nv = int(nv)
    if face_alive is None:
        face_alive = np.ones(F.shape[0], dtype=np.bool_)
    else:
        face_alive = np.asarray(face_alive, dtype=np.bool_)
    if vert_alive is None:
        vert_alive = np.ones(nv, dtype=np.bool_)
    else:
        vert_alive = np.asarray(vert_alive, dtype=np.bool_)

    (
        nbr,
        nbr_count,
        edge_u,
        edge_v,
        edge_f0,
        edge_f1,
        n_edges,
        hash_key,
        hash_val,
        vface,
        vface_count,
    ) = _build_topology_kernel(F, face_alive, np.int32(nv))

    return MeshTopology(
        F=F,
        face_alive=face_alive,
        nv=nv,
        nbr=nbr,
        nbr_count=nbr_count,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_f0=edge_f0,
        edge_f1=edge_f1,
        n_edges=int(n_edges),
        hash_key=hash_key,
        hash_val=hash_val,
        vface=vface,
        vface_count=vface_count,
        vert_alive=vert_alive,
    )


@njit(cache=True)
def _opposite_vertex(F: np.ndarray, fi: np.int32, a: np.int32, b: np.int32) -> np.int32:
    for k in range(3):
        x = F[fi, k]
        if x != a and x != b:
            return x
    return np.int32(-1)


@njit(cache=True)
def _sorted3(a: np.int32, b: np.int32, c: np.int32):
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return a, b, c


@njit(cache=True)
def link_condition_ok(
    a: np.int32,
    b: np.int32,
    F: np.ndarray,
    face_alive: np.ndarray,
    nbr: np.ndarray,
    nbr_count: np.ndarray,
    edge_f0: np.ndarray,
    edge_f1: np.ndarray,
    hash_key: np.ndarray,
    hash_val: np.ndarray,
    vface: np.ndarray,
    vface_count: np.ndarray,
) -> bool:
    """Manifold link condition for collapsing edge (a, b); keep=b, drop=a."""
    ei = _hash_lookup(hash_key, hash_val, a, b)
    if ei < 0:
        return False
    f0 = edge_f0[ei]
    f1 = edge_f1[ei]
    if f0 < 0:
        return False

    # Opposite vertices of faces on the edge.
    t0 = _opposite_vertex(F, f0, a, b)
    if t0 < 0:
        return False
    n_thirds = 1
    t1 = np.int32(-1)
    if f1 >= 0:
        t1 = _opposite_vertex(F, f1, a, b)
        if t1 < 0:
            return False
        if t1 != t0:
            n_thirds = 2
        else:
            return False

    if n_thirds > 2:
        return False

    # Common neighbors must equal thirds.
    ca = nbr_count[a]
    cb = nbr_count[b]
    n_common = 0
    for i in range(ca):
        na = nbr[a, i]
        if na == b:
            continue
        for j in range(cb):
            if nbr[b, j] == na:
                n_common += 1
                if n_thirds == 1:
                    if na != t0:
                        return False
                else:
                    if na != t0 and na != t1:
                        return False
                break
    if n_common != n_thirds:
        return False

    # Tetrahedron / valence-3 diamond.
    if ca <= 3 and cb <= 3 and n_thirds == 2:
        return False

    keep = b
    drop = a

    # Remapped faces incident to drop (excluding edge faces) must not collide.
    remapped = np.empty((MAX_VF, 3), dtype=np.int32)
    n_remap = 0
    for ii in range(vface_count[drop]):
        fi = vface[drop, ii]
        if not face_alive[fi]:
            continue
        if fi == f0 or fi == f1:
            continue
        x0 = F[fi, 0]
        x1 = F[fi, 1]
        x2 = F[fi, 2]
        if x0 == drop:
            x0 = keep
        if x1 == drop:
            x1 = keep
        if x2 == drop:
            x2 = keep
        if x0 == x1 or x1 == x2 or x2 == x0:
            continue
        s0, s1, s2 = _sorted3(x0, x1, x2)
        # Duplicate among remapped?
        for r in range(n_remap):
            if remapped[r, 0] == s0 and remapped[r, 1] == s1 and remapped[r, 2] == s2:
                return False
        if n_remap >= MAX_VF:
            return False
        remapped[n_remap, 0] = s0
        remapped[n_remap, 1] = s1
        remapped[n_remap, 2] = s2
        n_remap += 1

    # Collision with untouched faces around keep.
    for ii in range(vface_count[keep]):
        fi = vface[keep, ii]
        if not face_alive[fi]:
            continue
        if fi == f0 or fi == f1:
            continue
        # Skip faces that also contain drop (those are being remapped).
        has_drop = False
        for k in range(3):
            if F[fi, k] == drop:
                has_drop = True
                break
        if has_drop:
            continue
        e0, e1, e2 = _sorted3(F[fi, 0], F[fi, 1], F[fi, 2])
        for r in range(n_remap):
            if remapped[r, 0] == e0 and remapped[r, 1] == e1 and remapped[r, 2] == e2:
                return False

    return True


@njit(cache=True)
def pair_obtuse_edges(
    e_min: np.ndarray,
    e_max: np.ndarray,
    face_ids: np.ndarray,
    opp_ang: np.ndarray,
    opp_vert: np.ndarray,
    thr: float,
):
    """Return stacked (u,v,f0,f1,split_side) rows for interior obtuse edge pairs."""
    n = e_min.shape[0]
    out = np.empty((n, 5), dtype=np.int32)
    n_out = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and e_min[j] == e_min[i] and e_max[j] == e_max[i]:
            j += 1
        if j - i == 2:
            a0 = opp_ang[i]
            a1 = opp_ang[i + 1]
            if a0 >= thr and a1 >= thr:
                out[n_out, 0] = e_min[i]
                out[n_out, 1] = e_max[i]
                out[n_out, 2] = face_ids[i]
                out[n_out, 3] = face_ids[i + 1]
                if a0 >= a1:
                    out[n_out, 4] = opp_vert[i]
                else:
                    out[n_out, 4] = opp_vert[i + 1]
                n_out += 1
        i = j
    return out[:n_out].copy()
