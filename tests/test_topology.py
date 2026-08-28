"""Tests for array-backed Numba mesh topology."""
from __future__ import annotations

import numpy as np
import trimesh as tm

from pymcfs.remesh import collapse_ok_for_edge, collapse_short_edges, _apply_collapse_local, _vertex_neighbors, _edge_to_faces
from pymcfs.topology import (
    _hash_slot,
    _pack_edge_key,
    apply_collapse_local,
    build_topology,
    link_condition_ok,
    topology_collapse_buffers,
)


def test_build_topology_icosphere_manifold_edges():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    F = np.asarray(mesh.faces, dtype=np.int32)
    nv = len(mesh.vertices)
    topo = build_topology(F, nv)
    assert topo.n_edges > 0
    assert np.all(topo.edge_f0 >= 0)
    assert np.all(topo.edge_f1 >= 0)
    # Each vertex degree from CSR matches unique face corners.
    for v in range(nv):
        assert int(topo.nbr_count[v]) >= 3
        assert int(topo.vface_count[v]) >= 3


def test_build_topology_edge_slots_follow_face_walk_order():
    """Edge indices come from first-seen face-walk order, not the hash layout."""
    mesh = tm.creation.icosphere(subdivisions=3, radius=1.0)
    F = np.asarray(mesh.faces, dtype=np.int32)
    topo = build_topology(F, len(mesh.vertices))

    expected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for a, b, c in F:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            if key not in seen:
                seen.add(key)
                expected.append(key)

    assert topo.n_edges == len(expected)
    assert list(zip(topo.edge_u.tolist(), topo.edge_v.tolist())) == expected


def test_face_walk_edges_matches_python_reference():
    """Numba edge enumeration reproduces the Python visit order, tombstones included."""
    from pymcfs.remesh import _face_walk_undirected_edges
    from pymcfs.topology import face_walk_edges

    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    F = np.asarray(mesh.faces, dtype=np.int32)

    for face_alive in (
        np.ones(F.shape[0], dtype=bool),
        np.arange(F.shape[0]) % 3 != 0,
    ):
        expected = _face_walk_undirected_edges(F, face_alive)
        got = face_walk_edges(F, face_alive)
        assert [tuple(e) for e in got.tolist()] == expected


def test_edge_hash_probe_length_stays_low():
    """Guard the key mix: ``key % cap`` degenerates into one giant cluster.

    ``cap`` is a power of two and the smaller endpoint is shifted up by 32, so
    plain modulo keeps only bits of the larger endpoint and every key lands in
    ``[0, nv)``. Average probe length then grows to thousands.
    """
    mesh = tm.creation.icosphere(subdivisions=3, radius=1.0)
    F = np.asarray(mesh.faces, dtype=np.int32)
    topo = build_topology(F, len(mesh.vertices))
    cap = int(topo.hash_key.shape[0])

    probes = 0
    for ei in range(topo.n_edges):
        key = int(_pack_edge_key(np.int32(topo.edge_u[ei]), np.int32(topo.edge_v[ei])))
        h = int(_hash_slot(np.int64(key), np.int64(cap)))
        while int(topo.hash_key[h]) != key:
            probes += 1
            h = (h + 1) % cap
            assert probes <= cap, "edge key missing from hash table"
        probes += 1

    assert probes / topo.n_edges < 2.0


def test_link_condition_matches_collapse_ok():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int32)
    topo = build_topology(F, V.shape[0])
    # Spot-check a handful of edges.
    for ei in range(min(20, topo.n_edges)):
        a, b = int(topo.edge_u[ei]), int(topo.edge_v[ei])
        njit_ok = bool(
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
        api_ok = collapse_ok_for_edge(a, b, V, F, check_connectivity=False, topo=topo)
        assert njit_ok is api_ok


def test_collapse_short_edges_with_topology_backend():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    n0 = V.shape[0]
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    V2, F2, ncoll, *_ = collapse_short_edges(V, F, min_edge_length=0.15 * diag)
    assert ncoll > 0
    assert V2.shape[0] < n0
    assert np.all((F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0]))


def test_apply_collapse_local_matches_python():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    F = np.asarray(mesh.faces, dtype=np.int32)
    face_alive = np.ones(F.shape[0], dtype=bool)
    vert_alive = np.ones(len(mesh.vertices), dtype=bool)
    neighbors = _vertex_neighbors(F, len(mesh.vertices))
    edge_faces = _edge_to_faces(F)

    topo = build_topology(F, len(mesh.vertices), face_alive=face_alive, vert_alive=vert_alive)
    bufs = topology_collapse_buffers(topo, face_count=F.shape[0])

    F_py = F.copy()
    face_py = face_alive.copy()
    n_py = [set(s) for s in neighbors]
    ef_py = {k: set(v) for k, v in edge_faces.items()}

    keep, drop = 5, 0
    _apply_collapse_local(keep, drop, F_py, face_py, n_py, ef_py)
    apply_collapse_local(
        np.int32(keep),
        np.int32(drop),
        topo.F,
        topo.face_alive,
        topo.nbr,
        topo.nbr_count,
        *bufs,
        topo.hash_key,
        topo.hash_val,
        topo.vface,
        topo.vface_count,
    )

    assert np.array_equal(F_py, topo.F)
    assert np.array_equal(face_py, topo.face_alive)


def _edge_faces_brute(F: np.ndarray, face_alive: np.ndarray) -> dict[tuple[int, int], set[int]]:
    out: dict[tuple[int, int], set[int]] = {}
    for fi, (a, b, c) in enumerate(F):
        if not face_alive[fi]:
            continue
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            out.setdefault(key, set()).add(fi)
    return out


def _edge_faces_from_numba(
    face_alive: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_f0: np.ndarray,
    edge_f1: np.ndarray,
    n_edges: np.ndarray,
) -> dict[tuple[int, int], set[int]]:
    out: dict[tuple[int, int], set[int]] = {}
    for ei in range(int(n_edges[0])):
        u, v = int(edge_u[ei]), int(edge_v[ei])
        key = (u, v) if u < v else (v, u)
        faces: set[int] = set()
        for f in (int(edge_f0[ei]), int(edge_f1[ei])):
            if f >= 0 and face_alive[f]:
                faces.add(f)
        if faces:
            out[key] = faces
    return out


def test_apply_collapse_local_pass_invariants_icosphere():
    """Numba incremental collapse matches Python edge-face maps for pass 1."""
    from pymcfs.remesh import (
        _apply_collapse_local,
        _edge_key,
        _edge_to_faces,
        _face_walk_undirected_edges,
        _link_condition_ok_py,
        _vertex_neighbors,
    )

    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    min_len = 0.15 * diag
    face_alive = np.ones(F.shape[0], dtype=bool)
    edges = _face_walk_undirected_edges(F, face_alive)

    F_py = F.copy()
    face_py = face_alive.copy()
    n_py = _vertex_neighbors(F, V.shape[0])
    ef_py = _edge_to_faces(F)

    topo = build_topology(np.asarray(F, dtype=np.int32), V.shape[0], face_alive=face_alive.copy())
    bufs = topology_collapse_buffers(topo, face_count=F.shape[0])
    edge_u, edge_v, edge_f0, edge_f1, n_edges, edge_cap = bufs
    vert_alive = np.ones(V.shape[0], dtype=bool)

    for a, b in edges:
        if not vert_alive[a] or not vert_alive[b]:
            continue
        if _edge_key(a, b) not in ef_py:
            continue
        if float(np.linalg.norm(V[a] - V[b])) >= min_len:
            continue
        if not _link_condition_ok_py(a, b, n_py, ef_py, F_py):
            continue

        keep, drop = int(b), int(a)
        _apply_collapse_local(keep, drop, F_py, face_py, n_py, ef_py)
        apply_collapse_local(
            np.int32(keep),
            np.int32(drop),
            topo.F,
            topo.face_alive,
            topo.nbr,
            topo.nbr_count,
            edge_u,
            edge_v,
            edge_f0,
            edge_f1,
            n_edges,
            edge_cap,
            topo.hash_key,
            topo.hash_val,
            topo.vface,
            topo.vface_count,
        )
        vert_alive[drop] = False

        assert np.array_equal(F_py, topo.F)
        assert np.array_equal(face_py, topo.face_alive)
        brute = _edge_faces_brute(topo.F, topo.face_alive)
        numba = _edge_faces_from_numba(
            topo.face_alive, edge_u, edge_v, edge_f0, edge_f1, n_edges
        )
        assert brute == numba


def _collapse_reference(
    V: np.ndarray,
    F: np.ndarray,
    *,
    min_edge_length: float,
    poles: np.ndarray | None = None,
    pole_valid: np.ndarray | None = None,
):
    """Pure-Python short-edge collapse, the reference for the Numba sweep.

    Mirrors ``collapse_short_edges`` using the Python link condition and the
    Python incremental adjacency update.
    """
    from pymcfs.remesh import (
        _apply_collapse_local,
        _edge_to_faces,
        _face_walk_undirected_edges,
        _link_condition_ok_py,
        _vertex_neighbors,
    )

    V = np.asarray(V, dtype=float).copy()
    F = np.asarray(F, dtype=int).copy()
    nv = V.shape[0]
    poles = None if poles is None else np.asarray(poles, dtype=float).copy()
    pole_valid = None if pole_valid is None else np.asarray(pole_valid, dtype=bool).copy()

    face_alive = np.ones(F.shape[0], dtype=bool)
    vert_alive = np.ones(nv, dtype=bool)
    neighbors = _vertex_neighbors(F, nv)
    edge_faces = _edge_to_faces(F)
    total = 0

    while True:
        collapsed_this = 0
        for a, b in _face_walk_undirected_edges(F, face_alive):
            if not vert_alive[a] or not vert_alive[b]:
                continue
            if (a, b) not in edge_faces:
                continue
            if float(np.linalg.norm(V[a] - V[b])) >= min_edge_length:
                continue
            if not _link_condition_ok_py(a, b, neighbors, edge_faces, F):
                continue
            keep, drop = int(b), int(a)
            mid = 0.5 * (V[a] + V[b])
            if poles is not None:
                if np.linalg.norm(mid - poles[keep]) > np.linalg.norm(mid - poles[drop]):
                    poles[keep] = poles[drop]
                    if pole_valid is not None:
                        pole_valid[keep] = pole_valid[drop]
            V[keep] = mid
            vert_alive[drop] = False
            _apply_collapse_local(keep, drop, F, face_alive, neighbors, edge_faces)
            collapsed_this += 1
        if collapsed_this == 0:
            break
        total += collapsed_this

    if total:
        F = F[face_alive]
        used = np.unique(F.reshape(-1)) if F.size else np.zeros(0, dtype=int)
        remap = -np.ones(V.shape[0], dtype=int)
        remap[used] = np.arange(used.shape[0], dtype=int)
        V = V[used]
        F = remap[F] if F.size else F
        if poles is not None:
            poles = poles[used]
        if pole_valid is not None:
            pole_valid = pole_valid[used]
    return V, F, total, poles, pole_valid


def test_numba_collapse_sweep_matches_python_reference():
    """The Numba collapse pass must reproduce the Python pass exactly."""
    meshes = [
        tm.creation.icosphere(subdivisions=2, radius=1.0),
        tm.creation.icosphere(subdivisions=3, radius=1.0),
        tm.creation.box(extents=(1.0, 2.0, 3.0)).subdivide().subdivide(),
        tm.creation.capsule(radius=0.4, height=2.0, count=(16, 16)),
    ]
    for mesh in meshes:
        V = np.asarray(mesh.vertices, dtype=float)
        F = np.asarray(mesh.faces, dtype=int)
        diag = float(np.linalg.norm(V.max(0) - V.min(0)))
        poles = V * 0.5
        valid = np.zeros(V.shape[0], dtype=bool)
        valid[::3] = True

        for frac in (0.05, 0.15, 0.3):
            min_len = frac * diag
            eV, eF, en, ep, ev = _collapse_reference(
                V, F, min_edge_length=min_len, poles=poles, pole_valid=valid
            )
            gV, gF, gn, _fixed, gp, gv = collapse_short_edges(
                V, F, min_edge_length=min_len, poles=poles, pole_valid=valid
            )
            label = f"{len(mesh.vertices)}v frac={frac}"
            assert gn == en, label
            assert np.array_equal(gV, eV), label
            assert np.array_equal(gF, eF), label
            assert np.array_equal(gp, ep), label
            assert np.array_equal(gv, ev), label


def test_select_obtuse_split_batch_matches_python():
    from pymcfs.remesh import _obtuse_split_candidates
    from pymcfs.topology import select_obtuse_split_batch

    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    thr = 110.0 * (np.pi / 180.0)
    selected = _obtuse_split_candidates(V, F, thr=thr, short_edge=1e-12)
    if selected.shape[0] == 0:
        return

    used_verts: set[int] = set()
    py_batch: list[tuple[int, ...]] = []
    for row in selected:
        u, v, f0, f1, s = (int(row[i]) for i in range(5))
        if u in used_verts or v in used_verts or s in used_verts:
            continue
        used_verts.update((u, v, s))
        py_batch.append((u, v, f0, f1, s))
    if not py_batch:
        py_batch = [tuple(int(x) for x in selected[0])]
    face_used: set[int] = set()
    clean: list[tuple[int, ...]] = []
    for item in py_batch:
        if item[2] in face_used or item[3] in face_used:
            continue
        face_used.update((item[2], item[3]))
        clean.append(item)
    py_batch = clean or [tuple(int(x) for x in selected[0])]

    nb_batch = select_obtuse_split_batch(selected, np.int32(F.shape[0]))
    assert nb_batch.shape[0] == len(py_batch)
    for i, item in enumerate(py_batch):
        assert tuple(int(x) for x in nb_batch[i]) == item
