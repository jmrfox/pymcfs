"""Tests for array-backed Numba mesh topology."""
from __future__ import annotations

import numpy as np
import trimesh as tm

from pymcfs.remesh import collapse_ok_for_edge, collapse_short_edges, _apply_collapse_local, _vertex_neighbors, _edge_to_faces
from pymcfs.topology import (
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
    V2, F2, ncoll, _, _ = collapse_short_edges(V, F, min_edge_length=0.15 * diag)
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
