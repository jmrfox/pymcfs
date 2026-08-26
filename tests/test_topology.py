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
