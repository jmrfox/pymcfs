"""Tests for MCFS driver and remeshing."""
import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh as tm
import pytest

from pymcfs.laplacian import mcfs_cotangent_laplacian
from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.remesh import collapse_short_edges, split_obtuse_faces
from pymcfs.skeleton import skeletonize, thin_mesh
from pymcfs.validate import validate_mcfs_mesh


def test_collapse_short_edges_reduces_vertices():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    n0 = V.shape[0]
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    V2, F2, ncoll, *_ = collapse_short_edges(V, F, min_edge_length=0.15 * diag)
    assert ncoll > 0
    assert V2.shape[0] < n0
    assert F2.ndim == 2 and F2.shape[1] == 3
    assert np.all((F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0]))


def test_collapse_short_edges_golden_icosphere():
    """Bit-stable collapse vs pre-Numba golden on a fixed icosphere."""
    import hashlib

    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    V2, F2, ncoll, *_ = collapse_short_edges(V, F, min_edge_length=0.15 * diag)
    assert ncoll == 142
    assert V2.shape == (20, 3)
    assert F2.shape == (36, 3)
    digest = hashlib.sha256(
        np.ascontiguousarray(V2).tobytes() + np.ascontiguousarray(F2).tobytes()
    ).hexdigest()
    assert digest == "d4606e0df03de89e05e39eefa9c33c56d30409eedd63c84f18891b880e00c028"


def test_collapse_short_edges_carries_pole_valid_by_index():
    """A collapsed vertex inherits the validity of whichever pole it kept."""
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    poles = V * 0.5
    valid = np.zeros(V.shape[0], dtype=bool)
    valid[::2] = True

    V2, F2, ncoll, _, poles2, valid2 = collapse_short_edges(
        V, F, min_edge_length=0.15 * diag, poles=poles, pole_valid=valid
    )
    assert ncoll > 0
    assert valid2.shape[0] == V2.shape[0] == poles2.shape[0]

    validity_of_pole = {
        tuple(p): bool(v) for p, v in zip(poles.tolist(), valid.tolist())
    }
    for p, v in zip(poles2.tolist(), valid2.tolist()):
        assert validity_of_pole[tuple(p)] == v


def test_split_obtuse_faces_grows_pole_valid_with_new_vertices():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.01, 0.0],
            [0.5, -0.01, 0.0],
        ],
        dtype=float,
    )
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=int)
    valid = np.ones(V.shape[0], dtype=bool)
    V2, _F2, n, _fixed, _poles, valid2, _split = split_obtuse_faces(
        V, F, max_angle_deg=90.0, poles=V.copy(), pole_valid=valid
    )
    assert n >= 1
    assert valid2.shape[0] == V2.shape[0]
    # Pre-existing validity is preserved; new vertices are placeholders for the
    # caller's batched containment test.
    assert bool(valid2[: V.shape[0]].all())
    assert not bool(valid2[V.shape[0] :].any())


def test_split_obtuse_faces_runs():
    # Starlab splits an interior edge only when both incident triangles have
    # large opposite angles.
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.01, 0.0],
            [0.5, -0.01, 0.0],
        ],
        dtype=float,
    )
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=int)
    V2, F2, n, *_ = split_obtuse_faces(V, F, max_angle_deg=90.0)
    assert n >= 1
    assert V2.shape[0] > V.shape[0]
    assert F2.shape[0] >= 4


def test_mcfs_contract_reduces_area_and_vertices():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    n0, a0 = len(mesh.vertices), float(mesh.area)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.1,
        w_M=0.0,
        max_iterations=25,
        area_variation_factor=1e-5,
        verbose=False,
    )
    mcs.contract_until_convergence()
    assert mcs._surface_area() < a0
    assert mcs.V.shape[0] <= n0
    assert mcs.V.shape[0] < n0 or mcs.fixed.any()


def test_contract_geometry_uses_starlab_stacked_least_squares():
    """Guard against regressing to the amalgamated n×n system.

    Starlab / CGAL / the paper solve the stacked overdetermined system

        [W_L L; W_H; W_P] X ≈ [0; W_H V; W_P P]

    via normal equations. The older square amalgam

        (W_L L + W_H + W_P) X = W_H V + W_P P

    is a different operator and must stay rejected.
    """
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh, w_H=0.1, w_M=0.2, gate_exterior_poles=False, verbose=False
    )
    V0 = mcs.V.copy()
    wL, wH, wM = mcs._update_constraint_weights()
    L = mcfs_cotangent_laplacian(mcs.V, mcs.F).tocsr()
    diag = np.asarray(L.diagonal()).ravel()
    L_off = L - sp.diags(diag, format="csr", shape=L.shape)
    L_weighted = (sp.diags(wL) @ L_off) + sp.diags(diag, format="csr")
    A = sp.vstack([L_weighted, sp.diags(wH), sp.diags(wM)], format="csc")
    B = np.vstack([np.zeros_like(V0), wH[:, None] * V0, wM[:, None] * mcs.poles])
    assert A.shape == (3 * V0.shape[0], V0.shape[0])

    AtA = (A.T @ A).tocsc()
    AtB = A.T @ B
    amalgam = (sp.diags(wL) @ L + sp.diags(wH) + sp.diags(wM)).tocsc()
    # The two operators are not the same matrix.
    assert (AtA - amalgam).nnz > 0

    expected = np.column_stack([spla.spsolve(AtA, AtB[:, c]) for c in range(3)])
    mcs.contract_geometry()
    assert np.allclose(mcs.V, expected, atol=1e-9)
    assert float(mcs._surface_area()) < float(mesh.area)


def test_skeletonize_sphere_is_sparse():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    n0 = len(mesh.vertices)
    skel = skeletonize(mesh, max_iterations=30, w_H=0.1, w_M=0.0, compress_chains=True)
    assert skel.nodes.shape[0] > 0
    assert skel.edges.shape[0] > 0
    assert skel.nodes.shape[0] < n0
    G = skel.graph
    cycles = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)
    assert cycles <= 5


def test_thin_mesh_reduces_complexity():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    Vt, Ft = thin_mesh(mesh, max_iterations=20, w_M=0.0)
    assert Vt.shape[0] <= mesh.vertices.shape[0]
    assert Ft.shape[0] <= mesh.faces.shape[0]
    assert Vt.shape[0] < mesh.vertices.shape[0] or Ft.shape[0] < mesh.faces.shape[0]


def test_mcfs_timeout_stops():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.1,
        w_M=0.0,
        max_iterations=500,
        timeout_seconds=0.05,
        verbose=False,
    )
    t0 = time.monotonic()
    mcs.contract_until_convergence()
    assert time.monotonic() - t0 < 5.0


def test_collapse_does_not_raise_nameerror():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    diag = float(np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0)))
    V2, F2, n, *_ = collapse_short_edges(
        np.asarray(mesh.vertices, float),
        np.asarray(mesh.faces, int),
        min_edge_length=0.2 * diag,
    )
    assert isinstance(n, int)
    assert V2.ndim == 2


def test_validate_rejects_open_mesh():
    # Single triangle is not watertight
    mesh = tm.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    with pytest.raises(ValueError):
        validate_mcfs_mesh(mesh)


def _closed_cylinder(radius=0.5, height=2.0, radial=24, stacks=12):
    """Axially sampled watertight cylinder (trimesh's only has end rings)."""
    angles = np.linspace(0.0, 2.0 * np.pi, radial, endpoint=False)
    zs = np.linspace(-0.5 * height, 0.5 * height, stacks)
    ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    verts = [[float(x), float(y), float(z)] for z in zs for x, y in ring]
    top_c, bot_c = len(verts), len(verts) + 1
    verts.append([0.0, 0.0, 0.5 * height])
    verts.append([0.0, 0.0, -0.5 * height])
    faces: list[list[int]] = []
    for i in range(stacks - 1):
        for j in range(radial):
            j2 = (j + 1) % radial
            a, b = i * radial + j, i * radial + j2
            c, d = (i + 1) * radial + j2, (i + 1) * radial + j
            faces.extend([[a, b, c], [a, c, d]])
    base = (stacks - 1) * radial
    for j in range(radial):
        j2 = (j + 1) % radial
        faces.append([base + j, base + j2, top_c])
        faces.append([j2, j, bot_c])
    mesh = tm.Trimesh(vertices=np.asarray(verts, float), faces=np.asarray(faces, int), process=True)
    if not mesh.is_watertight:
        mesh.fix_normals()
    return mesh


def test_cylinder_skeleton_is_long_and_centered():
    mesh = _closed_cylinder()
    assert mesh.is_watertight
    skel = skeletonize(
        mesh,
        max_iterations=80,
        profile="starlab",
        timeout_seconds=60,
    )
    assert skel.nodes.shape[0] >= 2
    assert skel.edges.shape[0] >= 1
    total_len = float(
        sum(np.linalg.norm(skel.nodes[u] - skel.nodes[v]) for u, v in skel.edges)
    )
    # Expect a substantial fraction of the cylinder height (2.0)
    assert total_len > 0.8
    # Nodes should lie near the axis
    rad = np.linalg.norm(skel.nodes[:, :2], axis=1)
    assert float(rad.mean()) < 0.15
    G = skel.graph
    assert nx.number_connected_components(G) == 1
    assert G.number_of_edges() - G.number_of_nodes() + 1 == 0
    degrees = list(dict(G.degree()).values())
    assert degrees.count(1) == 2
    assert all(degree in (1, 2) for degree in degrees)
