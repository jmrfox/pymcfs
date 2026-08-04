"""Tests for CGAL-style MCFS driver and remeshing."""
import numpy as np
import networkx as nx
import trimesh as tm

from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.remesh import collapse_short_edges, split_obtuse_faces
from pymcfs.skeleton import skeletonize, thin_mesh
from pymcfs.mesh import example_mesh


def test_collapse_short_edges_reduces_vertices():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int)
    n0 = V.shape[0]
    # Aggressive threshold relative to bbox
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    V2, F2, ncoll, _, _ = collapse_short_edges(V, F, min_edge_length=0.15 * diag)
    assert ncoll > 0
    assert V2.shape[0] < n0
    assert F2.ndim == 2 and F2.shape[1] == 3
    # No degenerate faces
    assert np.all((F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0]))


def test_split_obtuse_faces_runs():
    # A single very flat triangle
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.01, 0.0]], dtype=float)
    F = np.array([[0, 1, 2]], dtype=int)
    V2, F2, n, *_ = split_obtuse_faces(V, F, max_angle_deg=90.0)
    assert n >= 1
    assert V2.shape[0] > V.shape[0]
    assert F2.shape[0] >= 2


def test_mcfs_contract_reduces_area_and_vertices():
    mesh = example_mesh("cylinder", radius=0.4, height=2.0, sections=24)
    n0, a0 = len(mesh.vertices), float(mesh.area)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        omega_L=1.0,
        omega_H=0.1,
        omega_P=0.0,
        is_medially_centered=False,
        max_iterations=25,
        area_variation_factor=1e-5,
        verbose=False,
    )
    mcs.contract_until_convergence()
    assert mcs._surface_area() < a0
    assert mcs.V.shape[0] <= n0
    # Remeshing should have collapsed something on a contracting cylinder
    assert mcs.V.shape[0] < n0 or mcs.fixed.any()


def test_skeletonize_cylinder_is_sparse_curve():
    mesh = example_mesh("cylinder", radius=0.4, height=2.0, sections=24)
    n0 = len(mesh.vertices)
    skel = skeletonize(
        mesh,
        mcf_iters=30,
        is_medially_centered=False,
        omega_P=0.0,
        compress_chains=True,
    )
    assert skel.nodes.shape[0] > 0
    assert skel.edges.shape[0] > 0
    assert skel.nodes.shape[0] < n0
    # Genus-0 tube → near-tree: cyclomatic number should be small
    G = skel.graph
    cycles = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)
    assert cycles <= 5


def test_thin_mesh_reduces_complexity():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    Vt, Ft = thin_mesh(mesh, mcf_iters=20, is_medially_centered=False, omega_P=0.0)
    assert Vt.shape[0] <= mesh.vertices.shape[0]
    assert Ft.shape[0] <= mesh.faces.shape[0]
    assert Vt.shape[0] < mesh.vertices.shape[0] or Ft.shape[0] < mesh.faces.shape[0]


def test_collapse_does_not_raise_nameerror():
    """Regression: old thinning referenced undefined incident/neighbors."""
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    diag = float(np.linalg.norm(mesh.vertices.max(0) - mesh.vertices.min(0)))
    V2, F2, n, _, _ = collapse_short_edges(
        np.asarray(mesh.vertices, float),
        np.asarray(mesh.faces, int),
        min_edge_length=0.2 * diag,
    )
    assert isinstance(n, int)
    assert V2.ndim == 2
