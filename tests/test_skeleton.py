import numpy as np
import networkx as nx
import trimesh as tm

from pymcfs.skeleton import skeletonize


def test_skeletonize_runs_and_outputs_graph():
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=1)
    skel = skeletonize(mesh, mcf_dt=2e-2, mcf_iters=5, knn=6, length_quantile=0.7)

    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    # With cycles allowed, just ensure we have edges
    assert skel.edges.shape[0] > 0


def test_skeletonize_mesh_only_mode_runs():
    # Exercise mesh-only collapse domain
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    skel = skeletonize(
        mesh,
        build_graph="mesh",
        collapse_domain="mesh_only",
        collapse_mode="pq",
        collapse_ratio=0.2,
        collapse_passes=2,
        compress_chains=True,
    )

    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.edges.shape[0] > 0


def test_skeletonize_resample_spacing():
    # After chain compression, resample to uniform spacing and check segment lengths
    tor = tm.creation.torus(major_radius=1.5, minor_radius=0.4, major_sections=48, minor_sections=12)
    h = 0.2
    skel = skeletonize(
        tor,
        build_graph="knn",
        mcf_dt=1e-2,
        mcf_iters=8,
        knn=12,
        collapse_passes=2,
        compress_chains=True,
        resample_spacing=h,
    )

    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.edges.shape[0] > 0
    # Max edge weight should be <= spacing (allow tiny numerical slack)
    G = skel.graph
    max_w = max(float(d["weight"]) for _, _, d in G.edges(data=True)) if G.number_of_edges() > 0 else 0.0
    assert max_w <= h * 1.05


def test_skeletonize_medial_protect_runs():
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    skel = skeletonize(
        mesh,
        build_graph="mesh",
        guidance_type="voronoi",
        guidance_weight=1.0,
        medial_protect=True,
        medial_protect_threshold=0.5,
        mcf_dt=1e-2,
        mcf_iters=8,
        collapse_passes=1,
    )

    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.edges.shape[0] > 0


def test_skeletonize_pq_mode_runs():
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    skel = skeletonize(
        mesh,
        build_graph="mesh",
        mcf_dt=1e-2,
        mcf_iters=5,
        collapse_passes=2,
        collapse_mode="pq",
        collapse_ratio=0.2,
    )

    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.edges.shape[0] > 0


def test_skeletonize_chain_compression_runs():
    tor = tm.creation.torus(major_radius=1.5, minor_radius=0.4, major_sections=48, minor_sections=12)
    skel = skeletonize(
        tor,
        build_graph="knn",
        mcf_dt=1e-2,
        mcf_iters=8,
        knn=10,
        collapse_passes=2,
        compress_chains=True,
    )

    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.edges.shape[0] > 0


def test_skeletonize_cylinder():
    # Create a simple vertical cylinder
    cyl = tm.primitives.Cylinder(radius=1.0, height=3.0, sections=32)
    skel = skeletonize(cyl, mcf_dt=2e-2, mcf_iters=10, knn=8, length_quantile=0.6)

    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.nodes.shape[0] > 5
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    # Allow cycles; just ensure we have some connectivity
    assert skel.edges.shape[0] > 0


def test_skeletonize_torus():
    # Create a torus; note MST will break the loop by design
    tor = tm.creation.torus(major_radius=2.0, minor_radius=0.5, major_sections=64, minor_sections=16)
    skel = skeletonize(tor, mcf_dt=1e-2, mcf_iters=15, knn=12, length_quantile=0.6)

    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.nodes.shape[0] > 8
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.edges.shape[0] > 0

    # With cycles allowed, a torus should yield at least one cycle in the pruned graph
    cycles = nx.cycle_basis(skel.graph)
    assert len(cycles) >= 1


def test_skeletonize_mesh_graph_mode():
    # Use mesh connectivity to build the initial graph
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    skel = skeletonize(
        mesh,
        build_graph="mesh",
        mcf_dt=1e-2,
        mcf_iters=5,
        knn=8,
        collapse_passes=2,
    )

    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.edges.shape[0] > 0


def test_skeletonize_with_voronoi_guidance_runs():
    # Voronoi guidance should run and return a valid structure
    mesh = tm.primitives.Cylinder(radius=0.5, height=2.0, sections=48)
    skel = skeletonize(
        mesh,
        build_graph="mesh",
        guidance_type="voronoi",
        guidance_weight=1.0,
        mcf_dt=1e-2,
        mcf_iters=10,
        knn=10,
        collapse_passes=2,
    )

    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.edges.shape[0] > 0
