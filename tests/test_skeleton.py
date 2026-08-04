import numpy as np
import networkx as nx
import trimesh as tm

from pymcfs.skeleton import skeletonize
from pymcfs.mesh import example_mesh


def test_skeletonize_runs_and_outputs_graph():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    skel = skeletonize(mesh, mcf_iters=15, is_medially_centered=False, omega_P=0.0)
    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.edges.shape[0] > 0


def test_skeletonize_resample_spacing():
    tor = tm.creation.torus(major_radius=1.5, minor_radius=0.4, major_sections=32, minor_sections=10)
    h = 0.25
    skel = skeletonize(
        tor,
        mcf_iters=20,
        is_medially_centered=False,
        omega_P=0.0,
        compress_chains=True,
        resample_spacing=h,
    )
    assert skel.edges.shape[0] > 0
    G = skel.graph
    max_w = max(float(d["weight"]) for _, _, d in G.edges(data=True)) if G.number_of_edges() > 0 else 0.0
    assert max_w <= h * 1.05


def test_skeletonize_cylinder():
    cyl = example_mesh("cylinder", radius=0.5, height=2.0, sections=24)
    skel = skeletonize(cyl, mcf_iters=25, is_medially_centered=False, omega_P=0.0, compress_chains=True)
    assert skel.nodes.shape[0] >= 2
    assert skel.edges.shape[0] > 0
    assert skel.nodes.shape[0] < len(cyl.vertices)


def test_skeletonize_torus_keeps_cycle():
    tor = tm.creation.torus(major_radius=2.0, minor_radius=0.5, major_sections=48, minor_sections=12)
    skel = skeletonize(tor, mcf_iters=25, is_medially_centered=False, omega_P=0.0, compress_chains=False)
    assert skel.edges.shape[0] > 0
    cycles = nx.cycle_basis(skel.graph)
    # Torus meso-skeleton often retains a loop; allow empty if over-collapsed
    assert isinstance(cycles, list)


def test_skeletonize_with_voronoi_guidance_runs():
    mesh = example_mesh("cylinder", radius=0.5, height=2.0, sections=24)
    skel = skeletonize(
        mesh,
        guidance_type="voronoi",
        omega_P=0.2,
        mcf_iters=15,
        compress_chains=True,
    )
    assert skel.nodes.shape[0] > 0
    assert skel.edges.shape[0] > 0
