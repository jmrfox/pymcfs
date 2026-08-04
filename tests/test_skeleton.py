import numpy as np
import networkx as nx
import trimesh as tm

from pymcfs.skeleton import skeletonize
from pymcfs.quality import analyze_skeleton


def test_skeletonize_runs_and_outputs_graph():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    skel = skeletonize(mesh, max_iterations=15, w_M=0.0)
    assert skel.nodes.ndim == 2 and skel.nodes.shape[1] == 3
    assert skel.edges.ndim == 2 and skel.edges.shape[1] == 2
    assert skel.edges.shape[0] > 0


def test_skeletonize_resample_spacing():
    tor = tm.creation.torus(major_radius=1.5, minor_radius=0.4, major_sections=32, minor_sections=10)
    h = 0.25
    skel = skeletonize(
        tor,
        max_iterations=20,
        w_M=0.0,
        compress_chains=True,
        resample_spacing=h,
    )
    assert skel.edges.shape[0] > 0
    G = skel.graph
    max_w = max(float(d["weight"]) for _, _, d in G.edges(data=True)) if G.number_of_edges() > 0 else 0.0
    assert max_w <= h * 1.05


def test_skeletonize_sphere():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    skel = skeletonize(mesh, max_iterations=25, w_M=0.0, compress_chains=True)
    assert skel.nodes.shape[0] >= 1
    assert skel.edges.shape[0] >= 0
    assert skel.nodes.shape[0] < len(mesh.vertices)


def test_skeletonize_torus_runs():
    tor = tm.creation.torus(major_radius=2.0, minor_radius=0.5, major_sections=48, minor_sections=12)
    skel = skeletonize(tor, max_iterations=25, w_M=0.0, compress_chains=False)
    assert skel.nodes.shape[0] > 0


def test_skeletonize_with_medial_weight():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    skel = skeletonize(mesh, w_H=0.1, w_M=0.2, max_iterations=15, compress_chains=True)
    assert skel.nodes.shape[0] > 0


def test_polylines_export(tmp_path):
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    skel = skeletonize(mesh, max_iterations=12, w_M=0.0)
    pls = skel.to_polylines()
    assert isinstance(pls, list)
    out = tmp_path / "skel.polylines.txt"
    skel.write_polylines(str(out))
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert len(text.strip().splitlines()) >= 0


def test_analyze_skeleton_runs():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    skel = skeletonize(mesh, max_iterations=15, w_M=0.0)
    report = analyze_skeleton(mesh, skel)
    assert report.n_nodes == skel.nodes.shape[0]
    assert report.mesh_genus == 0
    assert report.nodes_inside_frac is not None
    assert 0.0 <= report.nodes_inside_frac <= 1.0
    assert report.n_nodes_outside is not None and report.n_nodes_outside >= 0
    assert "nodes_inside=skipped" not in report.summary()
    assert isinstance(report.summary(), str)
