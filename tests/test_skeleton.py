import networkx as nx
import numpy as np
import trimesh as tm

from pymcfs.skeleton import (
    skeletonize,
    refine_skeleton,
    _resample_polyline_arc_length,
)
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
        refine=True,
        refine_spacing=h,
    )
    assert skel.edges.shape[0] > 0
    G = skel.graph
    max_w = max(float(d["weight"]) for _, _, d in G.edges(data=True)) if G.number_of_edges() > 0 else 0.0
    assert max_w <= h * 1.05


def test_skeletonize_refine_downsamples_and_evens():
    # Axially sampled cylinder (trimesh.creation.cylinder only has end rings).
    radial, stacks, radius, height = 24, 12, 0.5, 2.0
    verts = []
    for i in range(stacks + 1):
        z = -0.5 * height + height * (i / stacks)
        for j in range(radial):
            th = 2.0 * np.pi * j / radial
            verts.append([radius * np.cos(th), radius * np.sin(th), z])
    verts.append([0.0, 0.0, -0.5 * height])
    verts.append([0.0, 0.0, 0.5 * height])
    bottom_c, top_c = len(verts) - 2, len(verts) - 1
    faces = []
    for i in range(stacks):
        for j in range(radial):
            a = i * radial + j
            b = i * radial + (j + 1) % radial
            c = (i + 1) * radial + j
            d = (i + 1) * radial + (j + 1) % radial
            faces.extend([[a, b, d], [a, d, c]])
    for j in range(radial):
        a, b = j, (j + 1) % radial
        faces.append([bottom_c, b, a])
        a = stacks * radial + j
        b = stacks * radial + (j + 1) % radial
        faces.append([top_c, a, b])
    mesh = tm.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=True)

    raw = skeletonize(mesh, max_iterations=40, w_M=0.2, refine=False)
    refined = refine_skeleton(raw, mode="uniform")
    assert refined.nodes.shape[0] <= raw.nodes.shape[0]
    assert refined.edges.shape[0] > 0
    G = refined.graph
    assert nx.number_connected_components(G) == 1
    deg = dict(G.degree())
    assert sum(1 for d in deg.values() if d == 1) == 2


def test_refine_skeleton_compress_and_spacing_frac():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    skel = skeletonize(mesh, max_iterations=20, w_M=0.0, refine=False)
    compressed = refine_skeleton(skel, mode="compress")
    assert compressed.nodes.shape[0] <= skel.nodes.shape[0]
    assert all(d != 2 for _, d in compressed.graph.degree()) or compressed.graph.number_of_nodes() <= 2

    even = refine_skeleton(skel, mode="uniform", spacing_frac=0.05)
    assert even.nodes.shape[0] > 0
    lengths = [float(d["weight"]) for _, _, d in even.graph.edges(data=True)]
    if lengths:
        diag = float(np.linalg.norm(even.nodes.max(0) - even.nodes.min(0)))
        assert max(lengths) <= 0.05 * diag * 1.05 + 1e-9


def test_resample_polyline_preserves_endpoints():
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.15, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = _resample_polyline_arc_length(pts, spacing=0.25, closed=False)
    assert out.shape[0] >= 2
    assert np.allclose(out[0], pts[0])
    assert np.allclose(out[-1], pts[-1])
    segs = np.linalg.norm(out[1:] - out[:-1], axis=1)
    assert segs.max() <= 0.25 + 1e-9


def test_skeletonize_sphere():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    skel = skeletonize(mesh, max_iterations=25, w_M=0.0, refine="compress")
    assert skel.nodes.shape[0] >= 1
    assert skel.edges.shape[0] >= 0
    assert skel.nodes.shape[0] < len(mesh.vertices)


def test_skeletonize_torus_runs():
    tor = tm.creation.torus(major_radius=2.0, minor_radius=0.5, major_sections=48, minor_sections=12)
    skel = skeletonize(tor, max_iterations=25, w_M=0.0, refine=False)
    assert skel.nodes.shape[0] > 0
    G = skel.graph
    assert nx.number_connected_components(G) == 1
    cyclomatic = G.number_of_edges() - G.number_of_nodes() + 1
    assert cyclomatic == 1
    assert set(dict(G.degree()).values()) == {2}


def test_skeletonize_with_medial_weight():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    skel = skeletonize(mesh, w_H=0.1, w_M=0.2, profile="starlab", max_iterations=15, refine=True)
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
