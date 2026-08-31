"""Tests for tip extension toward lobe ends."""
from __future__ import annotations

import networkx as nx
import numpy as np
import trimesh as tm

from pymcfs.mesh import example_mesh
from pymcfs.refine import extend_tips, extend_tips_graph
from pymcfs.skeleton import Skeleton


def test_extend_tips_grows_short_cylinder_tip():
    """A tip stopped short of a cylinder end should move farther along +z."""
    mesh = example_mesh("cylinder", radius=0.5, height=2.0, sections=32)
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, -0.2]))
    G.add_node(1, pos=np.array([0.0, 0.0, 0.2]))
    G.add_edge(0, 1)

    H, n_added = extend_tips_graph(G, mesh, tip_extend_scale=1.0)
    assert n_added > 0
    zs = [float(H.nodes[n]["pos"][2]) for n in H.nodes]
    assert max(zs) > 0.7
    assert min(zs) < -0.7
    prox = tm.proximity.ProximityQuery(mesh)
    tips = [n for n in H.nodes if H.degree(n) == 1]
    tip_r = [
        float(np.abs(prox.signed_distance(np.asarray(H.nodes[n]["pos"]).reshape(1, 3))[0]))
        for n in tips
    ]
    # Tips should be near the surface relative to bbox (~diag ≈ 2.2).
    assert max(tip_r) < 0.15


def test_extend_tips_scale_zero_is_noop():
    mesh = example_mesh("cylinder", radius=0.5, height=2.0, sections=24)
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, -0.2]))
    G.add_node(1, pos=np.array([0.0, 0.0, 0.2]))
    G.add_edge(0, 1)
    H, n_added = extend_tips_graph(G, mesh, tip_extend_scale=0.0)
    assert n_added == 0
    assert H.number_of_nodes() == 2


def test_extend_tips_cone_recovers_from_bad_outward():
    """Cone search should find the lobe axis even if parent→tip is skewed."""
    mesh = example_mesh("cylinder", radius=0.5, height=2.0, sections=32)
    G = nx.Graph()
    # Parent near center; tip offset radially so tip-parent is not along +z.
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.15, 0.0, 0.15]))
    G.add_edge(0, 1)
    H, n_added = extend_tips_graph(
        G, mesh, tip_extend_scale=1.0, tip_cone_deg=50.0, n_cone=24
    )
    assert n_added > 0
    zs = [float(H.nodes[n]["pos"][2]) for n in H.nodes]
    assert max(abs(z) for z in zs) > 0.6


def test_extend_tips_wrapper():
    mesh = example_mesh("cylinder", radius=0.5, height=2.0, sections=24)
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, -0.3]))
    G.add_node(1, pos=np.array([0.0, 0.0, 0.3]))
    G.add_edge(0, 1)
    skel = Skeleton.from_graph(G)
    out = extend_tips(skel, mesh, tip_extend_scale=1.0)
    assert out.nodes.shape[0] > skel.nodes.shape[0]
