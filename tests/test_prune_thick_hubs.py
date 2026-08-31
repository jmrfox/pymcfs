"""Tests for thick-hub principal-branch pruning."""
from __future__ import annotations

import networkx as nx
import numpy as np
import trimesh as tm

from pymcfs.refine import prune_thick_hubs, prune_thick_hubs_graph
from pymcfs.skeleton import Skeleton


def _unit_sphere() -> tm.Trimesh:
    return tm.creation.icosphere(subdivisions=3, radius=1.0)


def test_prune_thick_hubs_keeps_longest_arms():
    mesh = _unit_sphere()
    G = nx.Graph()
    # Hub at center (thick); five leaf arms of unequal length.
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    arms = [
        (1, np.array([0.3, 0.0, 0.0])),  # short
        (2, np.array([0.0, 0.35, 0.0])),
        (3, np.array([-0.4, 0.0, 0.0])),
        (4, np.array([0.0, 0.0, 0.7])),  # long
        (5, np.array([0.0, 0.0, -0.85])),  # longest
    ]
    for i, p in arms:
        G.add_node(i, pos=p)
        G.add_edge(0, i)

    H, n_removed = prune_thick_hubs_graph(
        G, mesh, keep_hub_branches=2, hub_degree_min=4, hub_radius_frac=0.015
    )
    assert n_removed == 3
    leaves = [n for n in H.nodes if H.degree(n) == 1]
    assert len(leaves) == 2
    assert 4 in H.nodes and 5 in H.nodes
    assert 1 not in H.nodes and 2 not in H.nodes and 3 not in H.nodes


def test_prune_thick_hubs_skips_y_junction():
    mesh = _unit_sphere()
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.5, 0.0, 0.0]))
    G.add_node(2, pos=np.array([0.0, 0.5, 0.0]))
    G.add_node(3, pos=np.array([0.0, 0.0, 0.5]))
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    H, n_removed = prune_thick_hubs_graph(
        G, mesh, keep_hub_branches=2, hub_degree_min=4, hub_radius_frac=0.015
    )
    assert n_removed == 0
    assert H.number_of_nodes() == 4


def test_prune_thick_hubs_skips_thin_high_degree():
    """High-degree hub near the surface (thin) should not be culled."""
    mesh = _unit_sphere()
    G = nx.Graph()
    # Hub very near surface: |sd| << hub_radius_frac * diag (~0.04).
    G.add_node(0, pos=np.array([0.99, 0.0, 0.0]))
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, 5, endpoint=False)):
        p = np.array([0.99, 0.02 * np.cos(ang), 0.02 * np.sin(ang)])
        G.add_node(i + 1, pos=p)
        G.add_edge(0, i + 1)
    H, n_removed = prune_thick_hubs_graph(
        G, mesh, keep_hub_branches=2, hub_degree_min=4, hub_radius_frac=0.015
    )
    assert n_removed == 0
    assert H.number_of_nodes() == 6


def test_prune_thick_hubs_wrapper():
    mesh = _unit_sphere()
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    for i, z in enumerate([0.2, 0.3, 0.4, 0.8, 0.9]):
        G.add_node(i + 1, pos=np.array([0.0, 0.0, z if i % 2 == 0 else -z]))
        G.add_edge(0, i + 1)
    skel = Skeleton.from_graph(G)
    out = prune_thick_hubs(skel, mesh, keep_hub_branches=2)
    assert sum(1 for n in out.graph.nodes if out.graph.degree(n) == 1) == 2
