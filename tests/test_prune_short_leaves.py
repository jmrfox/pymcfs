"""Tests for thickness-scaled short-leaf pruning."""
from __future__ import annotations

import networkx as nx
import numpy as np
import trimesh as tm

from pymcfs.mesh import example_mesh
from pymcfs.refine import prune_short_leaves, prune_short_leaves_graph
from pymcfs.skeleton import Skeleton


def _unit_sphere() -> tm.Trimesh:
    return tm.creation.icosphere(subdivisions=3, radius=1.0)


def test_prune_short_leaves_removes_volume_spurs():
    """Short spurs from a deep hub are removed; a long structural leaf remains."""
    # Unit sphere: center thickness ≈ 1.
    mesh = _unit_sphere()
    G = nx.Graph()
    # Hub near center (thick).
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    # Short spur (length 0.2 << radius ~1).
    G.add_node(1, pos=np.array([0.2, 0.0, 0.0]))
    G.add_edge(0, 1)
    # Another short spur.
    G.add_node(2, pos=np.array([0.0, 0.25, 0.0]))
    G.add_edge(0, 2)
    # Long structural branch (length ~0.95); keep with length_scale=0.5.
    G.add_node(3, pos=np.array([0.0, 0.0, 0.5]))
    G.add_node(4, pos=np.array([0.0, 0.0, 0.95]))
    G.add_edge(0, 3)
    G.add_edge(3, 4)

    H, n_removed = prune_short_leaves_graph(G, mesh, length_scale=0.5)
    assert n_removed >= 2
    assert 1 not in H.nodes
    assert 2 not in H.nodes
    assert 0 in H.nodes
    assert 4 in H.nodes
    assert nx.is_connected(H)


def test_prune_short_leaves_keeps_pure_path():
    """Leaf–leaf chains with no junction are not pruned."""
    mesh = example_mesh("cylinder", radius=0.5, height=2.0, sections=24)
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, -0.8]))
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([0.0, 0.0, 0.8]))
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    H, n_removed = prune_short_leaves_graph(G, mesh, length_scale=1.0)
    assert n_removed == 0
    assert H.number_of_nodes() == 3


def test_prune_short_leaves_skeleton_wrapper():
    mesh = _unit_sphere()
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.15, 0.0, 0.0]))
    G.add_node(2, pos=np.array([0.0, 0.15, 0.0]))
    G.add_node(3, pos=np.array([0.0, 0.0, 0.9]))
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    skel = Skeleton.from_graph(G)
    out = prune_short_leaves(skel, mesh, length_scale=0.5)
    assert out.nodes.shape[0] == 2
    assert out.edges.shape[0] == 1
