"""Tests for exterior dangling-branch pruning."""
from __future__ import annotations

import networkx as nx
import numpy as np
import trimesh as tm

from pymcfs.cg_io import read_cg
from pymcfs.mesh import example_mesh
from pymcfs.quality import analyze_skeleton
from pymcfs.refine import prune_exterior_branches, prune_exterior_graph
from pymcfs.skeleton import Skeleton


def _skeleton_from_graph(G: nx.Graph) -> Skeleton:
    mapping = {n: i for i, n in enumerate(G.nodes)}
    G = nx.relabel_nodes(G, mapping, copy=True)
    nodes = np.array([G.nodes[n]["pos"] for n in G.nodes], dtype=float)
    edges = np.array([[u, v] for u, v in G.edges], dtype=int)
    return Skeleton(nodes=nodes, edges=edges, graph=G)


def test_prune_removes_exterior_leaf():
    mesh = example_mesh("cylinder", radius=0.5, height=2.0, sections=32)
    # Interior chain along axis + one exterior spur.
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, -0.5]))
    G.add_node(1, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(2, pos=np.array([0.0, 0.0, 0.5]))
    G.add_node(3, pos=np.array([2.0, 0.0, 0.0]))  # far outside
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)

    H, n_removed = prune_exterior_graph(G, mesh)
    assert n_removed == 1
    assert 3 not in H.nodes
    assert set(H.nodes) == {0, 1, 2}
    assert H.degree(1) == 2


def test_prune_keeps_interior_leaves():
    mesh = example_mesh("cylinder", radius=0.5, height=2.0, sections=32)
    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, -0.8]))
    G.add_node(1, pos=np.array([0.0, 0.0, 0.8]))
    G.add_edge(0, 1)
    H, n_removed = prune_exterior_graph(G, mesh)
    assert n_removed == 0
    assert H.number_of_nodes() == 2


def test_prune_ts4_raw_removes_long_exterior_spur():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    raw = root / "outputs/polylines/TS4/skeleton_raw.cg"
    mesh_path = root / "data/mesh/TS4.obj"
    if not raw.exists() or not mesh_path.exists():
        import pytest

        pytest.skip("TS4 outputs not present")

    mesh = tm.load(str(mesh_path), force="mesh", process=False)
    nodes, edges = read_cg(raw)
    G = nx.Graph()
    for i, p in enumerate(nodes):
        G.add_node(i, pos=np.asarray(p, dtype=float))
    for a, b in edges:
        G.add_edge(int(a), int(b))
    skel = _skeleton_from_graph(G)
    before = analyze_skeleton(mesh, skel)
    assert before.n_nodes_outside and before.n_nodes_outside >= 1

    pruned = prune_exterior_branches(skel, mesh)
    after = analyze_skeleton(mesh, pruned)
    # Long exterior leaf gone; may still have barely-outside mid-chain nodes.
    lens = np.linalg.norm(
        pruned.nodes[pruned.edges[:, 0]] - pruned.nodes[pruned.edges[:, 1]], axis=1
    ) if pruned.edges.size else np.zeros(0)
    assert lens.size == 0 or float(np.max(lens)) < 200.0
    assert after.n_nodes_outside is not None
    assert after.n_nodes_outside < before.n_nodes_outside
