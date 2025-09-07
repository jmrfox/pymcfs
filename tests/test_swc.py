import io
import os
import math
import numpy as np
import networkx as nx

from pymcfs.skeleton import Skeleton


def _make_graph(nodes: np.ndarray, edges: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    for i, p in enumerate(nodes):
        G.add_node(i, pos=np.asarray(p, dtype=float))
    for (u, v) in edges:
        u = int(u); v = int(v)
        w = float(np.linalg.norm(nodes[u] - nodes[v]))
        G.add_edge(u, v, weight=w)
    return G


def _parse_swc(path: str):
    rows = []
    header = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                header.append(line)
                continue
            parts = line.split()
            if len(parts) != 7:
                raise AssertionError("SWC row must have 7 columns")
            n = int(parts[0]); T = int(parts[1])
            x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
            R = float(parts[5]); P = int(parts[6])
            rows.append((n, T, x, y, z, R, P))
    return header, rows


def test_swc_basic_write(tmp_path):
    nodes = np.array([[0,0,0],[1,0,0],[2,0,0]], dtype=float)
    edges = np.array([[0,1],[1,2]], dtype=int)
    G = _make_graph(nodes, edges)
    skel = Skeleton(nodes=nodes, edges=edges, graph=G)

    out = tmp_path / 'simple.swc'
    skel.write_swc(str(out))

    assert out.exists(), "SWC file should be created"
    header, rows = _parse_swc(str(out))
    assert any('# Columns:' in h for h in header)
    assert len(rows) == nodes.shape[0]
    # Parent should always be -1 or smaller index
    for (n, T, x, y, z, R, P) in rows:
        if P != -1:
            assert P < n


def test_swc_cycle_break_annotation(tmp_path):
    # Triangle (single cycle)
    nodes = np.array([[0,0,0],[1,0,0],[0.5,0.866,0]], dtype=float)
    edges = np.array([[0,1],[1,2],[2,0]], dtype=int)
    G = _make_graph(nodes, edges)
    skel = Skeleton(nodes=nodes, edges=edges, graph=G)

    out = tmp_path / 'triangle.swc'
    skel.write_swc(str(out), break_cycles='mst', annotate=True)

    header, rows = _parse_swc(str(out))
    # Should annotate at least one removed edge
    ann = [h for h in header if h.startswith('# removed_edge')]
    assert len(ann) >= 1
    # Still have one row per node
    assert len(rows) == nodes.shape[0]


def test_swc_bfs_parent_precedes_child(tmp_path):
    # Branching structure
    nodes = np.array([
        [0,0,0], [1,0,0], [2,0,0],  # chain 0-1-2
        [1,1,0], [1,-1,0]           # branches from node 1
    ], dtype=float)
    edges = np.array([[0,1],[1,2],[1,3],[1,4]], dtype=int)
    G = _make_graph(nodes, edges)
    skel = Skeleton(nodes=nodes, edges=edges, graph=G)

    out = tmp_path / 'branch.swc'
    skel.write_swc(str(out))

    header, rows = _parse_swc(str(out))
    # Parent must precede child in SWC ordering
    id_to_idx = {n: i for i, (n, *_rest) in enumerate(rows)}
    for i, (n, T, x, y, z, R, P) in enumerate(rows):
        if P != -1:
            assert id_to_idx[P] < i
