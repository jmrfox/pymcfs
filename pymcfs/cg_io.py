"""Starlab Curve Graph (``.cg``) I/O.

Format (from ``curveskel_io_cg``)::

    # D:3 NV:<n> NE:<m>
    v x y z
    ...
    e i j          # 1-based vertex indices
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np


def write_cg(
    path: str | Path,
    nodes: np.ndarray,
    edges: np.ndarray,
) -> None:
    """Write an undirected curve skeleton as a Starlab ``.cg`` file.

    Parameters
    ----------
    path :
        Output file path (parent directories are created as needed).
    nodes : (n, 3) float
        Vertex positions.
    edges : (e, 2) int
        Undirected edges (0-based). Empty arrays are allowed.

    Raises
    ------
    ValueError
        If ``nodes`` or ``edges`` have the wrong shape.
    """
    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (n, 3)")
    if edges.size == 0:
        edges = np.zeros((0, 2), dtype=int)
    elif edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (e, 2)")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# D:3 NV:{nodes.shape[0]} NE:{edges.shape[0]}\n")
        for p in nodes:
            f.write(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        for u, v in edges:
            f.write(f"e {int(u) + 1} {int(v) + 1}\n")


def read_cg(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Starlab ``.cg`` file into ``(nodes, edges)`` (0-based edges).

    Parameters
    ----------
    path :
        Input ``.cg`` path.

    Returns
    -------
    nodes : (n, 3) float ndarray
    edges : (e, 2) int ndarray
        Converted from 1-based file indices.

    Raises
    ------
    ValueError
        If a line is malformed or unrecognized.
    """
    nodes: list[list[float]] = []
    edges: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            kind = parts[0].lower()
            if kind == "v":
                if len(parts) < 4:
                    raise ValueError(f"malformed vertex line: {line!r}")
                nodes.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif kind == "e":
                if len(parts) < 3:
                    raise ValueError(f"malformed edge line: {line!r}")
                edges.append([int(parts[1]) - 1, int(parts[2]) - 1])
            else:
                # Tolerate bare numeric rows (rare variants)
                if len(parts) == 3:
                    try:
                        nodes.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        continue
                    except ValueError:
                        pass
                raise ValueError(f"unrecognized .cg line: {line!r}")

    nodes_arr = (
        np.asarray(nodes, dtype=float) if nodes else np.zeros((0, 3), dtype=float)
    )
    edges_arr = (
        np.asarray(edges, dtype=int) if edges else np.zeros((0, 2), dtype=int)
    )
    return nodes_arr, edges_arr


def graph_from_cg(path: str | Path) -> nx.Graph:
    """Load a ``.cg`` file into a NetworkX graph with ``pos`` / ``weight`` attrs."""
    nodes, edges = read_cg(path)
    G = nx.Graph()
    for i, p in enumerate(nodes):
        G.add_node(i, pos=np.asarray(p, dtype=float))
    for u, v in edges:
        pu = np.asarray(G.nodes[int(u)]["pos"], dtype=float)
        pv = np.asarray(G.nodes[int(v)]["pos"], dtype=float)
        G.add_edge(int(u), int(v), weight=float(np.linalg.norm(pu - pv)))
    return G


def write_cg_from_graph(path: str | Path, G: nx.Graph) -> None:
    """Write a NetworkX skeleton graph (``pos`` attrs) as ``.cg``."""
    mapping = {n: i for i, n in enumerate(G.nodes)}
    nodes = np.array(
        [np.asarray(G.nodes[n]["pos"], dtype=float) for n in G.nodes],
        dtype=float,
    ) if G.number_of_nodes() else np.zeros((0, 3), dtype=float)
    edges = np.array(
        [[mapping[u], mapping[v]] for u, v in G.edges],
        dtype=int,
    ) if G.number_of_edges() else np.zeros((0, 2), dtype=int)
    write_cg(path, nodes, edges)
