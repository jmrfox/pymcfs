"""Skeleton curve-graph refinement helpers."""
from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

RefineMode = Literal["uniform", "compress"]

def _compress_degree_two_chains(G: nx.Graph) -> nx.Graph:
    """Compress sequences of degree-2 nodes into single edges connecting junctions.

    - Junctions are nodes with degree != 2; degree-1 leaves and degree>=3 junctions remain.
    - Chains between junctions become a single edge with weight equal to the sum of
      intermediate edge weights. Node positions for junctions are preserved.
    - If the entire graph is a single cycle (all degree==2), return G unchanged.
    """
    if G.number_of_nodes() == 0:
        return G
    deg = dict(G.degree())
    junctions = [n for n, d in deg.items() if d != 2]
    if len(junctions) == 0:
        # likely a pure cycle; leave as-is
        return G

    NG = nx.Graph()
    # Copy junction nodes with positions
    for n in junctions:
        NG.add_node(n, pos=np.array(G.nodes[n]["pos"]))

    visited: set[tuple[int, int]] = set()
    for u in junctions:
        for v in G.neighbors(u):
            if (u, v) in visited or (v, u) in visited:
                continue
            path_len = float(G[u][v].get("weight", np.linalg.norm(G.nodes[u]["pos"] - G.nodes[v]["pos"])) )
            prev = u
            curr = v
            visited.add((u, v))
            # Walk forward through degree-2 nodes
            while deg.get(curr, 0) == 2 and curr not in junctions:
                nbrs = list(G.neighbors(curr))
                nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
                path_len += float(G[curr][nxt].get("weight", np.linalg.norm(G.nodes[curr]["pos"] - G.nodes[nxt]["pos"])) )
                prev, curr = curr, nxt
                visited.add((prev, curr))

            # Now curr is a junction (or leaf)
            a, b = u, curr
            if a == b:
                continue
            w = path_len
            # Ensure nodes exist
            if a not in NG:
                NG.add_node(a, pos=np.array(G.nodes[a]["pos"]))
            if b not in NG:
                NG.add_node(b, pos=np.array(G.nodes[b]["pos"]))
            if NG.has_edge(a, b):
                # keep minimal weight if duplicate, though duplicates should be rare
                if w < NG[a][b]["weight"]:
                    NG[a][b]["weight"] = w
            else:
                NG.add_edge(a, b, weight=w)

    return NG


def _edge_euclidean(G: nx.Graph, u: int, v: int) -> float:
    data = G[u][v]
    if "weight" in data:
        return float(data["weight"])
    pu = np.asarray(G.nodes[u]["pos"], dtype=float)
    pv = np.asarray(G.nodes[v]["pos"], dtype=float)
    return float(np.linalg.norm(pu - pv))


def _median_edge_length(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 1.0
    lengths = [_edge_euclidean(G, u, v) for u, v in G.edges()]
    med = float(np.median(lengths))
    return med if med > 0 else float(np.mean(lengths) or 1.0)


def _skeleton_bbox_diagonal(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 1.0
    P = np.asarray([G.nodes[n]["pos"] for n in G.nodes], dtype=float)
    return float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0


def _iter_skeleton_chains(G: nx.Graph) -> list[tuple[list[int], bool]]:
    """Yield ``(node_path, closed)`` chains between terminals, or closed cycles."""
    if G.number_of_nodes() == 0:
        return []
    deg = dict(G.degree())
    terminals = {n for n, d in deg.items() if d != 2}
    chains: list[tuple[list[int], bool]] = []
    visited: set[tuple[int, int]] = set()

    def edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    if not terminals:
        # Pure cycle component(s): emit one closed loop per connected component.
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp)
            if sub.number_of_edges() == 0:
                continue
            start = next(iter(sub.nodes))
            cycle_edges = nx.find_cycle(sub, source=start)
            path = [u for u, _v in cycle_edges]
            chains.append((path, True))
        return chains

    for t in terminals:
        for nbr in G.neighbors(t):
            ek = edge_key(t, nbr)
            if ek in visited:
                continue
            path = [t, nbr]
            visited.add(ek)
            prev, curr = t, nbr
            while deg.get(curr, 0) == 2:
                nxts = [x for x in G.neighbors(curr) if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                visited.add(edge_key(curr, nxt))
                path.append(nxt)
                prev, curr = curr, nxt
            chains.append((path, False))
    return chains


def _resample_polyline_arc_length(
    points: np.ndarray,
    spacing: float,
    *,
    closed: bool = False,
) -> np.ndarray:
    """Resample a polyline (or closed loop) to roughly uniform arc-length spacing."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[0] == 0:
        return P.copy()
    if P.shape[0] == 1:
        return P.copy()
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    if closed:
        segs = np.linalg.norm(np.vstack([P[1:], P[:1]]) - P, axis=1)
    else:
        segs = np.linalg.norm(P[1:] - P[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(segs)])
    total = float(cum[-1])
    if total <= 1e-15:
        return P[:1].copy() if not closed else P[:1].copy()

    if closed:
        n_segs = max(3, int(np.ceil(total / spacing - 1e-12)))
        targets = (np.arange(n_segs, dtype=float) * (total / n_segs)) % total
    else:
        n_segs = max(1, int(np.ceil(total / spacing - 1e-12)))
        targets = np.linspace(0.0, total, n_segs + 1)

    # Extend for interpolation lookup on open curves; for closed, wrap.
    if closed:
        P_ext = np.vstack([P, P[0]])
        cum_ext = cum  # already includes full loop length at end
    else:
        P_ext = P
        cum_ext = cum

    out = np.empty((targets.shape[0], 3), dtype=float)
    for i, t in enumerate(targets):
        # Find segment with cum_ext[j] <= t <= cum_ext[j+1]
        j = int(np.searchsorted(cum_ext, t, side="right") - 1)
        j = max(0, min(j, len(cum_ext) - 2))
        t0, t1 = cum_ext[j], cum_ext[j + 1]
        if t1 <= t0:
            out[i] = P_ext[j]
            continue
        alpha = (t - t0) / (t1 - t0)
        out[i] = (1.0 - alpha) * P_ext[j] + alpha * P_ext[j + 1]
    if not closed:
        out[0] = P[0]
        out[-1] = P[-1]
    return out


def _resample_chains_uniform(G: nx.Graph, spacing: float) -> nx.Graph:
    """Resample each junction-to-junction chain by arc length at target ``spacing``.

    Junctions and leaves keep their positions; degree-2 samples are rebuilt so
    spacing is more even and typically slightly coarser than the raw MCFS curve.
    """
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return G
    spacing = float(spacing)
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    chains = _iter_skeleton_chains(G)
    if not chains:
        return G

    NG = nx.Graph()
    node_map: dict[int, int] = {}

    def ensure_terminal(n: int) -> int:
        if n in node_map:
            return node_map[n]
        nid = NG.number_of_nodes()
        NG.add_node(nid, pos=np.asarray(G.nodes[n]["pos"], dtype=float).copy())
        node_map[n] = nid
        return nid

    for path, closed in chains:
        pts = np.asarray([G.nodes[n]["pos"] for n in path], dtype=float)
        sampled = _resample_polyline_arc_length(pts, spacing, closed=closed)
        if closed:
            ids = []
            for pos in sampled:
                nid = NG.number_of_nodes()
                NG.add_node(nid, pos=np.asarray(pos, dtype=float))
                ids.append(nid)
            for a, b in zip(ids, ids[1:] + ids[:1]):
                w = float(np.linalg.norm(NG.nodes[a]["pos"] - NG.nodes[b]["pos"]))
                NG.add_edge(a, b, weight=w)
            continue

        ids: list[int] = []
        for i, pos in enumerate(sampled):
            if i == 0:
                ids.append(ensure_terminal(path[0]))
            elif i == len(sampled) - 1:
                ids.append(ensure_terminal(path[-1]))
            else:
                nid = NG.number_of_nodes()
                NG.add_node(nid, pos=np.asarray(pos, dtype=float))
                ids.append(nid)
        for a, b in zip(ids[:-1], ids[1:]):
            if a == b:
                continue
            w = float(np.linalg.norm(NG.nodes[a]["pos"] - NG.nodes[b]["pos"]))
            NG.add_edge(a, b, weight=w)

    return NG


def resolve_refine_options(
    *,
    refine: bool | RefineMode = False,
    refine_spacing: float | None = None,
    refine_spacing_frac: float | None = None,
    compress_chains: bool = False,
    resample_spacing: float | None = None,
) -> tuple[RefineMode | None, float | None, float | None]:
    """Map public refine / legacy flags to ``(mode, spacing, spacing_frac)``."""
    if refine is True:
        return "uniform", refine_spacing, refine_spacing_frac
    if isinstance(refine, str):
        if refine not in ("uniform", "compress"):
            raise ValueError(f"unknown refine mode: {refine!r}")
        return refine, refine_spacing, refine_spacing_frac  # type: ignore[return-value]
    # Legacy aliases when refine is left off.
    if resample_spacing is not None and float(resample_spacing) > 0:
        return "uniform", float(resample_spacing), None
    if compress_chains:
        return "compress", None, None
    return None, None, None


def refine_skeleton_graph(
    G: nx.Graph,
    *,
    mode: RefineMode = "uniform",
    spacing: float | None = None,
    spacing_frac: float | None = None,
) -> nx.Graph:
    """Refine a skeleton curve graph in-place-style (returns a new graph).

    Parameters
    ----------
    mode :
        ``"uniform"`` resample chains by arc length (default), or ``"compress"``
        to keep only junctions/leaves.
    spacing :
        Absolute target segment length for ``uniform``. If omitted, uses
        ``spacing_frac * bbox_diag``, else ``2 * median_edge_length``.
    spacing_frac :
        Relative spacing as a fraction of the skeleton axis-aligned bbox diagonal.
    """
    if G.number_of_nodes() == 0:
        return G
    if mode == "compress":
        return _compress_degree_two_chains(G)
    if mode != "uniform":
        raise ValueError(f"unknown refine mode: {mode!r}")

    if spacing is not None and spacing > 0:
        target = float(spacing)
    elif spacing_frac is not None and spacing_frac > 0:
        target = float(spacing_frac) * _skeleton_bbox_diagonal(G)
    else:
        target = 2.0 * _median_edge_length(G)
    return _resample_chains_uniform(G, target)


def refine_skeleton(
    skeleton: "Skeleton",
    *,
    mode: RefineMode = "uniform",
    spacing: float | None = None,
    spacing_frac: float | None = None,
) -> "Skeleton":
    """Refine a :class:`~pymcfs.skeleton.Skeleton` curve graph; returns a new instance."""
    from .skeleton import Skeleton

    G = refine_skeleton_graph(
        skeleton.graph, mode=mode, spacing=spacing, spacing_frac=spacing_frac
    )
    mapping = {n: i for i, n in enumerate(G.nodes)}
    if mapping:
        G = nx.relabel_nodes(G, mapping, copy=True)
    nodes_arr = (
        np.array([G.nodes[n]["pos"] for n in G.nodes], dtype=float)
        if G.number_of_nodes()
        else np.zeros((0, 3))
    )
    edges_arr = (
        np.array([[u, v] for u, v in G.edges], dtype=int)
        if G.number_of_edges()
        else np.zeros((0, 2), dtype=int)
    )
    return Skeleton(nodes=nodes_arr, edges=edges_arr, graph=G)
