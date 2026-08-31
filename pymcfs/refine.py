"""Post-contraction refine phase: prune, tip extension, and curve resampling.

Refine improves curve-graph quality after meso-skeleton conversion.
Resample only changes node density along chains — it does not re-contract.
"""
from __future__ import annotations

import logging
from typing import Literal

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ResampleMode = Literal["uniform", "compress"]

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


def resolve_resample_options(
    *,
    resample: bool | ResampleMode = False,
    resample_spacing: float | None = None,
    resample_spacing_frac: float | None = None,
) -> tuple[ResampleMode | None, float | None, float | None]:
    """Map public resample flags to ``(mode, spacing, spacing_frac)``."""
    if resample is True:
        return "uniform", resample_spacing, resample_spacing_frac
    if isinstance(resample, str):
        if resample not in ("uniform", "compress"):
            raise ValueError(f"unknown resample mode: {resample!r}")
        return resample, resample_spacing, resample_spacing_frac  # type: ignore[return-value]
    return None, None, None


def resample_skeleton_graph(
    G: nx.Graph,
    *,
    mode: ResampleMode = "uniform",
    spacing: float | None = None,
    spacing_frac: float | None = None,
) -> nx.Graph:
    """Resample a skeleton curve graph to control node density (new graph).

    Parameters
    ----------
    G :
        Input curve graph with node attribute ``pos``.
    mode :
        ``\"uniform\"`` resample chains by arc length (default), or ``\"compress\"``
        to keep only junctions/leaves.
    spacing :
        Absolute target segment length for ``uniform``. If omitted, uses
        ``spacing_frac * bbox_diag``, else ``2 * median_edge_length``.
    spacing_frac :
        Relative spacing as a fraction of the skeleton axis-aligned bbox diagonal.

    Returns
    -------
    networkx.Graph

    Raises
    ------
    ValueError
        If ``mode`` is not ``uniform`` or ``compress``.
    """
    if G.number_of_nodes() == 0:
        return G
    if mode == "compress":
        logger.debug("resample_skeleton_graph: compress mode, n=%d", G.number_of_nodes())
        return _compress_degree_two_chains(G)
    if mode != "uniform":
        raise ValueError(f"unknown resample mode: {mode!r}")

    if spacing is not None and spacing > 0:
        target = float(spacing)
    elif spacing_frac is not None and spacing_frac > 0:
        target = float(spacing_frac) * _skeleton_bbox_diagonal(G)
    else:
        target = 2.0 * _median_edge_length(G)
    logger.debug(
        "resample_skeleton_graph: uniform spacing=%.4g, n=%d",
        target,
        G.number_of_nodes(),
    )
    return _resample_chains_uniform(G, target)


def resample_skeleton(
    skeleton: "Skeleton",
    *,
    mode: ResampleMode = "uniform",
    spacing: float | None = None,
    spacing_frac: float | None = None,
) -> "Skeleton":
    """Change curve-node density on a skeleton; returns a new instance.

    Part of the refine phase (post-contraction). Does not change connectivity
    topology beyond compressing degree-2 chains when ``mode=\"compress\"``.

    Parameters
    ----------
    skeleton :
        Input :class:`~pymcfs.skeleton.Skeleton`.
    mode :
        ``\"uniform\"`` (default) arc-length resample, or ``\"compress\"`` to keep
        only junctions/leaves.
    spacing :
        Absolute target segment length for ``uniform``.
    spacing_frac :
        Relative spacing as a fraction of the skeleton bbox diagonal.

    Returns
    -------
    Skeleton
    """
    from .skeleton import Skeleton

    G = resample_skeleton_graph(
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


def prune_exterior_graph(
    G: nx.Graph,
    mesh: "tm.Trimesh",
    *,
    fast: bool = False,
) -> tuple[nx.Graph, int]:
    """Remove dangling branches whose tips lie outside ``mesh``.

    Iteratively deletes degree-1 nodes classified outside the volume (exact
    float64 containment by default). This catches contraction leaks that
    survive surface→curve conversion as long exterior leaf edges — without
    changing interior topology.

    Parameters
    ----------
    G :
        Curve graph with node attribute ``pos``.
    mesh :
        Closed triangle mesh used for containment.
    fast :
        Forwarded to :func:`pymcfs.medial.points_inside_mesh`.

    Returns
    -------
    G_new :
        Pruned copy of ``G``.
    n_removed :
        Number of nodes deleted.
    """
    import trimesh as tm

    from .medial import points_inside_mesh

    if not isinstance(mesh, tm.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    if G.number_of_nodes() == 0:
        return G.copy(), 0

    H = G.copy()
    removed = 0
    while H.number_of_nodes() > 0:
        nodes = list(H.nodes)
        P = np.array([H.nodes[n]["pos"] for n in nodes], dtype=float)
        inside = points_inside_mesh(mesh, P, fast=bool(fast))
        outside = {n for n, ok in zip(nodes, inside) if not bool(ok)}
        if not outside:
            break
        leaves = [n for n in outside if H.degree(n) <= 1]
        if not leaves:
            # Exterior nodes remain but are not dangling tips (e.g. barely
            # outside mid-chain); leave them — aggressive mid-chain cuts risk
            # breaking real topology.
            break
        for n in leaves:
            if n in H:
                H.remove_node(n)
                removed += 1
    return H, removed


def prune_exterior_branches(
    skeleton: "Skeleton",
    mesh: "tm.Trimesh",
    *,
    fast: bool = False,
) -> "Skeleton":
    """Remove dangling curve tips that lie outside the input mesh volume.

    Parameters
    ----------
    skeleton :
        Input curve skeleton.
    mesh :
        Closed triangle mesh used for containment.
    fast :
        Forwarded to :func:`pymcfs.medial.points_inside_mesh`.

    Returns
    -------
    Skeleton
        New skeleton with exterior dangling tips removed.

    See Also
    --------
    prune_exterior_graph
    """
    from .skeleton import Skeleton

    G, _n = prune_exterior_graph(skeleton.graph, mesh, fast=fast)
    return _skeleton_from_relabeled_graph(G)


def _skeleton_from_relabeled_graph(G: nx.Graph) -> "Skeleton":
    from .skeleton import Skeleton

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


def _leaf_branch_to_junction(G: nx.Graph, leaf: int) -> tuple[list, object | None, float]:
    """Walk from ``leaf`` along degree-2 nodes to the next non-degree-2 node.

    Returns
    -------
    path_nodes :
        Nodes strictly between ``leaf`` and the junction (excludes junction;
        includes ``leaf``). Empty if ``leaf`` is isolated.
    junction :
        Attachment node with degree ≠ 2, or None if the component is a pure
        path ending at another leaf / missing.
    length :
        Arc length of the path from ``leaf`` to ``junction`` (0 if none).
    """
    if leaf not in G or G.degree(leaf) != 1:
        return [], None, 0.0
    nbrs = list(G.neighbors(leaf))
    if not nbrs:
        return [leaf], None, 0.0

    path = [leaf]
    prev, curr = leaf, nbrs[0]
    length = float(
        np.linalg.norm(
            np.asarray(G.nodes[prev]["pos"], dtype=float)
            - np.asarray(G.nodes[curr]["pos"], dtype=float)
        )
    )
    while G.degree(curr) == 2:
        path.append(curr)
        nxts = [x for x in G.neighbors(curr) if x != prev]
        if not nxts:
            return path, None, length
        nxt = nxts[0]
        length += float(
            np.linalg.norm(
                np.asarray(G.nodes[curr]["pos"], dtype=float)
                - np.asarray(G.nodes[nxt]["pos"], dtype=float)
            )
        )
        prev, curr = curr, nxt

    # curr is junction (deg >= 3) or the other leaf (deg 1).
    if G.degree(curr) < 3:
        return path, None, length
    return path, curr, length


def prune_short_leaves_graph(
    G: nx.Graph,
    mesh: "tm.Trimesh",
    *,
    length_scale: float = 1.0,
) -> tuple[nx.Graph, int]:
    """Remove leaf branches shorter than ``length_scale`` × local thickness.

    Mild micro-spur filter only. Radial volume-filling arms at thick
    high-degree hubs are handled by :func:`prune_thick_hubs_graph`.

    Local thickness is the absolute signed distance from the **attachment
    junction** to ``mesh``. Short spurs are removed; long structural leaves
    and pure paths (leaf–leaf chains with no junction) are kept.

    Iterates until no leaf branch meets the criterion.

    Parameters
    ----------
    G :
        Curve graph with node attribute ``pos``.
    mesh :
        Closed triangle mesh used for thickness queries.
    length_scale :
        Multiplier on junction thickness. Branch arc length ``L`` is removed
        when ``L < length_scale * r_junction``. Default ``1.0``.

    Returns
    -------
    G_new :
        Pruned copy of ``G``.
    n_removed :
        Number of nodes deleted.
    """
    import trimesh as tm

    if not isinstance(mesh, tm.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    if G.number_of_nodes() == 0:
        return G.copy(), 0
    if float(length_scale) <= 0.0:
        return G.copy(), 0

    H = G.copy()
    removed = 0
    prox = tm.proximity.ProximityQuery(mesh)
    scale = float(length_scale)

    while H.number_of_nodes() > 0:
        leaves = [n for n in H.nodes if H.degree(n) == 1]
        to_remove: list = []
        for leaf in leaves:
            path, junction, length = _leaf_branch_to_junction(H, leaf)
            if junction is None or not path:
                continue
            jpos = np.asarray(H.nodes[junction]["pos"], dtype=float).reshape(1, 3)
            radius = float(np.abs(prox.signed_distance(jpos)[0]))
            if not np.isfinite(radius) or radius <= 0.0:
                continue
            if length < scale * radius:
                to_remove.extend(path)

        # Unique while preserving graph integrity (paths disjoint by construction
        # except at junctions, which are not in ``path``).
        uniq = []
        seen: set = set()
        for n in to_remove:
            if n not in seen and n in H:
                seen.add(n)
                uniq.append(n)
        if not uniq:
            break
        for n in uniq:
            H.remove_node(n)
            removed += 1
    return H, removed


def prune_short_leaves(
    skeleton: "Skeleton",
    mesh: "tm.Trimesh",
    *,
    length_scale: float = 1.0,
) -> "Skeleton":
    """Remove leaf branches shorter than local thickness (mild spur filter).

    For radial volume-filling arms at thick high-degree hubs, use
    :func:`prune_thick_hubs` instead.

    Parameters
    ----------
    skeleton :
        Input curve skeleton.
    mesh :
        Closed triangle mesh for thickness queries.
    length_scale :
        Multiplier on junction thickness (default 1.0).

    Returns
    -------
    Skeleton

    See Also
    --------
    prune_short_leaves_graph
    """
    G, _n = prune_short_leaves_graph(
        skeleton.graph, mesh, length_scale=float(length_scale)
    )
    return _skeleton_from_relabeled_graph(G)


def _mesh_bbox_diag(mesh: "tm.Trimesh") -> float:
    V = np.asarray(mesh.vertices, dtype=float)
    if V.size == 0:
        return 1.0
    bb = V.max(axis=0) - V.min(axis=0)
    return float(max(np.linalg.norm(bb), 1e-12))


def _leaf_arm_from_hub(
    G: nx.Graph, hub: object, neighbor: object
) -> tuple[list, float] | None:
    """If the chain hub→neighbor ends at a leaf, return (path_nodes, length).

    ``path_nodes`` excludes ``hub`` and includes the leaf. Returns None if the
    chain ends at another junction (non-leaf skeleton continuation).
    """
    if neighbor not in G or hub not in G:
        return None
    prev, curr = hub, neighbor
    path = [curr]
    length = float(
        np.linalg.norm(
            np.asarray(G.nodes[prev]["pos"], dtype=float)
            - np.asarray(G.nodes[curr]["pos"], dtype=float)
        )
    )
    while G.degree(curr) == 2:
        nxts = [x for x in G.neighbors(curr) if x != prev]
        if not nxts:
            break
        nxt = nxts[0]
        length += float(
            np.linalg.norm(
                np.asarray(G.nodes[curr]["pos"], dtype=float)
                - np.asarray(G.nodes[nxt]["pos"], dtype=float)
            )
        )
        path.append(nxt)
        prev, curr = curr, nxt
    if G.degree(curr) != 1:
        return None
    return path, length


def prune_thick_hubs_graph(
    G: nx.Graph,
    mesh: "tm.Trimesh",
    *,
    keep_hub_branches: int = 2,
    hub_degree_min: int = 4,
    hub_radius_frac: float = 0.015,
) -> tuple[nx.Graph, int]:
    """At thick high-degree hubs, keep only the longest leaf arms.

    Radial volume-filling spokes in thick compartments often have length
    comparable to local radius, so :func:`prune_short_leaves_graph` misses them.
    This pass targets hubs with ``degree >= hub_degree_min`` and
    ``|sd| >= hub_radius_frac * bbox_diag``, keeping the ``keep_hub_branches``
    longest leaf-side chains and deleting the rest. Non-leaf arms (connections
    into the rest of the skeleton) are never removed.

    Parameters
    ----------
    G :
        Curve graph with node attribute ``pos``.
    mesh :
        Closed triangle mesh for thickness queries.
    keep_hub_branches :
        Number of leaf arms to retain per thick hub (default 2).
    hub_degree_min :
        Minimum degree to treat a node as a hub (default 4; skips ordinary
        Y-junctions).
    hub_radius_frac :
        Minimum ``|signed_distance| / bbox_diag`` for a hub to count as thick.

    Returns
    -------
    G_new :
        Pruned copy of ``G``.
    n_removed :
        Number of nodes deleted.
    """
    import trimesh as tm

    if not isinstance(mesh, tm.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    if G.number_of_nodes() == 0:
        return G.copy(), 0
    keep = max(int(keep_hub_branches), 0)
    deg_min = max(int(hub_degree_min), 3)
    frac = float(max(hub_radius_frac, 0.0))
    if keep <= 0:
        return G.copy(), 0

    H = G.copy()
    prox = tm.proximity.ProximityQuery(mesh)
    diag = _mesh_bbox_diag(mesh)
    r_min = frac * diag
    removed = 0

    while True:
        hubs = [n for n in list(H.nodes) if H.degree(n) >= deg_min]
        removed_this = 0
        for hub in hubs:
            if hub not in H or H.degree(hub) < deg_min:
                continue
            hpos = np.asarray(H.nodes[hub]["pos"], dtype=float).reshape(1, 3)
            r_j = float(np.abs(prox.signed_distance(hpos)[0]))
            if not np.isfinite(r_j) or r_j < r_min:
                continue
            arms: list[tuple[float, list]] = []
            for nbr in list(H.neighbors(hub)):
                arm = _leaf_arm_from_hub(H, hub, nbr)
                if arm is None:
                    continue
                path, length = arm
                arms.append((float(length), path))
            if len(arms) <= keep:
                continue
            arms.sort(key=lambda t: t[0], reverse=True)
            for _length, path in arms[keep:]:
                for n in path:
                    if n in H and n != hub:
                        H.remove_node(n)
                        removed += 1
                        removed_this += 1
        if removed_this == 0:
            break
    return H, removed


def prune_thick_hubs(
    skeleton: "Skeleton",
    mesh: "tm.Trimesh",
    *,
    keep_hub_branches: int = 2,
    hub_degree_min: int = 4,
    hub_radius_frac: float = 0.015,
) -> "Skeleton":
    """At thick high-degree hubs, keep only the longest leaf arms.

    Parameters
    ----------
    skeleton :
        Input curve skeleton.
    mesh :
        Closed triangle mesh for thickness queries.
    keep_hub_branches :
        Number of leaf arms to retain per thick hub (default 2).
    hub_degree_min :
        Minimum degree to treat a node as a hub (default 4).
    hub_radius_frac :
        Minimum ``|signed_distance| / bbox_diag`` for a hub to count as thick.

    Returns
    -------
    Skeleton

    See Also
    --------
    prune_thick_hubs_graph
    """
    G, _n = prune_thick_hubs_graph(
        skeleton.graph,
        mesh,
        keep_hub_branches=int(keep_hub_branches),
        hub_degree_min=int(hub_degree_min),
        hub_radius_frac=float(hub_radius_frac),
    )
    return _skeleton_from_relabeled_graph(G)


def _pull_inside(
    mesh: "tm.Trimesh",
    prox: object,
    inside_fn,
    p_inside: np.ndarray,
    p_outside: np.ndarray,
    *,
    iters: int = 10,
) -> np.ndarray:
    """Binary-search the segment back to a point still classified inside."""
    lo = np.asarray(p_inside, dtype=float).reshape(3)
    hi = np.asarray(p_outside, dtype=float).reshape(3)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if bool(inside_fn(mid.reshape(1, 3))[0]):
            lo = mid
        else:
            hi = mid
    return lo


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(axis, dtype=float).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1e-15)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, tmp)
    u = u / max(float(np.linalg.norm(u)), 1e-15)
    v = np.cross(a, u)
    return u, v


def _cone_directions(
    axis: np.ndarray,
    *,
    cone_deg: float,
    n_samples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Unit directions in a cone about ``axis``, including ``axis`` itself."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-15)
    dirs = [axis.copy()]
    u, v = _orthonormal_basis(axis)
    half = np.deg2rad(float(cone_deg))
    for _ in range(max(int(n_samples), 0)):
        # Sample polar angle in [0, half], azimuth uniform.
        cos_a = 1.0 - rng.random() * (1.0 - np.cos(half))
        sin_a = float(np.sqrt(max(1.0 - cos_a * cos_a, 0.0)))
        phi = float(rng.random() * 2.0 * np.pi)
        d = cos_a * axis + sin_a * (np.cos(phi) * u + np.sin(phi) * v)
        nrm = float(np.linalg.norm(d))
        if nrm > 1e-15:
            dirs.append(d / nrm)
    return dirs


def _interior_travel(
    mesh: "tm.Trimesh",
    prox: object,
    inside_fn,
    start: np.ndarray,
    direction: np.ndarray,
    *,
    max_travel: float,
    step: float,
) -> float:
    """How far ``start`` can march along ``direction`` while staying inside."""
    curr = np.asarray(start, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    traveled = 0.0
    for _ in range(512):
        if traveled >= max_travel - 1e-15:
            break
        step_now = min(step, max_travel - traveled)
        cand = curr + step_now * direction
        if not bool(inside_fn(cand.reshape(1, 3))[0]):
            cand = _pull_inside(mesh, prox, inside_fn, curr, cand)
            delta = float(np.linalg.norm(cand - curr))
            if delta < 1e-12 * max(step, 1.0):
                break
            traveled += delta
            break
        curr = cand
        traveled += step_now
    return traveled


def extend_tips_graph(
    G: nx.Graph,
    mesh: "tm.Trimesh",
    *,
    tip_extend_scale: float = 1.0,
    tip_clearance_frac: float = 0.01,
    tip_cone_deg: float = 40.0,
    step_frac: float = 0.08,
    n_cone: int = 16,
    seed: int = 0,
) -> tuple[nx.Graph, int]:
    """Grow unfinished leaf tips toward the end of their lobe.

    For each degree-1 tip still deep inside ``mesh``, search directions in a
    cone about the outward branch axis, pick the longest interior ray, and
    march until near the surface or a bbox-scaled travel budget is exhausted.

    Intended for open-ended shapes without end-caps. Leave disabled for
    general / Starlab-like use.

    Parameters
    ----------
    G :
        Curve graph with node attribute ``pos``.
    mesh :
        Closed triangle mesh.
    tip_extend_scale :
        Max travel distance as a multiple of the mesh bbox diagonal
        (default ``1.0``). Larger → stronger extension. ``0`` disables.
    tip_clearance_frac :
        Stop when ``|signed_distance| / bbox_diag`` is at most this value
        (default ``0.01``).
    tip_cone_deg :
        Cone half-angle in degrees for direction search (default ``40``).
    step_frac :
        Step size as a fraction of ``max(tip_thickness, 0.01 * diag)``.
    n_cone :
        Extra cone sample directions (plus the outward axis).
    seed :
        RNG seed for cone sampling.

    Returns
    -------
    G_new :
        Copy of ``G`` with optional new nodes along extended tips.
    n_added :
        Number of nodes inserted.
    """
    import trimesh as tm

    from .medial import points_inside_mesh

    if not isinstance(mesh, tm.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    if G.number_of_nodes() == 0:
        return G.copy(), 0
    scale = float(tip_extend_scale)
    if scale <= 0.0:
        return G.copy(), 0
    clr = float(np.clip(tip_clearance_frac, 1e-6, 0.5))
    step_f = float(np.clip(step_frac, 1e-3, 0.5))
    cone_deg = float(max(tip_cone_deg, 0.0))

    H = G.copy()
    prox = tm.proximity.ProximityQuery(mesh)
    diag = _mesh_bbox_diag(mesh)
    stop_r = clr * diag
    max_travel_budget = scale * diag
    leaves = [n for n in list(H.nodes) if H.degree(n) == 1]
    if not leaves:
        return H, 0

    next_id = 0
    int_ids = [int(n) for n in H.nodes if isinstance(n, (int, np.integer))]
    if int_ids:
        next_id = max(int_ids) + 1
    else:
        while next_id in H:
            next_id += 1

    rng = np.random.default_rng(int(seed))
    added = 0

    def _inside(pts: np.ndarray) -> np.ndarray:
        return points_inside_mesh(mesh, pts, fast=False)

    for leaf in leaves:
        if leaf not in H or H.degree(leaf) != 1:
            continue
        parent = next(iter(H.neighbors(leaf)))
        tip = np.asarray(H.nodes[leaf]["pos"], dtype=float).reshape(3)
        ppos = np.asarray(H.nodes[parent]["pos"], dtype=float).reshape(3)
        outward = tip - ppos
        nrm = float(np.linalg.norm(outward))
        if not np.isfinite(nrm) or nrm < 1e-15:
            continue
        outward = outward / nrm

        r0 = float(np.abs(prox.signed_distance(tip.reshape(1, 3))[0]))
        if not np.isfinite(r0) or r0 <= stop_r:
            continue

        step = max(step_f * max(r0, 0.01 * diag), stop_r * 0.5)
        dirs = _cone_directions(
            outward, cone_deg=cone_deg, n_samples=int(n_cone), rng=rng
        )
        best_dir = outward
        best_travel = -1.0
        for d in dirs:
            travel = _interior_travel(
                mesh,
                prox,
                _inside,
                tip,
                d,
                max_travel=max_travel_budget,
                step=step,
            )
            if travel > best_travel:
                best_travel = travel
                best_dir = d
        if best_travel < step * 0.5:
            continue

        curr = tip.copy()
        traveled = 0.0
        prev_node = leaf
        for _ in range(512):
            if traveled >= max_travel_budget - 1e-15:
                break
            r_curr = float(np.abs(prox.signed_distance(curr.reshape(1, 3))[0]))
            if r_curr <= stop_r:
                break
            step_now = min(step, max_travel_budget - traveled)
            cand = curr + step_now * best_dir
            if not bool(_inside(cand.reshape(1, 3))[0]):
                cand = _pull_inside(mesh, prox, _inside, curr, cand)
                delta = float(np.linalg.norm(cand - curr))
                if delta < 1e-12 * max(diag, 1.0):
                    break
                step_now = delta
            nid = next_id
            next_id += 1
            while nid in H:
                nid = next_id
                next_id += 1
            H.add_node(nid, pos=np.asarray(cand, dtype=float))
            H.add_edge(
                prev_node,
                nid,
                weight=float(np.linalg.norm(cand - curr)),
            )
            prev_node = nid
            curr = cand
            traveled += step_now
            added += 1
            if float(np.abs(prox.signed_distance(curr.reshape(1, 3))[0])) <= stop_r:
                break

    return H, added


def extend_tips(
    skeleton: "Skeleton",
    mesh: "tm.Trimesh",
    *,
    tip_extend_scale: float = 1.0,
    tip_clearance_frac: float = 0.01,
    tip_cone_deg: float = 40.0,
    step_frac: float = 0.08,
    n_cone: int = 16,
    seed: int = 0,
) -> "Skeleton":
    """Grow unfinished leaf tips toward the end of their lobe.

    Intended for open-ended shapes without end-caps; leave off for general use.

    Parameters
    ----------
    skeleton :
        Input curve skeleton.
    mesh :
        Closed triangle mesh.
    tip_extend_scale :
        Max travel as a multiple of bbox diagonal (default 1.0). ``0`` disables.
    tip_clearance_frac :
        Stop near the surface when ``|signed_distance| / bbox_diag`` is at most
        this value (default 0.01).
    tip_cone_deg :
        Cone half-angle in degrees for direction search (default 40).
    step_frac, n_cone, seed :
        Step size fraction, cone sample count, and RNG seed.

    Returns
    -------
    Skeleton

    See Also
    --------
    extend_tips_graph
    """
    G, _n = extend_tips_graph(
        skeleton.graph,
        mesh,
        tip_extend_scale=float(tip_extend_scale),
        tip_clearance_frac=float(tip_clearance_frac),
        tip_cone_deg=float(tip_cone_deg),
        step_frac=float(step_frac),
        n_cone=int(n_cone),
        seed=int(seed),
    )
    return _skeleton_from_relabeled_graph(G)
