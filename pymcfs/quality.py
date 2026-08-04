"""Skeleton quality analysis relative to the input surface mesh."""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import trimesh as tm

from .skeleton import Skeleton


@dataclass
class SkeletonQualityReport:
    """Summary of how well a skeleton describes an input mesh."""

    n_nodes: int
    n_edges: int
    n_components: int
    n_junctions: int
    n_leaves: int

    nodes_inside_frac: float | None
    n_nodes_outside: int | None
    exterior_node_indices: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=int))

    edges_inside_frac: float | None = 1.0
    n_edges_outside: int | None = 0
    exterior_edge_indices: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=int))

    mesh_genus: int | None = None
    mesh_euler_chi: int | None = None
    skeleton_cyclomatic: int = 0
    topology_consistent: bool | None = None

    mean_node_distance_to_surface: float | None = None
    max_node_distance_to_surface: float | None = None

    def summary(self) -> str:
        lines = [
            f"nodes={self.n_nodes} edges={self.n_edges} components={self.n_components}",
            f"junctions={self.n_junctions} leaves={self.n_leaves}",
        ]
        if self.nodes_inside_frac is not None and self.n_nodes_outside is not None:
            lines.append(
                f"nodes_inside={self.nodes_inside_frac:.3f} ({self.n_nodes_outside} outside)"
            )
        else:
            lines.append("nodes_inside=skipped")
        if self.edges_inside_frac is not None and self.n_edges_outside is not None:
            lines.append(
                f"edges_inside={self.edges_inside_frac:.3f} ({self.n_edges_outside} exit)"
            )
        else:
            lines.append("edges_inside=skipped")
        lines.append(f"skeleton_cyclomatic={self.skeleton_cyclomatic}")
        if self.mesh_genus is not None:
            lines.append(f"mesh_genus={self.mesh_genus} topology_ok={self.topology_consistent}")
        if self.mean_node_distance_to_surface is not None:
            lines.append(
                f"mean_dist_to_surface={self.mean_node_distance_to_surface:.4g} "
                f"max={self.max_node_distance_to_surface:.4g}"
            )
        return "; ".join(lines)


def _mesh_genus(mesh: tm.Trimesh) -> tuple[int | None, int | None]:
    try:
        chi = int(mesh.euler_number)
    except Exception:
        return None, None
    if mesh.is_watertight and chi % 2 == 0:
        g = (2 - chi) // 2
        if g >= 0:
            return int(g), chi
    return None, chi


def _cyclomatic(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    return int(G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G))


def analyze_skeleton(
    mesh: tm.Trimesh,
    skeleton: Skeleton,
    *,
    edge_samples: int = 8,
    check_contains: bool = True,
) -> SkeletonQualityReport:
    """Compare a skeleton to its source mesh.

    Checks (extensible):
    - fraction of nodes inside the mesh volume
    - fraction of edges whose samples stay inside
    - topological consistency: mesh genus vs skeleton cyclomatic number
    - distance of nodes to the surface
    """
    if not isinstance(mesh, tm.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    if not isinstance(skeleton, Skeleton):
        raise TypeError("skeleton must be a Skeleton")

    G = skeleton.graph
    nodes = np.asarray(skeleton.nodes, dtype=float)
    edges = np.asarray(skeleton.edges, dtype=int)
    n_nodes = int(nodes.shape[0]) if nodes.size else 0
    n_edges = int(edges.shape[0]) if edges.size else G.number_of_edges()

    deg = dict(G.degree()) if G.number_of_nodes() else {}
    n_junctions = sum(1 for d in deg.values() if d >= 3)
    n_leaves = sum(1 for d in deg.values() if d == 1)
    n_components = nx.number_connected_components(G) if G.number_of_nodes() else 0
    cyc = _cyclomatic(G)
    genus, chi = _mesh_genus(mesh)
    if genus is None:
        topo_ok: bool | None = None
    else:
        topo_ok = (cyc >= genus) and (n_components == 1 or genus == 0)

    exterior_nodes = np.zeros(0, dtype=int)
    nodes_inside_frac: float | None = 1.0
    n_outside: int | None = 0
    mean_d = max_d = None
    if n_nodes > 0 and check_contains:
        try:
            inside = np.asarray(mesh.contains(nodes), dtype=bool)
            exterior_nodes = np.where(~inside)[0]
            n_outside = int(exterior_nodes.size)
            nodes_inside_frac = float(np.mean(inside)) if inside.size else 1.0
        except Exception:
            nodes_inside_frac = None
            n_outside = None
        try:
            prox = tm.proximity.ProximityQuery(mesh)
            d = np.abs(prox.signed_distance(nodes))
            mean_d = float(np.mean(d))
            max_d = float(np.max(d))
        except Exception:
            mean_d = max_d = None

    exterior_edges: list[int] = []
    edges_inside_frac: float | None = 1.0
    n_edges_out: int | None = 0
    if n_edges > 0 and n_nodes > 0 and check_contains and edge_samples > 0:
        try:
            E = edges if edges.size else np.array(list(G.edges), dtype=int)
            for ei, (a, b) in enumerate(E):
                a, b = int(a), int(b)
                if a >= n_nodes or b >= n_nodes:
                    continue
                ts = np.linspace(0.0, 1.0, edge_samples + 2)[1:-1]
                pts = (1.0 - ts)[:, None] * nodes[a] + ts[:, None] * nodes[b]
                if not bool(np.all(mesh.contains(pts))):
                    exterior_edges.append(ei)
            n_edges_out = len(exterior_edges)
            edges_inside_frac = 1.0 - (n_edges_out / max(len(E), 1))
        except Exception:
            edges_inside_frac = None
            n_edges_out = None
    elif not check_contains:
        nodes_inside_frac = None
        n_outside = None
        edges_inside_frac = None
        n_edges_out = None

    return SkeletonQualityReport(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_components=int(n_components),
        n_junctions=int(n_junctions),
        n_leaves=int(n_leaves),
        nodes_inside_frac=nodes_inside_frac,
        n_nodes_outside=n_outside,
        exterior_node_indices=np.asarray(exterior_nodes, dtype=int),
        edges_inside_frac=edges_inside_frac,
        n_edges_outside=n_edges_out,
        exterior_edge_indices=np.asarray(exterior_edges, dtype=int),
        mesh_genus=genus,
        mesh_euler_chi=chi,
        skeleton_cyclomatic=cyc,
        topology_consistent=topo_ok,
        mean_node_distance_to_surface=mean_d,
        max_node_distance_to_surface=max_d,
    )
