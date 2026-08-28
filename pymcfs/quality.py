"""Skeleton quality analysis relative to the input surface mesh."""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import trimesh as tm

from .skeleton import Skeleton

# Hard-reject floor used when remesh growth / numerical failure aborts a run.
_REJECT_SCORE = -1.0e9


@dataclass
class SkeletonQualityReport:
    """Summary of how well a skeleton describes an input mesh.

    Attributes
    ----------
    n_nodes, n_edges, n_components, n_junctions, n_leaves :
        Curve-graph size and degree counts (junction = degree ≥ 3).
    nodes_inside_frac, n_nodes_outside, exterior_node_indices :
        Node containment vs the mesh volume (None when skipped / failed).
    edges_inside_frac, n_edges_outside, exterior_edge_indices :
        Edge sample containment (None when skipped / failed).
    mesh_genus, mesh_euler_chi :
        Mesh topology when estimable from Euler characteristic.
    skeleton_cyclomatic :
        ``E - V + C`` for the skeleton graph.
    topology_consistent :
        Soft consistency check used by :func:`analyze_skeleton`: True when
        ``cyclomatic >= genus`` and components are compatible (not the same as
        the stricter equality preferred by :func:`score_skeleton`).
    mean_node_distance_to_surface, max_node_distance_to_surface :
        Absolute signed-distance magnitudes to the surface when available.
    """

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
        """Compact multi-field summary string for logs."""
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


@dataclass
class SkeletonScore:
    """Ordered fitness for parameter sweeps and the MCFS oracle.

    Higher ``value`` is better. Rejected runs sit near ``_REJECT_SCORE``.
    Ranking priorities (largest influence first):
    topology match (cyclomatic == genus), containment, then compactness.

    Attributes
    ----------
    value :
        Scalar fitness (higher better).
    rejected :
        True for hard-fail runs (remesh growth, nonfinite, area overshoot).
    reject_reason :
        Short reason when ``rejected``.
    topology_delta :
        ``skeleton_cyclomatic - mesh_genus`` when genus is known.
    nodes_inside_frac, edges_inside_frac, n_nodes_outside :
        Copied containment fields from the report when available.
    n_junctions, n_nodes, n_leaves, n_components :
        Compactness / connectivity counts used in the score.
    """

    value: float
    rejected: bool = False
    reject_reason: str | None = None
    topology_delta: int | None = None
    nodes_inside_frac: float | None = None
    edges_inside_frac: float | None = None
    n_nodes_outside: int | None = None
    n_junctions: int = 0
    n_nodes: int = 0
    n_leaves: int = 0
    n_components: int = 0

    def summary(self) -> str:
        """One-line score summary for logs."""
        if self.rejected:
            return f"rejected value={self.value:.4g} reason={self.reject_reason}"
        return (
            f"value={self.value:.4g} topo_delta={self.topology_delta} "
            f"inside={self.nodes_inside_frac} outside={self.n_nodes_outside} "
            f"junctions={self.n_junctions} nodes={self.n_nodes}"
        )

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

    Checks:
    - fraction of nodes inside the mesh volume
    - fraction of edges whose samples stay inside
    - topological consistency: mesh genus vs skeleton cyclomatic number
      (``cyclomatic >= genus`` soft check; see :attr:`SkeletonQualityReport.topology_consistent`)
    - distance of nodes to the surface

    Parameters
    ----------
    mesh :
        Source surface mesh.
    skeleton :
        Curve skeleton to evaluate.
    edge_samples :
        Interior samples per edge for containment (endpoints excluded).
    check_contains :
        If False, skip containment / surface-distance queries.

    Returns
    -------
    SkeletonQualityReport

    Raises
    ------
    TypeError
        If ``mesh`` or ``skeleton`` has the wrong type.
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


def score_skeleton(
    report: SkeletonQualityReport,
    *,
    remesh_growth_rejected: bool = False,
    remesh_growth_ratio: float | None = None,
    nonfinite: bool = False,
    area_overshoot: bool = False,
) -> SkeletonScore:
    """Rank a skeleton for parameter selection.

    Hard rejects (remesh blow-up, non-finite verts, area overshoot) score near
    ``_REJECT_SCORE``. Otherwise favors, in order:

    1. ``skeleton_cyclomatic == mesh_genus`` (penalize missing *and* excess cycles)
    2. nodes/edges inside the mesh (exterior nodes heavily penalized)
    3. compactness (fewer junctions, then fewer nodes/leaves)

    Parameters
    ----------
    report :
        Output of :func:`analyze_skeleton`.
    remesh_growth_rejected :
        True if contraction aborted on vertex-count growth
        (:attr:`~pymcfs.mcfs.MeanCurvatureFlowSkeletonization.aborted_remesh_growth`).
    remesh_growth_ratio :
        Optional ``n_max / n0`` recorded in the reject reason.
    nonfinite :
        True if non-finite coordinates were observed during contraction.
    area_overshoot :
        True if surface area grew beyond the driver's sanity threshold
        (:attr:`~pymcfs.mcfs.MeanCurvatureFlowSkeletonization.area_overshoot_seen`).

    Returns
    -------
    SkeletonScore
    """
    if remesh_growth_rejected:
        ratio = remesh_growth_ratio if remesh_growth_ratio is not None else float("nan")
        return SkeletonScore(
            value=_REJECT_SCORE + 1.0e6,
            rejected=True,
            reject_reason=f"remesh_growth ratio={ratio:.3g}",
            n_junctions=report.n_junctions,
            n_nodes=report.n_nodes,
            n_leaves=report.n_leaves,
            n_components=report.n_components,
        )
    if nonfinite:
        return SkeletonScore(
            value=_REJECT_SCORE + 2.0e6,
            rejected=True,
            reject_reason="nonfinite_vertices",
            n_junctions=report.n_junctions,
            n_nodes=report.n_nodes,
            n_leaves=report.n_leaves,
            n_components=report.n_components,
        )
    if area_overshoot:
        return SkeletonScore(
            value=_REJECT_SCORE + 3.0e6,
            rejected=True,
            reject_reason="area_overshoot",
            n_junctions=report.n_junctions,
            n_nodes=report.n_nodes,
            n_leaves=report.n_leaves,
            n_components=report.n_components,
        )

    value = 0.0
    topo_delta: int | None = None
    if report.mesh_genus is not None:
        topo_delta = int(report.skeleton_cyclomatic - report.mesh_genus)
        if topo_delta < 0:
            # Missing cycles for topological holes — severe.
            value -= 1.0e6 * float(-topo_delta)
        elif topo_delta > 0:
            # Extra loops / spurious cycles — still bad, milder than missing.
            value -= 1.0e4 * float(topo_delta)
        else:
            value += 1.0e5
        if report.n_components > 1 and report.mesh_genus > 0:
            value -= 1.0e4 * float(report.n_components - 1)
        elif report.n_components > 1:
            value -= 1.0e3 * float(report.n_components - 1)

    nodes_inside = report.nodes_inside_frac
    edges_inside = report.edges_inside_frac
    n_outside = report.n_nodes_outside
    if nodes_inside is not None:
        value += 1.0e3 * float(nodes_inside)
    if edges_inside is not None:
        value += 5.0e2 * float(edges_inside)
    if n_outside is not None:
        value -= 1.0e4 * float(n_outside)

    value -= 10.0 * float(report.n_junctions)
    value -= 0.1 * float(report.n_nodes)
    value -= 1.0 * float(report.n_leaves)

    return SkeletonScore(
        value=float(value),
        rejected=False,
        reject_reason=None,
        topology_delta=topo_delta,
        nodes_inside_frac=nodes_inside,
        edges_inside_frac=edges_inside,
        n_nodes_outside=n_outside,
        n_junctions=int(report.n_junctions),
        n_nodes=int(report.n_nodes),
        n_leaves=int(report.n_leaves),
        n_components=int(report.n_components),
    )
