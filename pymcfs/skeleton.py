"""Public API for extracting a 1D curve skeleton from a closed triangle mesh.

``skeletonize`` runs contraction → curve conversion → refine.
``contract_mesh`` stops after the contracted meso-skeleton surface.
Text export writes polylines (``.polylines.txt``) only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import logging
import networkx as nx
import numpy as np
import trimesh as tm

from .config import (
    ContractionSettings,
    McfsProfile,
    RefineSettings,
    ResampleMode,
    SkeletonizeSettings,
)
from .params import BranchingPreference, McfsParams
from .refine import (
    resample_skeleton,
    resample_skeleton_graph,
    resolve_resample_options,
    prune_exterior_branches,
    prune_exterior_graph,
    prune_short_leaves,
    prune_short_leaves_graph,
    prune_thick_hubs,
    prune_thick_hubs_graph,
    extend_tips,
    extend_tips_graph,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Application / robust defaults (complex tubular meshes).
_ROBUST_ATTRACTION = 0.5
_ROBUST_MEDIAL = 5.0
# Starlab mcfskel defaults (parity harness).
_STARLAB_ATTRACTION = 0.1
_STARLAB_MEDIAL = 0.2


def resolve_mcfs_profile(
    profile: McfsProfile | None,
    *,
    attraction_weight: float,
    medial_weight: float,
    gate_exterior_poles: bool | None,
    mesh: tm.Trimesh | None = None,
    branching: str = "sparse",
) -> tuple[float, float, bool]:
    """Pick contraction weights and pole-gating from a named profile.

    Attraction weight keeps vertices from sliding too far; medial weight pulls
    toward interior Voronoi poles (approximate medial axis). Gating skips poles
    that lie outside the mesh volume.

    - ``profile=\"starlab\"`` — Starlab weights (0.1 / 0.2), ungated poles,
      unless the caller overrode weights or set ``gate_exterior_poles``.
    - ``profile=None`` / ``\"robust\"`` — defaults (0.5 / 5.0), gated poles,
      unless overridden.
    - ``profile=\"auto\"`` — mesh-conditioned proposal via
      :func:`pymcfs.params.propose_mcfs_params` (requires ``mesh``).
      ``branching`` is forwarded (default ``\"sparse\"``).
      ``gate_exterior_poles`` overrides when not ``None``.

    Returns
    -------
    attraction_weight, medial_weight, gate_exterior_poles

    Raises
    ------
    ValueError
        If ``profile`` is unknown, or ``profile=\"auto\"`` without ``mesh``.
    """
    if profile not in (None, "robust", "starlab", "auto"):
        raise ValueError(
            f"profile must be None, 'robust', 'starlab', or 'auto'; got {profile!r}"
        )

    if profile == "auto":
        if mesh is None:
            raise ValueError("profile='auto' requires a mesh to propose parameters")
        from .params import propose_mcfs_params

        proposed = propose_mcfs_params(mesh, branching=branching)  # type: ignore[arg-type]
        wh = float(attraction_weight)
        wm = float(medial_weight)
        # When caller left robust defaults, take the proposal; otherwise keep overrides.
        if float(attraction_weight) == _ROBUST_ATTRACTION and float(medial_weight) == _ROBUST_MEDIAL:
            wh = float(proposed.attraction_weight)
            wm = float(proposed.medial_weight)
        gate = (
            bool(gate_exterior_poles)
            if gate_exterior_poles is not None
            else bool(proposed.gate_exterior_poles)
        )
        return wh, wm, gate

    if profile == "starlab":
        default_wh, default_wm, default_gate = (
            _STARLAB_ATTRACTION,
            _STARLAB_MEDIAL,
            False,
        )
    else:
        default_wh, default_wm, default_gate = (
            _ROBUST_ATTRACTION,
            _ROBUST_MEDIAL,
            True,
        )

    if profile == "starlab" and float(attraction_weight) == _ROBUST_ATTRACTION:
        # Function signature default is robust; remap when requesting starlab.
        wh = float(default_wh)
    else:
        wh = float(attraction_weight)

    if profile == "starlab" and float(medial_weight) == _ROBUST_MEDIAL:
        wm = float(default_wm)
    else:
        wm = float(medial_weight)

    gate = default_gate if gate_exterior_poles is None else bool(gate_exterior_poles)
    return wh, wm, gate


def _merge_skeletonize_options(
    *,
    settings: SkeletonizeSettings | None,
    params: McfsParams | None,
    attraction_weight: float | None,
    medial_weight: float | None,
    gate_exterior_poles: bool | None,
    fast_gating: bool | None,
    use_cholmod: bool | None,
    profile: McfsProfile | None,
    branching: BranchingPreference | str | None,
    max_iterations: int | None,
    timeout_seconds: float | None,
    min_edge_length: float | None,
    max_triangle_angle: float | None,
    area_variation_factor: float | None,
    max_vertex_growth: float | None,
    pinned_attraction_floor: float | None,
    keep_largest_component: bool | None,
    resample: bool | ResampleMode | None,
    resample_spacing: float | None,
    resample_spacing_frac: float | None,
    prune_exterior: bool | None,
    prune_short_leaves: bool | None,
    short_leaf_scale: float | None,
    prune_thick_hubs: bool | None,
    keep_hub_branches: int | None,
    hub_degree_min: int | None,
    hub_radius_frac: float | None,
    extend_tips: bool | None,
    tip_extend_scale: float | None,
    tip_clearance_frac: float | None,
    tip_cone_deg: float | None,
    validate: bool | None,
    parameter_search: bool | None,
    max_search_contracts: int | None,
) -> tuple[ContractionSettings, RefineSettings, bool, bool, int]:
    """Merge SkeletonizeSettings / McfsParams / thin kwargs (kwargs win)."""
    s = settings if settings is not None else SkeletonizeSettings()
    c = s.contraction
    r = s.refine

    aw = c.attraction_weight
    mw = c.medial_weight
    gate = c.gate_exterior_poles
    br: BranchingPreference | str = c.branching
    prof = c.profile
    fg = c.fast_gating
    chol = c.use_cholmod
    max_it = c.max_iterations
    timeout = c.timeout_seconds
    min_el = c.min_edge_length
    max_ang = c.max_triangle_angle
    area_var = c.area_variation_factor
    max_vg = c.max_vertex_growth
    pin_floor = c.pinned_attraction_floor

    if params is not None:
        aw = params.attraction_weight
        mw = params.medial_weight
        gate = params.gate_exterior_poles
        br = params.branching

    def _pick(override, base):
        return base if override is None else override

    aw = _pick(attraction_weight, aw)
    mw = _pick(medial_weight, mw)
    gate = _pick(gate_exterior_poles, gate)
    fg = bool(_pick(fast_gating, fg))
    chol = _pick(use_cholmod, chol)
    prof = _pick(profile, prof)
    br = _pick(branching, br)
    max_it = int(_pick(max_iterations, max_it))
    timeout = _pick(timeout_seconds, timeout)
    min_el = _pick(min_edge_length, min_el)
    max_ang = float(_pick(max_triangle_angle, max_ang))
    area_var = float(_pick(area_variation_factor, area_var))
    max_vg = _pick(max_vertex_growth, max_vg)
    pin_floor = float(_pick(pinned_attraction_floor, pin_floor))

    contraction = ContractionSettings(
        attraction_weight=float(aw),
        medial_weight=float(mw),
        profile=prof,
        branching=br,  # type: ignore[arg-type]
        gate_exterior_poles=gate,
        fast_gating=fg,
        use_cholmod=chol,
        max_iterations=max_it,
        timeout_seconds=timeout,
        min_edge_length=min_el,
        max_triangle_angle=max_ang,
        area_variation_factor=area_var,
        max_vertex_growth=max_vg,
        pinned_attraction_floor=pin_floor,
    )

    refine = RefineSettings(
        prune_exterior=bool(_pick(prune_exterior, r.prune_exterior)),
        prune_short_leaves=bool(_pick(prune_short_leaves, r.prune_short_leaves)),
        short_leaf_scale=float(_pick(short_leaf_scale, r.short_leaf_scale)),
        prune_thick_hubs=bool(_pick(prune_thick_hubs, r.prune_thick_hubs)),
        keep_hub_branches=int(_pick(keep_hub_branches, r.keep_hub_branches)),
        hub_degree_min=int(_pick(hub_degree_min, r.hub_degree_min)),
        hub_radius_frac=float(_pick(hub_radius_frac, r.hub_radius_frac)),
        extend_tips=bool(_pick(extend_tips, r.extend_tips)),
        tip_extend_scale=float(_pick(tip_extend_scale, r.tip_extend_scale)),
        tip_clearance_frac=float(_pick(tip_clearance_frac, r.tip_clearance_frac)),
        tip_cone_deg=float(_pick(tip_cone_deg, r.tip_cone_deg)),
        keep_largest_component=bool(
            _pick(keep_largest_component, r.keep_largest_component)
        ),
        resample=_pick(resample, r.resample),  # type: ignore[arg-type]
        resample_spacing=_pick(resample_spacing, r.resample_spacing),
        resample_spacing_frac=_pick(resample_spacing_frac, r.resample_spacing_frac),
    )

    do_validate = bool(_pick(validate, s.validate))
    do_search = bool(_pick(parameter_search, s.parameter_search))
    max_contracts = int(_pick(max_search_contracts, s.max_search_contracts))
    return contraction, refine, do_validate, do_search, max_contracts


@dataclass
class Skeleton:
    """1D curve skeleton: nodes and edges describing the medial centerline.

    Produced by contracting the input surface to a thin meso-skeleton, then
    collapsing that surface to a curve graph (optionally refined).

    Attributes
    ----------
    nodes : (k, 3) float ndarray
        Node positions in 3D.
    edges : (e, 2) int ndarray
        Undirected edges as pairs of indices into ``nodes``.
    graph : networkx.Graph
        Same connectivity with node attribute ``pos`` and edge ``weight``.
    """

    nodes: np.ndarray
    edges: np.ndarray
    graph: nx.Graph

    @classmethod
    def from_graph(cls, G: nx.Graph) -> "Skeleton":
        """Build a densely indexed :class:`Skeleton` from a curve graph.

        Parameters
        ----------
        G :
            Undirected graph with node attribute ``pos`` (and optional edge
            ``weight``). Nodes are relabeled to ``0..n-1``.

        Returns
        -------
        Skeleton
        """
        H = G.copy()
        mapping = {n: i for i, n in enumerate(H.nodes)}
        if mapping:
            H = nx.relabel_nodes(H, mapping, copy=True)
        nodes_arr = (
            np.array([H.nodes[n]["pos"] for n in H.nodes], dtype=float)
            if H.number_of_nodes()
            else np.zeros((0, 3))
        )
        edges_arr = (
            np.array([[u, v] for u, v in H.edges], dtype=int)
            if H.number_of_edges()
            else np.zeros((0, 2), dtype=int)
        )
        return cls(nodes=nodes_arr, edges=edges_arr, graph=H)

    def to_polylines(self) -> list[np.ndarray]:
        """Split the skeleton into polylines at junctions (degree ≠ 2).

        Returns
        -------
        list of (k_i, 3) float arrays
            Each array is an ordered sequence of 3D points along one maximal path
            between junctions/leaves (polyline decomposition).
        """
        G = self.graph
        if G.number_of_nodes() == 0:
            return []
        for n in G.nodes:
            if "pos" not in G.nodes[n]:
                # fall back to nodes array if densely indexed
                try:
                    G.nodes[n]["pos"] = np.asarray(self.nodes[int(n)], dtype=float)
                except Exception:
                    pass

        deg = dict(G.degree())
        terminals = {n for n, d in deg.items() if d != 2}
        if not terminals:
            # Pure cycle: emit one closed loop starting at an arbitrary node
            start = next(iter(G.nodes))
            cycle = nx.find_cycle(G, source=start)
            pts = [np.asarray(G.nodes[u]["pos"], dtype=float) for u, v in cycle]
            pts.append(np.asarray(G.nodes[cycle[0][0]]["pos"], dtype=float))
            return [np.vstack(pts)]

        polylines: list[np.ndarray] = []
        visited: set[tuple[int, int]] = set()

        def edge_key(a: int, b: int) -> tuple[int, int]:
            return (a, b) if a < b else (b, a)

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
                pts = np.array([G.nodes[n]["pos"] for n in path], dtype=float)
                polylines.append(pts)
        return polylines

    def write_polylines(self, filepath: str) -> None:
        """Write polylines as text: ``N x y z x y z ...`` per line.

        Parameters
        ----------
        filepath :
            Output path (created/overwritten).
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for pl in self.to_polylines():
                coords = " ".join(f"{float(c):.9g}" for p in pl for c in p)
                f.write(f"{pl.shape[0]} {coords}\n")

    def write_cg(self, filepath: str) -> None:
        """Write the skeleton as a Starlab Curve Graph (``.cg``) file.

        Parameters
        ----------
        filepath :
            Output ``.cg`` path.
        """
        from .cg_io import write_cg

        write_cg(filepath, self.nodes, self.edges)

    def plot_3d(
        self,
        mesh: tm.Trimesh | tuple[np.ndarray, np.ndarray] | None = None,
        *,
        show_nodes: bool = False,
        node_size: float = 4.0,
        node_color: str = "#d62728",
        edge_color: str = "#1f77b4",
        edge_width: float = 4.0,
        mesh_color: str = "#AAAAAA",
        mesh_opacity: float = 0.3,
        title: str | None = None,
        autoshow: bool = True,
    ) -> object:
        """Interactive 3D visualization of the skeleton with optional mesh overlay.

        Parameters
        ----------
        mesh : trimesh.Trimesh or (V,F) or None
            If provided, overlays the original mesh using Plotly Mesh3d. You can pass
            either a trimesh.Trimesh instance or a tuple (vertices, faces).
        show_nodes : bool
            If True, also draw node markers in addition to skeleton edges.
        node_size : float
            Marker size for nodes when show_nodes=True.
        node_color : str
            Color for node markers.
        edge_color : str
            Color for skeleton edges.
        edge_width : float
            Line width for skeleton edges.
        mesh_color : str
            Color for mesh surface.
        mesh_opacity : float
            Opacity for mesh surface in [0,1].
        title : str or None
            Figure title.
        autoshow : bool
            If True, call fig.show() before returning (useful in notebooks).

        Returns
        -------
        plotly.graph_objects.Figure
            The created Plotly figure.

        Raises
        ------
        ImportError
            If Plotly is not installed (``uv sync --extra viz`` / ``pymcfs[viz]``).
        TypeError
            If ``mesh`` is neither a Trimesh nor a ``(V, F)`` tuple.
        """
        from .viz import require_plotly

        go = require_plotly()

        traces: list[go.BaseTraceType] = []

        # Optional mesh overlay
        if mesh is not None:
            if isinstance(mesh, tm.Trimesh):
                V = np.asarray(mesh.vertices, dtype=float)
                F = np.asarray(mesh.faces, dtype=int)
            else:
                try:
                    V, F = mesh  # type: ignore[misc]
                    V = np.asarray(V, dtype=float)
                    F = np.asarray(F, dtype=int)
                except Exception:
                    raise TypeError("mesh must be a trimesh.Trimesh or a (V,F) tuple")
            if V.size > 0 and F.size > 0:
                mesh_trace = go.Mesh3d(
                    x=V[:, 0], y=V[:, 1], z=V[:, 2],
                    i=F[:, 0], j=F[:, 1], k=F[:, 2],
                    color=mesh_color,
                    opacity=float(np.clip(mesh_opacity, 0.0, 1.0)),
                    name="mesh",
                    flatshading=True,
                    lighting=dict(ambient=0.6, diffuse=0.7, roughness=0.9),
                    showscale=False,
                )
                traces.append(mesh_trace)

        # Skeleton edges as line segments
        P = np.asarray(self.nodes, dtype=float)
        E = np.asarray(self.edges, dtype=int)
        if P.size > 0 and E.size > 0:
            xs: list[float | None] = []
            ys: list[float | None] = []
            zs: list[float | None] = []
            for (a, b) in E:
                pa = P[int(a)]
                pb = P[int(b)]
                xs.extend([float(pa[0]), float(pb[0]), None])
                ys.extend([float(pa[1]), float(pb[1]), None])
                zs.extend([float(pa[2]), float(pb[2]), None])
            edge_trace = go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=edge_color, width=float(edge_width)),
                name="skeleton",
            )
            traces.append(edge_trace)

        # Optional node markers
        if show_nodes and P.size > 0:
            node_trace = go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2],
                mode="markers",
                marker=dict(size=float(node_size), color=node_color),
                name="nodes",
            )
            traces.append(node_trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title or "Skeleton 3D",
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
                aspectmode="data",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        if autoshow:
            fig.show()
        return fig

    def resample(
        self,
        *,
        mode: ResampleMode = "uniform",
        spacing: float | None = None,
        spacing_frac: float | None = None,
    ) -> "Skeleton":
        """Return a copy with arc-length resampling or chain compression.

        Parameters
        ----------
        mode, spacing, spacing_frac :
            Forwarded to :func:`resample_skeleton`.

        Returns
        -------
        Skeleton
        """
        return resample_skeleton(
            self, mode=mode, spacing=spacing, spacing_frac=spacing_frac
        )


def _coerce_mesh(mesh: Union[tm.Trimesh, object]) -> tm.Trimesh:
    """Accept ``trimesh.Trimesh`` or ``MeshManager``; raise ``TypeError`` otherwise."""
    if isinstance(mesh, tm.Trimesh):
        return mesh
    try:
        from .mesh import MeshManager
    except Exception:
        MeshManager = None  # type: ignore
    if MeshManager is not None and isinstance(mesh, MeshManager):  # type: ignore
        return mesh.to_trimesh()  # type: ignore[return-value]
    raise TypeError("mesh must be a trimesh.Trimesh or MeshManager")


def skeletonize(
    mesh: Union[tm.Trimesh, object],
    *,
    settings: SkeletonizeSettings | None = None,
    params: McfsParams | None = None,
    attraction_weight: float | None = None,
    medial_weight: float | None = None,
    gate_exterior_poles: bool | None = None,
    fast_gating: bool | None = None,
    use_cholmod: bool | None = None,
    profile: McfsProfile | None = None,
    branching: BranchingPreference | str | None = None,
    max_iterations: int | None = None,
    timeout_seconds: float | None = None,
    min_edge_length: float | None = None,
    max_triangle_angle: float | None = None,
    area_variation_factor: float | None = None,
    max_vertex_growth: float | None = None,
    pinned_attraction_floor: float | None = None,
    keep_largest_component: bool | None = None,
    resample: bool | ResampleMode | None = None,
    resample_spacing: float | None = None,
    resample_spacing_frac: float | None = None,
    prune_exterior: bool | None = None,
    prune_short_leaves: bool | None = None,
    short_leaf_scale: float | None = None,
    prune_thick_hubs: bool | None = None,
    keep_hub_branches: int | None = None,
    hub_degree_min: int | None = None,
    hub_radius_frac: float | None = None,
    extend_tips: bool | None = None,
    tip_extend_scale: float | None = None,
    tip_clearance_frac: float | None = None,
    tip_cone_deg: float | None = None,
    validate: bool | None = None,
    parameter_search: bool | None = None,
    max_search_contracts: int | None = None,
    verbose: bool = False,
    log: Optional[logging.Logger] = None,
) -> Skeleton:
    """Extract a 1D curve skeleton from a closed triangle mesh.

    Contracts the surface with mean-curvature flow toward the medial axis
    (Voronoi poles — approximate centerline targets inside the volume), then
    converts the thin meso-skeleton surface into a curve graph and runs the
    refine phase (prune / optional tip extension / resample for curve density).

    Parameters
    ----------
    mesh :
        Input closed triangle mesh (``trimesh.Trimesh`` or ``MeshManager``).
    settings :
        Optional :class:`~pymcfs.config.SkeletonizeSettings` bundle.
    params :
        Optional :class:`~pymcfs.params.McfsParams` (overrides settings weights).
    attraction_weight, medial_weight :
        Contraction weights. Attraction keeps vertices from sliding too far;
        medial pulls toward interior Voronoi poles. Thin kwargs override
        ``settings`` / ``params``.
    gate_exterior_poles :
        If True, apply medial weight only for poles inside the mesh.
        Default True for robust profile; False for ``profile=\"starlab\"``.
    fast_gating :
        Use the mesh's Embree ray backend (``pymcfs[embree]``) for pole
        containment instead of the exact float64 traverser. Much faster, but
        single precision: only safe for meshes at unit-ish scale near the
        origin. See :class:`pymcfs.mcfs.MeanCurvatureFlowSkeletonization`.
    use_cholmod :
        If True, require scikit-sparse CHOLMOD. If False, force SuperLU.
        If None (default), use CHOLMOD when importable.
    profile :
        ``None`` / ``\"robust\"`` — gated poles, defaults ``0.5`` / ``5``.
        ``\"starlab\"`` — ungated poles, ``0.1`` / ``0.2`` (parity dumps).
        ``\"auto\"`` — mesh-conditioned proposal from :func:`pymcfs.params.propose_mcfs_params`.
    branching :
        Branching preference when ``profile=\"auto\"``: ``\"sparse\"``
        (default, fewest junctions), ``\"balanced\"``, or ``\"dense\"``.
    max_iterations, timeout_seconds :
        Contraction stop limits.
    min_edge_length, max_triangle_angle :
        Remesh thresholds during contraction.
    area_variation_factor :
        Relative area change for convergence (vs initial surface area).
    max_vertex_growth :
        Abort when vertex count exceeds ``max_vertex_growth * n0`` (default 4.0).
    pinned_attraction_floor :
        Numerical floor for pinned-vertex attraction and remesh short-edge epsilon.
    keep_largest_component :
        If True, keep only the largest connected component of the curve graph.
    resample, resample_spacing, resample_spacing_frac :
        Optional curve-density resampling in the refine phase (off by default).
        ``resample=True`` / ``\"uniform\"`` arc-length resamples chains;
        ``resample=\"compress\"`` keeps only junctions/leaves.
        Distinct from ``parameter_search`` (which re-contracts with nearby weights).
    prune_exterior :
        If True (default), remove dangling curve tips outside the input mesh
        after conversion (see :func:`pymcfs.refine.prune_exterior_branches`).
    prune_short_leaves :
        If True (default), mild micro-spur prune
        (see :func:`pymcfs.refine.prune_short_leaves`).
    short_leaf_scale :
        Multiplier for short-leaf pruning (default 1.0).
    prune_thick_hubs :
        If True (default), cull extra leaf arms at thick high-degree hubs
        (see :func:`pymcfs.refine.prune_thick_hubs`).
    keep_hub_branches, hub_degree_min, hub_radius_frac :
        Thick-hub prune controls.
    extend_tips :
        If True, grow unfinished leaf tips toward lobe ends (default False).
        Useful for open-ended shapes without end-caps; leave off for general use.
    tip_extend_scale :
        Max tip travel as a multiple of bbox diagonal when ``extend_tips``
        is True (default 1.0).
    tip_clearance_frac, tip_cone_deg :
        Tip-extension clearance and cone half-angle.
    validate :
        Run mesh validation before contraction.
    parameter_search :
        If True, try a small set of nearby weights and refine-phase settings
        via :func:`pymcfs.search.search_mcfs_params` and return the best
        skeleton (~4× contraction cost). Default False. Not the same as
        ``resample`` (curve post-processing only).
    max_search_contracts :
        Cap on contraction trials when ``parameter_search`` is True (default 4).
    verbose, log :
        Progress logging.

    Returns
    -------
    Skeleton
        Curve skeleton with ``nodes``, ``edges``, and ``graph``.

    Raises
    ------
    TypeError
        If ``mesh`` is not a Trimesh or MeshManager.
    ValueError
        If validation fails, ``profile`` is invalid, or ``profile=\"auto\"``
        lacks a usable mesh.
    ImportError
        If ``use_cholmod=True`` but scikit-sparse CHOLMOD is unavailable.
    """
    _log = log or logger
    m = _coerce_mesh(mesh)

    contraction, refine, do_validate, do_search, max_contracts = (
        _merge_skeletonize_options(
            settings=settings,
            params=params,
            attraction_weight=attraction_weight,
            medial_weight=medial_weight,
            gate_exterior_poles=gate_exterior_poles,
            fast_gating=fast_gating,
            use_cholmod=use_cholmod,
            profile=profile,
            branching=branching,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            min_edge_length=min_edge_length,
            max_triangle_angle=max_triangle_angle,
            area_variation_factor=area_variation_factor,
            max_vertex_growth=max_vertex_growth,
            pinned_attraction_floor=pinned_attraction_floor,
            keep_largest_component=keep_largest_component,
            resample=resample,
            resample_spacing=resample_spacing,
            resample_spacing_frac=resample_spacing_frac,
            prune_exterior=prune_exterior,
            prune_short_leaves=prune_short_leaves,
            short_leaf_scale=short_leaf_scale,
            prune_thick_hubs=prune_thick_hubs,
            keep_hub_branches=keep_hub_branches,
            hub_degree_min=hub_degree_min,
            hub_radius_frac=hub_radius_frac,
            extend_tips=extend_tips,
            tip_extend_scale=tip_extend_scale,
            tip_clearance_frac=tip_clearance_frac,
            tip_cone_deg=tip_cone_deg,
            validate=validate,
            parameter_search=parameter_search,
            max_search_contracts=max_search_contracts,
        )
    )

    if do_search:
        from .search import search_mcfs_params

        return search_mcfs_params(
            m,
            settings=SkeletonizeSettings(
                contraction=contraction,
                refine=refine,
                validate=do_validate,
                parameter_search=False,
                max_search_contracts=max_contracts,
            ),
            max_search_contracts=max_contracts,
            verbose=verbose,
            log=_log,
        ).skeleton

    wh, wm, gate = resolve_mcfs_profile(
        contraction.profile,
        attraction_weight=contraction.attraction_weight,
        medial_weight=contraction.medial_weight,
        gate_exterior_poles=contraction.gate_exterior_poles,
        mesh=m,
        branching=contraction.branching,
    )

    from .mcfs import MeanCurvatureFlowSkeletonization

    _log.info(
        "skeletonize: start (max_iters=%d, attraction=%.3g, medial=%.3g, "
        "gate_poles=%s, profile=%s)",
        contraction.max_iterations,
        wh,
        wm,
        gate,
        contraction.profile,
    )
    driver = MeanCurvatureFlowSkeletonization(
        m,
        attraction_weight=wh,
        medial_weight=wm,
        gate_exterior_poles=gate,
        fast_gating=bool(contraction.fast_gating),
        use_cholmod=contraction.use_cholmod,
        min_edge_length=contraction.min_edge_length,
        max_triangle_angle=float(contraction.max_triangle_angle),
        area_variation_factor=float(contraction.area_variation_factor),
        max_iterations=int(contraction.max_iterations),
        timeout_seconds=contraction.timeout_seconds,
        max_vertex_growth=contraction.max_vertex_growth,
        pinned_attraction_floor=float(contraction.pinned_attraction_floor),
        validate=bool(do_validate),
        verbose=verbose,
        log=_log,
    )
    driver.contract_until_convergence()
    skel = driver.convert_to_skeleton(
        resample=refine.resample,
        resample_spacing=refine.resample_spacing,
        resample_spacing_frac=refine.resample_spacing_frac,
        keep_largest_component=bool(refine.keep_largest_component),
        prune_exterior=bool(refine.prune_exterior),
        prune_short_leaves=bool(refine.prune_short_leaves),
        short_leaf_scale=float(refine.short_leaf_scale),
        prune_thick_hubs=bool(refine.prune_thick_hubs),
        keep_hub_branches=int(refine.keep_hub_branches),
        hub_degree_min=int(refine.hub_degree_min),
        hub_radius_frac=float(refine.hub_radius_frac),
        extend_tips=bool(refine.extend_tips),
        tip_extend_scale=float(refine.tip_extend_scale),
        tip_clearance_frac=float(refine.tip_clearance_frac),
        tip_cone_deg=float(refine.tip_cone_deg),
    )
    _log.info(
        "skeletonize: done (nodes=%d, edges=%d)",
        skel.nodes.shape[0],
        skel.edges.shape[0],
    )
    return skel


def contract_mesh(
    mesh: Union[tm.Trimesh, object],
    *,
    settings: ContractionSettings | SkeletonizeSettings | None = None,
    params: McfsParams | None = None,
    attraction_weight: float | None = None,
    medial_weight: float | None = None,
    gate_exterior_poles: bool | None = None,
    fast_gating: bool | None = None,
    use_cholmod: bool | None = None,
    profile: McfsProfile | None = None,
    branching: BranchingPreference | str | None = None,
    max_iterations: int | None = None,
    timeout_seconds: float | None = None,
    min_edge_length: float | None = None,
    max_triangle_angle: float | None = None,
    area_variation_factor: float | None = None,
    max_vertex_growth: float | None = None,
    pinned_attraction_floor: float | None = None,
    validate: bool | None = None,
    verbose: bool = False,
    log: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract a closed mesh toward its medial axis; return the meso-skeleton.

    The meso-skeleton is still a triangle surface — thinner and more
    centerline-like than the input — before conversion to a 1D curve.
    Same contraction controls as :func:`skeletonize` (no curve conversion or
    refine-phase flags).

    Parameters
    ----------
    mesh :
        Input closed triangle mesh (``trimesh.Trimesh`` or ``MeshManager``).
    settings, params, attraction_weight, medial_weight, ... :
        Same contraction meaning as :func:`skeletonize`.

    Returns
    -------
    V : (n, 3) float ndarray
        Contracted vertex positions.
    F : (m, 3) int ndarray
        Triangle indices of the meso-skeleton surface.

    Raises
    ------
    TypeError
        If ``mesh`` is not a Trimesh or MeshManager.
    ValueError
        If validation fails or ``profile`` is invalid.
    ImportError
        If ``use_cholmod=True`` but CHOLMOD is unavailable.
    """
    _log = log or logger
    m = _coerce_mesh(mesh)

    if isinstance(settings, SkeletonizeSettings):
        skel_settings = settings
    elif isinstance(settings, ContractionSettings):
        skel_settings = SkeletonizeSettings(contraction=settings)
    else:
        skel_settings = None

    contraction, _refine, do_validate, _search, _mc = _merge_skeletonize_options(
        settings=skel_settings,
        params=params,
        attraction_weight=attraction_weight,
        medial_weight=medial_weight,
        gate_exterior_poles=gate_exterior_poles,
        fast_gating=fast_gating,
        use_cholmod=use_cholmod,
        profile=profile,
        branching=branching,
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
        min_edge_length=min_edge_length,
        max_triangle_angle=max_triangle_angle,
        area_variation_factor=area_variation_factor,
        max_vertex_growth=max_vertex_growth,
        pinned_attraction_floor=pinned_attraction_floor,
        keep_largest_component=None,
        resample=None,
        resample_spacing=None,
        resample_spacing_frac=None,
        prune_exterior=None,
        prune_short_leaves=None,
        short_leaf_scale=None,
        prune_thick_hubs=None,
        keep_hub_branches=None,
        hub_degree_min=None,
        hub_radius_frac=None,
        extend_tips=None,
        tip_extend_scale=None,
        tip_clearance_frac=None,
        tip_cone_deg=None,
        validate=validate,
        parameter_search=False,
        max_search_contracts=None,
    )

    wh, wm, gate = resolve_mcfs_profile(
        contraction.profile,
        attraction_weight=contraction.attraction_weight,
        medial_weight=contraction.medial_weight,
        gate_exterior_poles=contraction.gate_exterior_poles,
        mesh=m,
        branching=contraction.branching,
    )

    from .mcfs import MeanCurvatureFlowSkeletonization

    _log.info(
        "contract_mesh: start (max_iters=%d, attraction=%.3g, medial=%.3g)",
        contraction.max_iterations,
        wh,
        wm,
    )
    driver = MeanCurvatureFlowSkeletonization(
        m,
        attraction_weight=wh,
        medial_weight=wm,
        gate_exterior_poles=gate,
        fast_gating=bool(contraction.fast_gating),
        use_cholmod=contraction.use_cholmod,
        min_edge_length=contraction.min_edge_length,
        max_triangle_angle=float(contraction.max_triangle_angle),
        area_variation_factor=float(contraction.area_variation_factor),
        max_iterations=int(contraction.max_iterations),
        timeout_seconds=contraction.timeout_seconds,
        max_vertex_growth=contraction.max_vertex_growth,
        pinned_attraction_floor=float(contraction.pinned_attraction_floor),
        validate=bool(do_validate),
        verbose=verbose,
        log=_log,
    )
    driver.contract_until_convergence()
    _log.info(
        "contract_mesh: done (n=%d, f=%d)",
        driver.V.shape[0],
        driver.F.shape[0],
    )
    return driver.V.copy(), driver.F.copy()


def curve_skeleton_from_mesh(
    V: np.ndarray,
    F: np.ndarray,
    *,
    mesh: tm.Trimesh | None = None,
    resample: bool | ResampleMode = False,
    resample_spacing: float | None = None,
    resample_spacing_frac: float | None = None,
    keep_largest_component: bool = False,
    prune_exterior: bool = True,
    prune_short_leaves: bool = True,
    short_leaf_scale: float = 1.0,
    prune_thick_hubs: bool = True,
    keep_hub_branches: int = 2,
    hub_degree_min: int = 4,
    hub_radius_frac: float = 0.015,
    extend_tips: bool = False,
    tip_extend_scale: float = 1.0,
    tip_clearance_frac: float = 0.01,
    tip_cone_deg: float = 40.0,
) -> Skeleton:
    """Convert a triangle mesh surface to a 1D curve-graph :class:`Skeleton`.

    Parameters
    ----------
    V : (n, 3) float
        Vertex positions (typically a meso-skeleton from :func:`contract_mesh`).
    F : (m, 3) int
        Triangle indices.
    mesh :
        Original closed surface used when pruning/extending. If None, mesh-based
        post-steps are skipped.
    resample, resample_spacing, resample_spacing_frac :
        Same meaning as in :func:`skeletonize`.
    keep_largest_component :
        If True, keep only the largest connected component.
    prune_exterior :
        If True (default) and ``mesh`` is provided, remove exterior dangling tips.
    prune_short_leaves :
        If True (default) and ``mesh`` is provided, mild micro-spur prune.
    short_leaf_scale :
        Multiplier for short_leaf pruning (default 1.0).
    prune_thick_hubs, keep_hub_branches, hub_degree_min, hub_radius_frac :
        Thick-hub principal-branch cull (default on).
    extend_tips, tip_extend_scale, tip_clearance_frac, tip_cone_deg :
        Tip extension (default off).

    Returns
    -------
    Skeleton
    """
    from .mcfs import meso_surface_to_curve_graph

    G = meso_surface_to_curve_graph(V, F)
    if keep_largest_component and G.number_of_nodes() > 0:
        comps = list(nx.connected_components(G))
        if len(comps) > 1:
            G = G.subgraph(max(comps, key=len)).copy()
    if prune_exterior and mesh is not None and G.number_of_nodes() > 0:
        G, _n = prune_exterior_graph(G, mesh)
    if prune_short_leaves and mesh is not None and G.number_of_nodes() > 0:
        G, _n = prune_short_leaves_graph(
            G, mesh, length_scale=float(short_leaf_scale)
        )
    if prune_thick_hubs and mesh is not None and G.number_of_nodes() > 0:
        G, _n = prune_thick_hubs_graph(
            G,
            mesh,
            keep_hub_branches=int(keep_hub_branches),
            hub_degree_min=int(hub_degree_min),
            hub_radius_frac=float(hub_radius_frac),
        )
    if extend_tips and mesh is not None and G.number_of_nodes() > 0:
        G, _n = extend_tips_graph(
            G,
            mesh,
            tip_extend_scale=float(tip_extend_scale),
            tip_clearance_frac=float(tip_clearance_frac),
            tip_cone_deg=float(tip_cone_deg),
        )
    mode, spacing, spacing_frac = resolve_resample_options(
        resample=resample,
        resample_spacing=resample_spacing,
        resample_spacing_frac=resample_spacing_frac,
    )
    if mode is not None:
        G = resample_skeleton_graph(
            G, mode=mode, spacing=spacing, spacing_frac=spacing_frac
        )
    return Skeleton.from_graph(G)


# Re-export refine-phase API for stable ``from pymcfs.skeleton import …``
__all__ = [
    "Skeleton",
    "skeletonize",
    "contract_mesh",
    "curve_skeleton_from_mesh",
    "resample_skeleton",
    "resample_skeleton_graph",
    "prune_exterior_branches",
    "prune_exterior_graph",
    "prune_short_leaves",
    "prune_short_leaves_graph",
    "prune_thick_hubs",
    "prune_thick_hubs_graph",
    "extend_tips",
    "extend_tips_graph",
    "resolve_resample_options",
    "resolve_mcfs_profile",
    "ResampleMode",
    "McfsProfile",
]
