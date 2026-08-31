"""Try a few nearby contraction weights and refine settings; keep the best.

Distinct from curve ``resample`` (node density only). Ranks candidates with
:func:`pymcfs.quality.score_skeleton` plus light tip/hub penalties.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import trimesh as tm

from .config import SkeletonizeSettings
from .mcfs import MeanCurvatureFlowSkeletonization
from .quality import SkeletonScore, analyze_skeleton, score_skeleton
from .refine import _leaf_arm_from_hub, _mesh_bbox_diag
from .skeleton import McfsProfile, Skeleton, _merge_skeletonize_options, resolve_mcfs_profile

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_ATTRACTION_MIN, _ATTRACTION_MAX = 0.25, 2.0
_RATIO_MIN, _RATIO_MAX = 1.5, 20.0


@dataclass
class McfsSearchResult:
    """Best trial from :func:`search_mcfs_params`.

    Attributes
    ----------
    skeleton :
        Winning curve skeleton.
    attraction_weight, medial_weight :
        Contraction weights used for the winning trial.
    gate_exterior_poles :
        Pole gating flag used during contraction.
    keep_hub_branches, tip_extend_scale :
        Refine-phase settings for the winning convert step.
    score :
        Ranking score (higher better).
    n_contracts, n_converts :
        How many contraction / convert evaluations ran.
    trials :
        Per-trial summaries when ``return_trials`` was True; else empty.
    """

    skeleton: Skeleton
    attraction_weight: float
    medial_weight: float
    gate_exterior_poles: bool
    keep_hub_branches: int
    tip_extend_scale: float
    score: SkeletonScore
    n_contracts: int = 0
    n_converts: int = 0
    trials: list[dict[str, Any]] = field(default_factory=list)


def _clamp_weights(attraction_weight: float, medial_weight: float) -> tuple[float, float]:
    wh = float(np.clip(attraction_weight, _ATTRACTION_MIN, _ATTRACTION_MAX))
    ratio = float(medial_weight) / max(wh, 1e-12)
    ratio = float(np.clip(ratio, _RATIO_MIN, _RATIO_MAX))
    return wh, wh * ratio


def _weight_candidates(
    attraction0: float, medial0: float, *, max_contracts: int
) -> list[tuple[float, float]]:
    """Build a small unique list of ``(attraction, medial)`` around the base."""
    raw = [
        (attraction0, medial0),
        (attraction0, 0.5 * medial0),
        (attraction0, min(1.5 * medial0, 20.0 * attraction0)),
        (0.75 * attraction0, 0.75 * medial0),
    ]
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for wh, wm in raw:
        cwh, cwm = _clamp_weights(wh, wm)
        key = (round(cwh, 6), round(cwm, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((cwh, cwm))
        if len(out) >= max(1, int(max_contracts)):
            break
    return out


def _refine_variants(
    *,
    prune_thick_hubs: bool,
    keep_hub_branches: int,
    extend_tips: bool,
    tip_extend_scale: float,
) -> list[tuple[int, float]]:
    """Return ``(keep_hub_branches, tip_extend_scale)`` variants."""
    keeps = (
        sorted({1, 2, 3, int(keep_hub_branches)})
        if prune_thick_hubs
        else [int(keep_hub_branches)]
    )
    scales = (
        sorted({0.5, 1.0, float(tip_extend_scale)})
        if extend_tips
        else [float(tip_extend_scale)]
    )
    # Prefer caller's defaults first, then other combos; cap converts.
    preferred = (int(keep_hub_branches), float(tip_extend_scale))
    combos = [preferred]
    for k in keeps:
        for s in scales:
            pair = (k, s)
            if pair not in combos:
                combos.append(pair)
    return combos[:6]


def score_skeleton_candidate(
    report,
    mesh: tm.Trimesh,
    skeleton: Skeleton,
    *,
    remesh_growth_rejected: bool = False,
    remesh_growth_ratio: float | None = None,
    nonfinite: bool = False,
    area_overshoot: bool = False,
    tip_clearance_frac: float = 0.01,
    keep_hub_branches: int = 2,
    hub_degree_min: int = 4,
    hub_radius_frac: float = 0.015,
) -> SkeletonScore:
    """Rank a skeleton candidate for parameter search (base score + tip/hub penalties)."""
    base = score_skeleton(
        report,
        remesh_growth_rejected=remesh_growth_rejected,
        remesh_growth_ratio=remesh_growth_ratio,
        nonfinite=nonfinite,
        area_overshoot=area_overshoot,
    )
    if base.rejected or skeleton.nodes.shape[0] == 0:
        return base

    diag = _mesh_bbox_diag(mesh)
    prox = tm.proximity.ProximityQuery(mesh)
    G = skeleton.graph
    leaves = [n for n in G.nodes if G.degree(n) == 1]
    tip_pen = 0.0
    if leaves:
        excesses = []
        for n in leaves:
            p = np.asarray(G.nodes[n]["pos"], dtype=float).reshape(1, 3)
            r = float(np.abs(prox.signed_distance(p)[0]))
            excesses.append(max(0.0, r / diag - float(tip_clearance_frac)))
        tip_pen = float(np.mean(excesses))

    hub_pen = 0.0
    r_min = float(hub_radius_frac) * diag
    keep = int(keep_hub_branches)
    for n in G.nodes:
        if G.degree(n) < int(hub_degree_min):
            continue
        p = np.asarray(G.nodes[n]["pos"], dtype=float).reshape(1, 3)
        r = float(np.abs(prox.signed_distance(p)[0]))
        if r < r_min:
            continue
        n_leaf = 0
        for nbr in G.neighbors(n):
            if _leaf_arm_from_hub(G, n, nbr) is not None:
                n_leaf += 1
        hub_pen += float(max(0, n_leaf - keep))

    value = float(base.value) - 1.0e3 * tip_pen - 50.0 * hub_pen
    return SkeletonScore(
        value=value,
        rejected=base.rejected,
        reject_reason=base.reject_reason,
        topology_delta=base.topology_delta,
        nodes_inside_frac=base.nodes_inside_frac,
        edges_inside_frac=base.edges_inside_frac,
        n_nodes_outside=base.n_nodes_outside,
        n_junctions=base.n_junctions,
        n_nodes=base.n_nodes,
        n_leaves=base.n_leaves,
        n_components=base.n_components,
    )


def search_mcfs_params(
    mesh: Union[tm.Trimesh, object],
    *,
    settings: SkeletonizeSettings | None = None,
    attraction_weight: float | None = None,
    medial_weight: float | None = None,
    gate_exterior_poles: bool | None = None,
    fast_gating: bool | None = None,
    use_cholmod: bool | None = None,
    profile: McfsProfile | None = None,
    branching: str | None = None,
    max_iterations: int | None = None,
    timeout_seconds: float | None = None,
    min_edge_length: float | None = None,
    max_triangle_angle: float | None = None,
    area_variation_factor: float | None = None,
    max_vertex_growth: float | None = None,
    pinned_attraction_floor: float | None = None,
    keep_largest_component: bool | None = None,
    resample: bool | str | None = None,
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
    max_search_contracts: int | None = None,
    return_trials: bool = False,
    verbose: bool = False,
    log: Optional[logging.Logger] = None,
) -> McfsSearchResult:
    """Try nearby contraction weights and refine settings; return the best skeleton.

    Re-contracts with a small set of attraction / medial weights around the
    resolved profile, then tries a few refine-phase variants. Ranking uses
    :func:`score_skeleton_candidate`. Not the same as curve ``resample``.

    Parameters
    ----------
    mesh :
        Input closed triangle mesh.
    settings :
        Optional :class:`~pymcfs.config.SkeletonizeSettings` base.
    profile, branching, attraction_weight, medial_weight, ... :
        Same meaning as :func:`pymcfs.skeleton.skeletonize`. Base weights come
        from the resolved profile; search explores neighbors.
    max_search_contracts :
        Maximum number of contraction runs (default 4).
    return_trials :
        If True, populate :attr:`McfsSearchResult.trials`.

    Returns
    -------
    McfsSearchResult
        Winning skeleton, weights, refine settings, and score.
    """
    from .skeleton import _coerce_mesh

    _log = log or logger
    m = _coerce_mesh(mesh)
    contraction, refine, do_validate, _search, max_contracts = (
        _merge_skeletonize_options(
            settings=settings,
            params=None,
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
            parameter_search=False,
            max_search_contracts=max_search_contracts,
        )
    )

    wh0, wm0, gate = resolve_mcfs_profile(
        contraction.profile,
        attraction_weight=contraction.attraction_weight,
        medial_weight=contraction.medial_weight,
        gate_exterior_poles=contraction.gate_exterior_poles,
        mesh=m,
        branching=contraction.branching,
    )
    # Search always gates poles for production-like ranking.
    gate = True if contraction.gate_exterior_poles is None else bool(gate)

    weights = _weight_candidates(wh0, wm0, max_contracts=max_contracts)
    refine_variants = _refine_variants(
        prune_thick_hubs=bool(refine.prune_thick_hubs),
        keep_hub_branches=int(refine.keep_hub_branches),
        extend_tips=bool(refine.extend_tips),
        tip_extend_scale=float(refine.tip_extend_scale),
    )

    _log.info(
        "parameter_search: %d weight candidates, %d refine variants "
        "(base attraction=%.3g medial=%.3g)",
        len(weights),
        len(refine_variants),
        wh0,
        wm0,
    )

    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    n_contracts = 0
    n_converts = 0

    for wh, wm in weights:
        n_contracts += 1
        _log.info("parameter_search: contract attraction=%.3g medial=%.3g", wh, wm)
        driver = MeanCurvatureFlowSkeletonization(
            m,
            attraction_weight=float(wh),
            medial_weight=float(wm),
            gate_exterior_poles=bool(gate),
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
            verbose=False,
            log=_log,
        )
        driver.contract_until_convergence()
        growth = driver.remesh_growth_ratio()
        aborted = bool(driver.aborted_remesh_growth)
        overshoot = bool(driver.area_overshoot_seen)
        nonfinite = (
            not bool(np.isfinite(driver.V).all()) if driver.V.size else False
        )

        if aborted or nonfinite or driver.V.shape[0] == 0 or driver.F.shape[0] == 0:
            from .quality import SkeletonQualityReport

            report = SkeletonQualityReport(
                n_nodes=0,
                n_edges=0,
                n_components=0,
                n_junctions=0,
                n_leaves=0,
                nodes_inside_frac=None,
                n_nodes_outside=None,
            )
            score = score_skeleton(
                report,
                remesh_growth_rejected=aborted,
                remesh_growth_ratio=growth,
                nonfinite=nonfinite,
                area_overshoot=overshoot and not aborted,
            )
            row = {
                "attraction_weight": wh,
                "medial_weight": wm,
                "keep_hub_branches": int(refine.keep_hub_branches),
                "tip_extend_scale": float(refine.tip_extend_scale),
                "score": score,
                "skeleton": None,
                "rejected": True,
            }
            trials.append(row)
            if best is None or score.value > best["score"].value:
                best = row
            continue

        for keep_k, tip_s in refine_variants:
            n_converts += 1
            skel = driver.convert_to_skeleton(
                resample=refine.resample,
                resample_spacing=refine.resample_spacing,
                resample_spacing_frac=refine.resample_spacing_frac,
                keep_largest_component=bool(refine.keep_largest_component),
                prune_exterior=bool(refine.prune_exterior),
                prune_short_leaves=bool(refine.prune_short_leaves),
                short_leaf_scale=float(refine.short_leaf_scale),
                prune_thick_hubs=bool(refine.prune_thick_hubs),
                keep_hub_branches=int(keep_k),
                hub_degree_min=int(refine.hub_degree_min),
                hub_radius_frac=float(refine.hub_radius_frac),
                extend_tips=bool(refine.extend_tips),
                tip_extend_scale=float(tip_s),
                tip_clearance_frac=float(refine.tip_clearance_frac),
                tip_cone_deg=float(refine.tip_cone_deg),
            )
            if skel.nodes.shape[0] == 0:
                from .quality import SkeletonQualityReport

                report = SkeletonQualityReport(
                    n_nodes=0,
                    n_edges=0,
                    n_components=0,
                    n_junctions=0,
                    n_leaves=0,
                    nodes_inside_frac=None,
                    n_nodes_outside=None,
                )
            else:
                report = analyze_skeleton(m, skel)
            score = score_skeleton_candidate(
                report,
                m,
                skel,
                remesh_growth_rejected=False,
                remesh_growth_ratio=growth,
                nonfinite=False,
                area_overshoot=overshoot,
                tip_clearance_frac=float(refine.tip_clearance_frac),
                keep_hub_branches=int(keep_k),
                hub_degree_min=int(refine.hub_degree_min),
                hub_radius_frac=float(refine.hub_radius_frac),
            )
            row = {
                "attraction_weight": wh,
                "medial_weight": wm,
                "keep_hub_branches": int(keep_k),
                "tip_extend_scale": float(tip_s),
                "score": score,
                "skeleton": skel,
                "rejected": bool(score.rejected),
            }
            trials.append(row)
            if verbose:
                _log.info(
                    "parameter_search: attraction=%.3g medial=%.3g keep=%d tip_s=%.3g "
                    "score=%.4g rejected=%s",
                    wh,
                    wm,
                    keep_k,
                    tip_s,
                    score.value,
                    score.rejected,
                )
            if best is None or score.value > best["score"].value:
                best = row

    if best is None or best.get("skeleton") is None:
        # Fall back: single skeletonize-equivalent with base weights.
        driver = MeanCurvatureFlowSkeletonization(
            m,
            attraction_weight=float(wh0),
            medial_weight=float(wm0),
            gate_exterior_poles=bool(gate),
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
            verbose=False,
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
        report = (
            analyze_skeleton(m, skel)
            if skel.nodes.shape[0]
            else None
        )
        if report is None:
            from .quality import SkeletonQualityReport

            report = SkeletonQualityReport(
                n_nodes=0,
                n_edges=0,
                n_components=0,
                n_junctions=0,
                n_leaves=0,
                nodes_inside_frac=None,
                n_nodes_outside=None,
            )
        score = score_skeleton_candidate(
            report,
            m,
            skel,
            tip_clearance_frac=float(refine.tip_clearance_frac),
            keep_hub_branches=int(refine.keep_hub_branches),
            hub_degree_min=int(refine.hub_degree_min),
            hub_radius_frac=float(refine.hub_radius_frac),
        )
        best = {
            "attraction_weight": wh0,
            "medial_weight": wm0,
            "keep_hub_branches": int(refine.keep_hub_branches),
            "tip_extend_scale": float(refine.tip_extend_scale),
            "score": score,
            "skeleton": skel,
            "rejected": bool(score.rejected),
        }
        n_contracts += 1
        n_converts += 1

    assert best is not None and best["skeleton"] is not None
    _log.info(
        "parameter_search: done best attraction=%.3g medial=%.3g score=%.4g "
        "(contracts=%d converts=%d)",
        float(best["attraction_weight"]),
        float(best["medial_weight"]),
        best["score"].value,
        n_contracts,
        n_converts,
    )
    return McfsSearchResult(
        skeleton=best["skeleton"],
        attraction_weight=float(best["attraction_weight"]),
        medial_weight=float(best["medial_weight"]),
        gate_exterior_poles=bool(gate),
        keep_hub_branches=int(best["keep_hub_branches"]),
        tip_extend_scale=float(best["tip_extend_scale"]),
        score=best["score"],
        n_contracts=n_contracts,
        n_converts=n_converts,
        trials=trials if (return_trials or verbose) else [],
    )
