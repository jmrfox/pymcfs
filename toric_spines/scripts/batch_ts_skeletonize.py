#!/usr/bin/env python3
"""Batch-skeletonize all ``toric_spines/data/mesh/TS*.obj`` with sparse oracle + validation.

Primary path: ``profile=\"auto\"`` / ``branching=\"sparse\"`` via the MCFS driver.
On hard fail or soft containment fail (``nodes_inside_frac < 0.85``), retry a
small sparse-favoring ``(attraction_weight, ratio)`` grid and keep the best
acceptable trial.

Usage:
  uv run python toric_spines/scripts/batch_ts_skeletonize.py
  uv run python toric_spines/scripts/batch_ts_skeletonize.py --mesh TS1 --mesh TS2
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import trimesh as tm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcfs import (
    MeanCurvatureFlowSkeletonization,
    SkeletonQualityReport,
    analyze_skeleton,
    load_and_repair,
    propose_mcfs_params,
    score_skeleton,
    search_mcfs_params,
)

logger = logging.getLogger(__name__)

MESH_DIR = ROOT / "toric_spines" / "data" / "mesh"
OUT_ROOT = ROOT / "outputs" / "polylines"

# Soft containment gate: allow ~15% exterior nodes.
MIN_NODES_INSIDE_FRAC = 0.85

# Sparse-favoring retry grid (medial_weight = ratio * attraction_weight).
RETRY_ATTRACTION = (0.25, 0.5, 1.0)
RETRY_RATIOS = (2.0, 3.0, 5.0)

SUMMARY_FIELDS = [
    "name",
    "status",
    "attraction_weight",
    "medial_weight",
    "ratio",
    "source",
    "iters",
    "elapsed_s",
    "growth",
    "aborted_remesh",
    "area_overshoot",
    "nonfinite",
    "nodes_inside_frac",
    "n_nodes_outside",
    "n_junctions",
    "n_leaves",
    "n_nodes",
    "n_components",
    "cyclomatic",
    "mesh_genus",
    "topo_delta",
    "score",
    "rejected",
    "reject_reason",
    "warn",
]


def _discover_meshes(filters: list[str] | None) -> list[Path]:
    paths = sorted(MESH_DIR.glob("TS*.obj"))
    if not filters:
        return paths
    wanted = {f.strip().removesuffix(".obj") for f in filters}
    selected = [p for p in paths if p.stem in wanted]
    missing = wanted - {p.stem for p in selected}
    if missing:
        raise FileNotFoundError(f"meshes not found under {MESH_DIR}: {sorted(missing)}")
    return selected


def _empty_report() -> SkeletonQualityReport:
    return SkeletonQualityReport(
        n_nodes=0,
        n_edges=0,
        n_components=0,
        n_junctions=0,
        n_leaves=0,
        nodes_inside_frac=None,
        n_nodes_outside=None,
    )


def _run_driver(
    mesh: tm.Trimesh,
    *,
    attraction_weight: float,
    medial_weight: float,
    gate_exterior_poles: bool,
    max_iterations: int,
    timeout_seconds: float | None,
    max_vertex_growth: float,
    prune_thick_hubs: bool = True,
    keep_hub_branches: int = 2,
    extend_tips: bool = True,
    tip_extend_scale: float = 1.0,
) -> dict:
    t0 = time.perf_counter()
    driver = MeanCurvatureFlowSkeletonization(
        mesh,
        attraction_weight=float(attraction_weight),
        medial_weight=float(medial_weight),
        gate_exterior_poles=bool(gate_exterior_poles),
        max_iterations=int(max_iterations),
        timeout_seconds=timeout_seconds,
        max_vertex_growth=float(max_vertex_growth),
        validate=False,  # already validated / repaired upstream
        verbose=False,
    )
    iters = driver.contract_until_convergence()
    growth = driver.remesh_growth_ratio()
    aborted = bool(driver.aborted_remesh_growth)
    overshoot = bool(driver.area_overshoot_seen)
    nonfinite = not bool(np.isfinite(driver.V).all()) if driver.V.size else False

    skel = None
    report: SkeletonQualityReport
    if not aborted and not nonfinite and driver.V.shape[0] > 0 and driver.F.shape[0] > 0:
        skel = driver.convert_to_skeleton(
            resample=False,
            prune_exterior=True,
            prune_short_leaves=True,
            prune_thick_hubs=bool(prune_thick_hubs),
            keep_hub_branches=int(keep_hub_branches),
            extend_tips=bool(extend_tips),
            tip_extend_scale=float(tip_extend_scale),
        )
        if skel.nodes.shape[0] == 0:
            report = _empty_report()
            score = score_skeleton(
                report,
                remesh_growth_rejected=False,
                remesh_growth_ratio=growth,
                nonfinite=False,
                area_overshoot=overshoot,
            )
        else:
            report = analyze_skeleton(mesh, skel)
            score = score_skeleton(
                report,
                remesh_growth_rejected=False,
                remesh_growth_ratio=growth,
                nonfinite=False,
                area_overshoot=overshoot,
            )
    else:
        report = _empty_report()
        score = score_skeleton(
            report,
            remesh_growth_rejected=aborted,
            remesh_growth_ratio=growth,
            nonfinite=nonfinite,
            area_overshoot=overshoot and not aborted,
        )

    return {
        "attraction_weight": float(attraction_weight),
        "medial_weight": float(medial_weight),
        "ratio": (
            float(medial_weight) / float(attraction_weight)
            if attraction_weight
            else float("nan")
        ),
        "iters": int(iters),
        "elapsed_s": time.perf_counter() - t0,
        "growth": float(growth),
        "aborted_remesh": int(aborted),
        "area_overshoot": int(overshoot),
        "nonfinite": int(nonfinite),
        "skel": skel,
        "report": report,
        "score": score,
        "nodes_inside_frac": score.nodes_inside_frac,
        "n_nodes_outside": score.n_nodes_outside,
        "n_junctions": score.n_junctions,
        "n_leaves": score.n_leaves,
        "n_nodes": score.n_nodes,
        "n_components": score.n_components,
        "cyclomatic": report.skeleton_cyclomatic,
        "mesh_genus": report.mesh_genus,
        "topo_delta": score.topology_delta,
        "score_value": float(score.value),
        "rejected": int(score.rejected),
        "reject_reason": score.reject_reason,
    }


def _hard_fail(row: dict) -> bool:
    if row["aborted_remesh"] or row["nonfinite"]:
        return True
    if row["skel"] is None:
        return True
    if int(row["n_nodes"]) <= 0:
        return True
    if row["rejected"]:
        return True
    return False


def _soft_fail(row: dict) -> bool:
    frac = row["nodes_inside_frac"]
    if frac is None:
        return True
    return float(frac) < MIN_NODES_INSIDE_FRAC


def _soft_warn(row: dict) -> str:
    bits: list[str] = []
    if row["topo_delta"] is not None and int(row["topo_delta"]) != 0:
        bits.append(f"topo_delta={row['topo_delta']}")
    if int(row["n_junctions"]) >= 40:
        bits.append(f"high_junctions={row['n_junctions']}")
    return ";".join(bits)


def _acceptable(row: dict) -> bool:
    return not _hard_fail(row) and not _soft_fail(row)


def _write_outputs(case_dir: Path, row: dict, *, name: str) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    skel = row["skel"]
    report = row["report"]
    score = row["score"]
    if skel is not None and skel.nodes.shape[0] > 0:
        skel.write_polylines(str(case_dir / "skeleton_sparse.polylines.txt"))
        skel.write_cg(str(case_dir / "skeleton_sparse.cg"))
    quality = case_dir / "quality.txt"
    quality.write_text(
        "\n".join(
            [
                f"name={name}",
                f"source={row.get('source', '')}",
                f"attraction_weight={row['attraction_weight']:g} "
                f"medial_weight={row['medial_weight']:g} ratio={row['ratio']:.4g}",
                f"iters={row['iters']} elapsed_s={row['elapsed_s']:.2f}",
                f"growth={row['growth']:.4g} aborted={row['aborted_remesh']} "
                f"overshoot={row['area_overshoot']} nonfinite={row['nonfinite']}",
                report.summary() if report is not None else "report=none",
                score.summary() if score is not None else "score=none",
                f"warn={row.get('warn', '')}",
                f"status={row.get('status', '')}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _retry_grid(
    mesh: tm.Trimesh,
    *,
    max_iterations: int,
    timeout_seconds: float | None,
    max_vertex_growth: float,
    prune_thick_hubs: bool,
    keep_hub_branches: int,
    extend_tips: bool,
    tip_extend_scale: float,
) -> list[dict]:
    trials: list[dict] = []
    for aw in RETRY_ATTRACTION:
        for ratio in RETRY_RATIOS:
            mw = float(aw) * float(ratio)
            logger.info(
                "  retry attraction_weight=%g medial_weight=%g (r=%g) …",
                aw,
                mw,
                ratio,
            )
            row = _run_driver(
                mesh,
                attraction_weight=float(aw),
                medial_weight=mw,
                gate_exterior_poles=True,
                max_iterations=max_iterations,
                timeout_seconds=timeout_seconds,
                max_vertex_growth=max_vertex_growth,
                prune_thick_hubs=prune_thick_hubs,
                keep_hub_branches=keep_hub_branches,
                extend_tips=extend_tips,
                tip_extend_scale=tip_extend_scale,
            )
            row["source"] = f"retry_aw{aw:g}_r{ratio:g}"
            logger.info(
                "    score=%.4g inside=%s outside=%s junctions=%s (%.1fs)",
                row["score_value"],
                row["nodes_inside_frac"],
                row["n_nodes_outside"],
                row["n_junctions"],
                row["elapsed_s"],
            )
            trials.append(row)
    return trials


def _pick_best(trials: list[dict]) -> dict | None:
    if not trials:
        return None
    acceptable = [t for t in trials if _acceptable(t)]
    pool = acceptable if acceptable else trials
    return max(pool, key=lambda t: float(t["score_value"]))


def _run_parameter_search(
    mesh: tm.Trimesh,
    *,
    max_iterations: int,
    timeout_seconds: float | None,
    prune_thick_hubs: bool,
    keep_hub_branches: int,
    extend_tips: bool,
    tip_extend_scale: float,
) -> dict:
    t0 = time.perf_counter()
    result = search_mcfs_params(
        mesh,
        profile="auto",
        branching="sparse",
        max_iterations=int(max_iterations),
        timeout_seconds=timeout_seconds,
        validate=False,
        prune_thick_hubs=bool(prune_thick_hubs),
        keep_hub_branches=int(keep_hub_branches),
        extend_tips=bool(extend_tips),
        tip_extend_scale=float(tip_extend_scale),
        return_trials=True,
        verbose=False,
    )
    skel = result.skeleton
    report = analyze_skeleton(mesh, skel) if skel.nodes.shape[0] else _empty_report()
    score = result.score
    return {
        "attraction_weight": float(result.attraction_weight),
        "medial_weight": float(result.medial_weight),
        "ratio": (
            float(result.medial_weight) / float(result.attraction_weight)
            if result.attraction_weight
            else float("nan")
        ),
        "iters": -1,
        "elapsed_s": time.perf_counter() - t0,
        "growth": float("nan"),
        "aborted_remesh": 0,
        "area_overshoot": 0,
        "nonfinite": 0,
        "skel": skel,
        "report": report,
        "score": score,
        "nodes_inside_frac": score.nodes_inside_frac,
        "n_nodes_outside": score.n_nodes_outside,
        "n_junctions": score.n_junctions,
        "n_leaves": score.n_leaves,
        "n_nodes": score.n_nodes,
        "n_components": score.n_components,
        "cyclomatic": report.skeleton_cyclomatic,
        "mesh_genus": report.mesh_genus,
        "topo_delta": score.topology_delta,
        "score_value": float(score.value),
        "rejected": int(score.rejected),
        "reject_reason": score.reject_reason,
        "source": "parameter_search",
        "n_contracts": result.n_contracts,
        "n_converts": result.n_converts,
    }


def process_mesh(
    path: Path,
    *,
    max_iterations: int,
    timeout_seconds: float | None,
    max_vertex_growth: float,
    out_root: Path,
    prune_thick_hubs: bool,
    keep_hub_branches: int,
    extend_tips: bool,
    tip_extend_scale: float,
    parameter_search: bool = False,
) -> dict:
    name = path.stem
    logger.info("=== %s ===", name)
    mesh = load_and_repair(str(path))
    proposed = propose_mcfs_params(mesh, branching="sparse")
    logger.info("  oracle: %s", proposed.summary())

    if parameter_search:
        logger.info("  parameter_search …")
        chosen = _run_parameter_search(
            mesh,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            prune_thick_hubs=prune_thick_hubs,
            keep_hub_branches=keep_hub_branches,
            extend_tips=extend_tips,
            tip_extend_scale=tip_extend_scale,
        )
        logger.info(
            "  search: score=%.4g attraction_weight=%g medial_weight=%g "
            "contracts=%s converts=%s (%.1fs)",
            chosen["score_value"],
            chosen["attraction_weight"],
            chosen["medial_weight"],
            chosen.get("n_contracts"),
            chosen.get("n_converts"),
            chosen["elapsed_s"],
        )
        if _acceptable(chosen):
            status = "ok"
        elif chosen["skel"] is not None and int(chosen["n_nodes"]) > 0:
            status = "needs_review"
        else:
            status = "failed"
    else:
        primary = _run_driver(
            mesh,
            attraction_weight=float(proposed.attraction_weight),
            medial_weight=float(proposed.medial_weight),
            gate_exterior_poles=bool(proposed.gate_exterior_poles),
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            max_vertex_growth=max_vertex_growth,
            prune_thick_hubs=prune_thick_hubs,
            keep_hub_branches=keep_hub_branches,
            extend_tips=extend_tips,
            tip_extend_scale=tip_extend_scale,
        )
        primary["source"] = "oracle_sparse"
        logger.info(
            "  primary: score=%.4g inside=%s outside=%s junctions=%s "
            "growth=%.2f (%.1fs)",
            primary["score_value"],
            primary["nodes_inside_frac"],
            primary["n_nodes_outside"],
            primary["n_junctions"],
            primary["growth"],
            primary["elapsed_s"],
        )

        if _acceptable(primary):
            chosen = primary
            status = "ok"
        else:
            reason = "hard" if _hard_fail(primary) else "soft_containment"
            logger.info("  primary failed (%s); retrying sparse grid …", reason)
            trials = [primary] + _retry_grid(
                mesh,
                max_iterations=max_iterations,
                timeout_seconds=timeout_seconds,
                max_vertex_growth=max_vertex_growth,
                prune_thick_hubs=prune_thick_hubs,
                keep_hub_branches=keep_hub_branches,
                extend_tips=extend_tips,
                tip_extend_scale=tip_extend_scale,
            )
            chosen = _pick_best(trials)
            assert chosen is not None
            if _acceptable(chosen) and chosen is not primary:
                status = "retried_ok"
            elif _acceptable(chosen):
                status = "ok"
            elif chosen["skel"] is not None and int(chosen["n_nodes"]) > 0:
                status = "needs_review"
            else:
                status = "failed"

    chosen = dict(chosen)
    chosen["name"] = name
    chosen["status"] = status
    chosen["warn"] = _soft_warn(chosen)
    if chosen["warn"]:
        logger.info("  warn: %s", chosen["warn"])

    case_dir = out_root / name
    _write_outputs(case_dir, chosen, name=name)
    logger.info(
        "  -> %s attraction_weight=%g medial_weight=%g score=%.4g wrote %s",
        status,
        chosen["attraction_weight"],
        chosen["medial_weight"],
        chosen["score_value"],
        case_dir,
    )
    return chosen


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in SUMMARY_FIELDS}
            if "score_value" in r and out.get("score") is None:
                out["score"] = r["score_value"]
            w.writerow(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mesh",
        action="append",
        default=None,
        help="Mesh stem (e.g. TS1) or repeat; default = all TS*.obj",
    )
    ap.add_argument("--max-iterations", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-vertex-growth", type=float, default=4.0)
    ap.add_argument(
        "--no-thick-hubs",
        action="store_true",
        help="Disable thick-hub principal-branch prune",
    )
    ap.add_argument("--keep-hub-branches", type=int, default=2)
    ap.add_argument(
        "--extend-tips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extend unfinished tips (default: on for this batch script)",
    )
    ap.add_argument(
        "--tip-extend-scale",
        type=float,
        default=1.0,
        help="Max tip travel as multiple of bbox diagonal (default 1.0)",
    )
    ap.add_argument(
        "--parameter-search",
        action="store_true",
        help="Use pymcfs.search_mcfs_params (~4 contractions) instead of "
        "oracle + sparse retry (default off)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_ROOT,
        help=f"Output root (default {OUT_ROOT})",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    meshes = _discover_meshes(args.mesh)
    if not meshes:
        logger.error("No TS*.obj under %s", MESH_DIR)
        return 1

    out_root = args.out if args.out.is_absolute() else (ROOT / args.out)
    timeout = None if args.timeout <= 0 else float(args.timeout)

    logger.info("meshes (%d): %s", len(meshes), [p.stem for p in meshes])
    logger.info(
        "post: thick_hubs=%s keep=%d extend_tips=%s tip_extend_scale=%s "
        "parameter_search=%s",
        not args.no_thick_hubs,
        args.keep_hub_branches,
        args.extend_tips,
        args.tip_extend_scale,
        args.parameter_search,
    )
    rows: list[dict] = []
    for path in meshes:
        row = process_mesh(
            path,
            max_iterations=int(args.max_iterations),
            timeout_seconds=timeout,
            max_vertex_growth=float(args.max_vertex_growth),
            out_root=out_root,
            prune_thick_hubs=not bool(args.no_thick_hubs),
            keep_hub_branches=int(args.keep_hub_branches),
            extend_tips=bool(args.extend_tips),
            tip_extend_scale=float(args.tip_extend_scale),
            parameter_search=bool(args.parameter_search),
        )
        # Flatten score for CSV
        row["score"] = row.get("score_value")
        rows.append(row)

    summary_path = out_root / "batch_summary.csv"
    _write_summary(summary_path, rows)
    logger.info("\nWrote %s", summary_path)
    for r in rows:
        logger.info(
            "  %-6s %-12s inside=%s outside=%s junc=%s score=%.4g",
            r["name"],
            r["status"],
            r["nodes_inside_frac"],
            r["n_nodes_outside"],
            r["n_junctions"],
            r["score"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
