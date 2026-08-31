#!/usr/bin/env python3
"""Grid-search MCFS (attraction_weight, medial_weight) and score skeletons.

Sweeps in (scale, ratio) space with ``medial_weight = ratio * attraction_weight``
to avoid the unstable low-attraction / high-medial corner. Ranks trials with
:func:`pymcfs.quality.score_skeleton` (topology match, containment, compactness)
and early-aborts remesh blow-ups.

Usage:
  uv run python toric_spines/scripts/sweep_mcfs_params.py --mesh toric_spines/data/mesh/TS2.obj
  uv run python toric_spines/scripts/sweep_mcfs_params.py --mesh ts2 --mesh ts1 --top-k 3
  uv run python toric_spines/scripts/sweep_mcfs_params.py --mesh ts2 --attraction-weight 0.5,1.0 --ratios 5,10
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
    analyze_skeleton,
    mesh_mcfs_features,
    propose_mcfs_params,
    score_skeleton,
)

logger = logging.getLogger(__name__)

MESH_PRESETS = {
    "ts1": ROOT / "toric_spines" / "data" / "mesh" / "TS1.obj",
    "ts2": ROOT / "toric_spines" / "data" / "mesh" / "TS2.obj",
    "ts76": ROOT / "toric_spines" / "data" / "mesh" / "TS76.obj",
}

DEFAULT_ATTRACTION = (0.25, 0.5, 1.0, 2.0)
DEFAULT_RATIOS = (2.0, 5.0, 10.0, 20.0)


def _resolve_mesh(spec: str) -> Path:
    key = spec.strip().lower()
    if key in MESH_PRESETS:
        return MESH_PRESETS[key]
    p = Path(spec)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _load_mesh(path: Path) -> tm.Trimesh:
    mesh = tm.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, tm.Trimesh):
        raise TypeError(f"expected Trimesh from {path}")
    return mesh


def _run_trial(
    mesh: tm.Trimesh,
    *,
    attraction_weight: float,
    medial_weight: float,
    max_iterations: int,
    timeout_seconds: float | None,
    max_vertex_growth: float,
) -> dict:
    t0 = time.perf_counter()
    driver = MeanCurvatureFlowSkeletonization(
        mesh,
        attraction_weight=float(attraction_weight),
        medial_weight=float(medial_weight),
        gate_exterior_poles=True,
        max_iterations=int(max_iterations),
        timeout_seconds=timeout_seconds,
        max_vertex_growth=float(max_vertex_growth),
        validate=True,
        verbose=False,
    )
    n0 = int(driver._n0)
    iters = driver.contract_until_convergence()
    growth = driver.remesh_growth_ratio()
    aborted = bool(driver.aborted_remesh_growth)
    overshoot = bool(driver.area_overshoot_seen)
    nonfinite = not bool(np.isfinite(driver.V).all()) if driver.V.size else False

    skel = None
    report = None
    score = None
    if not aborted and not nonfinite and driver.V.shape[0] > 0 and driver.F.shape[0] > 0:
        skel = driver.convert_to_skeleton(resample=False)
        report = analyze_skeleton(mesh, skel)
        score = score_skeleton(
            report,
            remesh_growth_rejected=False,
            remesh_growth_ratio=growth,
            nonfinite=False,
            area_overshoot=overshoot,
        )
    else:
        # Minimal empty report for reject scoring.
        from pymcfs import SkeletonQualityReport

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

    elapsed = time.perf_counter() - t0
    row = {
        "attraction_weight": attraction_weight,
        "medial_weight": medial_weight,
        "ratio": medial_weight / attraction_weight if attraction_weight else float("nan"),
        "iters": iters,
        "n0": n0,
        "n_final": int(driver.V.shape[0]),
        "growth": growth,
        "aborted_remesh": int(aborted),
        "area_overshoot": int(overshoot),
        "nonfinite": int(nonfinite),
        "score": score.value if score is not None else float("nan"),
        "rejected": int(score.rejected) if score is not None else 1,
        "reject_reason": score.reject_reason if score is not None else None,
        "topo_delta": score.topology_delta if score is not None else None,
        "nodes_inside_frac": (
            score.nodes_inside_frac if score is not None else None
        ),
        "n_nodes_outside": score.n_nodes_outside if score is not None else None,
        "n_junctions": score.n_junctions if score is not None else 0,
        "n_nodes": score.n_nodes if score is not None else 0,
        "n_leaves": score.n_leaves if score is not None else 0,
        "cyclomatic": report.skeleton_cyclomatic if report is not None else None,
        "mesh_genus": report.mesh_genus if report is not None else None,
        "elapsed_s": elapsed,
        "skel": skel,
        "score_obj": score,
    }
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "attraction_weight",
        "medial_weight",
        "ratio",
        "iters",
        "n0",
        "n_final",
        "growth",
        "aborted_remesh",
        "area_overshoot",
        "nonfinite",
        "score",
        "rejected",
        "reject_reason",
        "topo_delta",
        "nodes_inside_frac",
        "n_nodes_outside",
        "n_junctions",
        "n_nodes",
        "n_leaves",
        "cyclomatic",
        "mesh_genus",
        "elapsed_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def sweep_mesh(
    mesh_path: Path,
    *,
    attraction_weights: list[float],
    ratios: list[float],
    max_iterations: int,
    timeout_seconds: float | None,
    max_vertex_growth: float,
    out_dir: Path,
    top_k: int,
) -> list[dict]:
    mesh = _load_mesh(mesh_path)
    name = mesh_path.stem
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    feats = mesh_mcfs_features(mesh)
    proposed = propose_mcfs_params(mesh, features=feats)
    logger.info("=== %s ===", name)
    logger.info("  features: %s", feats.summary())
    logger.info("  oracle:   %s", proposed.summary())

    rows: list[dict] = []
    for aw in attraction_weights:
        for r in ratios:
            mw = float(aw) * float(r)
            logger.info(
                "  trial attraction_weight=%g medial_weight=%g (r=%g) ...",
                aw,
                mw,
                r,
            )
            row = _run_trial(
                mesh,
                attraction_weight=float(aw),
                medial_weight=mw,
                max_iterations=max_iterations,
                timeout_seconds=timeout_seconds,
                max_vertex_growth=max_vertex_growth,
            )
            logger.info(
                "    score=%.4g rejected=%s growth=%.2f junctions=%s "
                "outside=%s (%.1fs)",
                row["score"],
                row["rejected"],
                row["growth"],
                row["n_junctions"],
                row["n_nodes_outside"],
                row["elapsed_s"],
            )
            rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    csv_path = case_dir / "sweep.csv"
    _write_csv(csv_path, rows_sorted)
    logger.info("  wrote %s", csv_path)

    for i, row in enumerate(rows_sorted[: max(0, top_k)]):
        skel = row.get("skel")
        if skel is None:
            continue
        out = case_dir / (
            f"top{i + 1}_aw{row['attraction_weight']:g}_"
            f"mw{row['medial_weight']:g}.polylines.txt"
        )
        skel.write_polylines(str(out))
        logger.info("  top-%d polylines -> %s score=%.4g", i + 1, out, row["score"])

    if rows_sorted:
        best = rows_sorted[0]
        logger.info(
            "  BEST: attraction_weight=%g medial_weight=%g score=%.4g %s",
            best["attraction_weight"],
            best["medial_weight"],
            best["score"],
            best.get("reject_reason") or "",
        )
    return rows_sorted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mesh",
        action="append",
        default=None,
        help="Mesh path or preset (ts1, ts2, ts76). Repeatable. Default: ts2",
    )
    ap.add_argument(
        "--attraction-weight",
        type=str,
        default=",".join(str(x) for x in DEFAULT_ATTRACTION),
        help="Comma-separated attraction_weight scales (legacy: w_H)",
    )
    ap.add_argument(
        "--ratios",
        type=str,
        default=",".join(str(x) for x in DEFAULT_RATIOS),
        help="Comma-separated medial_weight/attraction_weight ratios",
    )
    ap.add_argument("--max-iterations", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-vertex-growth", type=float, default=4.0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "sweeps",
        help="Output directory root",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    specs = args.mesh or ["ts2"]
    attraction_weights = _parse_floats(args.attraction_weight)
    ratios = _parse_floats(args.ratios)
    timeout = args.timeout if args.timeout > 0 else None

    for spec in specs:
        path = _resolve_mesh(spec)
        if not path.exists():
            logger.error("missing mesh: %s", path)
            return 1
        sweep_mesh(
            path,
            attraction_weights=attraction_weights,
            ratios=ratios,
            max_iterations=args.max_iterations,
            timeout_seconds=timeout,
            max_vertex_growth=args.max_vertex_growth,
            out_dir=args.out,
            top_k=args.top_k,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
