#!/usr/bin/env python3
"""Grid-search MCFS (w_H, w_M) and score skeletons.

Sweeps in (scale, ratio) space with ``w_M = ratio * w_H`` to avoid the unstable
low-``w_H`` / high-``w_M`` corner. Ranks trials with :func:`pymcfs.quality.score_skeleton`
(topology match, containment, compactness) and early-aborts remesh blow-ups.

Usage:
  uv run python scripts/sweep_mcfs_params.py --mesh data/mesh/TS2.obj
  uv run python scripts/sweep_mcfs_params.py --mesh ts2 --mesh ts1 --top-k 3
  uv run python scripts/sweep_mcfs_params.py --mesh ts2 --w-H 0.5,1.0 --ratios 5,10
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import trimesh as tm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.params import mesh_mcfs_features, propose_mcfs_params
from pymcfs.quality import analyze_skeleton, score_skeleton

MESH_PRESETS = {
    "ts1": ROOT / "data" / "mesh" / "TS1.obj",
    "ts2": ROOT / "data" / "mesh" / "TS2.obj",
    "ts76": ROOT / "data" / "mesh" / "TS76.obj",
}

DEFAULT_W_H = (0.25, 0.5, 1.0, 2.0)
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
    w_H: float,
    w_M: float,
    max_iterations: int,
    timeout_seconds: float | None,
    max_vertex_growth: float,
) -> dict:
    t0 = time.perf_counter()
    driver = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=float(w_H),
        w_M=float(w_M),
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
        skel = driver.convert_to_skeleton(refine=False)
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
        from pymcfs.quality import SkeletonQualityReport

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
        "w_H": w_H,
        "w_M": w_M,
        "ratio": w_M / w_H if w_H else float("nan"),
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
        "w_H",
        "w_M",
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
    w_H_values: list[float],
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
    print(f"=== {name} ===")
    print(f"  features: {feats.summary()}")
    print(f"  oracle:   {proposed.summary()}")

    rows: list[dict] = []
    for wh in w_H_values:
        for r in ratios:
            wm = float(wh) * float(r)
            print(f"  trial w_H={wh:g} w_M={wm:g} (r={r:g}) ...", flush=True)
            row = _run_trial(
                mesh,
                w_H=float(wh),
                w_M=wm,
                max_iterations=max_iterations,
                timeout_seconds=timeout_seconds,
                max_vertex_growth=max_vertex_growth,
            )
            print(
                f"    score={row['score']:.4g} rejected={row['rejected']} "
                f"growth={row['growth']:.2f} junctions={row['n_junctions']} "
                f"outside={row['n_nodes_outside']} ({row['elapsed_s']:.1f}s)",
                flush=True,
            )
            rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    csv_path = case_dir / "sweep.csv"
    _write_csv(csv_path, rows_sorted)
    print(f"  wrote {csv_path}")

    for i, row in enumerate(rows_sorted[: max(0, top_k)]):
        skel = row.get("skel")
        if skel is None:
            continue
        out = case_dir / (
            f"top{i + 1}_wH{row['w_H']:g}_wM{row['w_M']:g}.polylines.txt"
        )
        skel.write_polylines(str(out))
        print(f"  top-{i + 1} polylines -> {out} score={row['score']:.4g}")

    if rows_sorted:
        best = rows_sorted[0]
        print(
            f"  BEST: w_H={best['w_H']:g} w_M={best['w_M']:g} "
            f"score={best['score']:.4g} {best.get('reject_reason') or ''}"
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
        "--w-H",
        type=str,
        default=",".join(str(x) for x in DEFAULT_W_H),
        help="Comma-separated w_H scales",
    )
    ap.add_argument(
        "--ratios",
        type=str,
        default=",".join(str(x) for x in DEFAULT_RATIOS),
        help="Comma-separated w_M/w_H ratios",
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
    args = ap.parse_args()

    specs = args.mesh or ["ts2"]
    w_H_values = _parse_floats(args.w_H)
    ratios = _parse_floats(args.ratios)
    timeout = args.timeout if args.timeout > 0 else None

    for spec in specs:
        path = _resolve_mesh(spec)
        if not path.exists():
            print(f"missing mesh: {path}", file=sys.stderr)
            return 1
        sweep_mesh(
            path,
            w_H_values=w_H_values,
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
