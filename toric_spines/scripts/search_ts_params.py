#!/usr/bin/env python3
"""Run compact MCFS parameter search for one TS mesh.

Uses :func:`pymcfs.search.search_mcfs_params` (~4 contractions + cleanup
variants). Distinct from the full grid in ``sweep_mcfs_params.py``.

Usage:
  uv run python toric_spines/scripts/search_ts_params.py TS3
  uv run python toric_spines/scripts/search_ts_params.py TS76 --extend-tips
  uv run python toric_spines/scripts/search_ts_params.py toric_spines/data/mesh/TS1.obj --return-trials
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcfs import analyze_skeleton, load_and_repair, propose_mcfs_params, search_mcfs_params

logger = logging.getLogger(__name__)

MESH_DIR = ROOT / "toric_spines" / "data" / "mesh"
OUT_ROOT = ROOT / "outputs" / "parameter_search"


def _resolve_mesh(spec: str) -> Path:
    raw = spec.strip()
    stem = Path(raw).name.removesuffix(".obj")
    # Accept TS3 / ts3 / TS3.obj
    if stem.upper().startswith("TS"):
        p = MESH_DIR / f"{stem.upper()}.obj"
        if p.is_file():
            return p
    p = Path(raw)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if p.is_file():
        return p
    raise FileNotFoundError(f"mesh not found: {spec!r} (looked under {MESH_DIR})")


def _write_outputs(
    case_dir: Path,
    *,
    name: str,
    result,
    mesh,
    elapsed_s: float,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    skel = result.skeleton
    report = analyze_skeleton(mesh, skel) if skel.nodes.shape[0] else None
    score = result.score

    if skel.nodes.shape[0] > 0:
        skel.write_polylines(str(case_dir / "skeleton_search.polylines.txt"))
        skel.write_cg(str(case_dir / "skeleton_search.cg"))

    quality = case_dir / "quality.txt"
    quality.write_text(
        "\n".join(
            [
                f"name={name}",
                f"source=parameter_search",
                f"attraction_weight={result.attraction_weight:g} "
                f"medial_weight={result.medial_weight:g}",
                f"gate_exterior_poles={result.gate_exterior_poles}",
                f"keep_hub_branches={result.keep_hub_branches}",
                f"tip_extend_scale={result.tip_extend_scale:g}",
                f"n_contracts={result.n_contracts} n_converts={result.n_converts}",
                f"elapsed_s={elapsed_s:.2f}",
                report.summary() if report is not None else "report=none",
                score.summary(),
                "",
            ]
        ),
        encoding="utf-8",
    )

    if result.trials:
        trials_path = case_dir / "trials.csv"
        with trials_path.open("w", newline="", encoding="utf-8") as f:
            fields = [
                "attraction_weight",
                "medial_weight",
                "keep_hub_branches",
                "tip_extend_scale",
                "score",
                "rejected",
                "reject_reason",
                "n_nodes",
                "n_junctions",
                "n_leaves",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in result.trials:
                sc = t["score"]
                w.writerow(
                    {
                        "attraction_weight": t["attraction_weight"],
                        "medial_weight": t["medial_weight"],
                        "keep_hub_branches": t["keep_hub_branches"],
                        "tip_extend_scale": t["tip_extend_scale"],
                        "score": float(sc.value),
                        "rejected": int(sc.rejected),
                        "reject_reason": sc.reject_reason or "",
                        "n_nodes": sc.n_nodes,
                        "n_junctions": sc.n_junctions,
                        "n_leaves": sc.n_leaves,
                    }
                )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "mesh",
        help="TS stem (e.g. TS3) or path to .obj under toric_spines/data/mesh/",
    )
    ap.add_argument("--profile", default="auto", choices=("auto", "robust", "starlab"))
    ap.add_argument("--branching", default="sparse", choices=("sparse", "balanced", "dense"))
    ap.add_argument("--max-iterations", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=300.0, help="0 = no timeout")
    ap.add_argument("--max-search-contracts", type=int, default=4)
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
        help="Extend unfinished tips (default: on)",
    )
    ap.add_argument("--tip-extend-scale", type=float, default=1.0)
    ap.add_argument(
        "--return-trials",
        action="store_true",
        help="Write trials.csv with every contract/cleanup evaluation",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_ROOT,
        help=f"Output root (default {OUT_ROOT})",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    path = _resolve_mesh(args.mesh)
    name = path.stem
    out_root = args.out if args.out.is_absolute() else (ROOT / args.out)
    timeout = None if args.timeout <= 0 else float(args.timeout)

    logger.info("=== parameter_search %s (%s) ===", name, path)
    mesh = load_and_repair(str(path))
    proposed = propose_mcfs_params(mesh, branching=args.branching)
    logger.info("  oracle base: %s", proposed.summary())
    logger.info(
        "  search: profile=%s contracts≤%d thick_hubs=%s extend_tips=%s",
        args.profile,
        args.max_search_contracts,
        not args.no_thick_hubs,
        args.extend_tips,
    )

    t0 = time.perf_counter()
    result = search_mcfs_params(
        mesh,
        profile=args.profile,
        branching=args.branching,
        max_iterations=int(args.max_iterations),
        timeout_seconds=timeout,
        validate=False,
        max_search_contracts=int(args.max_search_contracts),
        prune_thick_hubs=not bool(args.no_thick_hubs),
        keep_hub_branches=int(args.keep_hub_branches),
        extend_tips=bool(args.extend_tips),
        tip_extend_scale=float(args.tip_extend_scale),
        return_trials=bool(args.return_trials) or bool(args.verbose),
        verbose=bool(args.verbose),
    )
    elapsed = time.perf_counter() - t0

    case_dir = out_root / name
    _write_outputs(case_dir, name=name, result=result, mesh=mesh, elapsed_s=elapsed)

    logger.info(
        "  best: attraction_weight=%g medial_weight=%g keep=%d tip_s=%g",
        result.attraction_weight,
        result.medial_weight,
        result.keep_hub_branches,
        result.tip_extend_scale,
    )
    logger.info(
        "  score=%s contracts=%d converts=%d (%.1fs)",
        result.score.summary(),
        result.n_contracts,
        result.n_converts,
        elapsed,
    )
    logger.info("  wrote %s", case_dir)
    return 0 if not result.score.rejected and result.skeleton.nodes.shape[0] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
