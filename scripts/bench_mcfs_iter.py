#!/usr/bin/env python3
"""Micro-benchmark for Tier A MCFS iteration cost.

Usage:
  uv run python scripts/bench_mcfs_iter.py
  uv run python scripts/bench_mcfs_iter.py --mesh fixtures/parity/sindorelax/input.off --iters 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.parity import load_mesh
from pymcfs.spd_solve import cholmod_available


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mesh",
        type=Path,
        default=ROOT / "fixtures" / "parity" / "sindorelax" / "input.off",
    )
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--w_H", type=float, default=0.5)
    ap.add_argument("--w_M", type=float, default=5.0)
    ap.add_argument(
        "--no-cholmod",
        action="store_true",
        help="Force SciPy SuperLU even if CHOLMOD is available",
    )
    args = ap.parse_args()
    if not args.mesh.is_file():
        print(f"mesh not found: {args.mesh}", file=sys.stderr)
        return 1

    mesh = load_mesh(args.mesh)
    print(f"loading done: {args.mesh.name} n={len(mesh.vertices)} f={len(mesh.faces)}")
    t_init = time.perf_counter()
    driver = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=args.w_H,
        w_M=args.w_M,
        gate_exterior_poles=True,
        use_cholmod=False if args.no_cholmod else None,
        max_iterations=args.iters,
        timeout_seconds=None,
        verbose=False,
    )
    print(
        f"init_s={time.perf_counter() - t_init:.3f} "
        f"spd={'cholmod' if driver._use_cholmod else 'superlu'} "
        f"contains={driver._n_contains} cholmod_available={cholmod_available()}"
    )
    n_contains_before = int(driver._n_contains)
    t0 = time.perf_counter()
    for i in range(args.iters):
        t_i = time.perf_counter()
        driver.contract()
        print(
            f"  iter {i + 1}: {(time.perf_counter() - t_i) * 1000:.1f} ms "
            f"n={driver.V.shape[0]} f={driver.F.shape[0]}"
        )
    elapsed = time.perf_counter() - t0
    mean_ms = 1000.0 * elapsed / max(args.iters, 1)
    print(f"iters={args.iters} mean_ms={mean_ms:.2f} total_s={elapsed:.3f}")
    print(
        f"contains_calls={driver._n_contains} "
        f"(+{driver._n_contains - n_contains_before} during contracts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
