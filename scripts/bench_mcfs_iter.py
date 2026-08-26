#!/usr/bin/env python3
"""Micro-benchmark for MCFS iteration cost.

Usage:
  uv run python scripts/bench_mcfs_iter.py
  uv run python scripts/bench_mcfs_iter.py --mesh fixtures/parity/sindorelax/input.off --iters 5
  uv run python scripts/bench_mcfs_iter.py --profile --iters 1
  uv run python scripts/bench_mcfs_iter.py --mesh data/mesh/TS1.obj --profile --iters 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcfs.laplacian import mcfs_cotangent_laplacian
from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.parity import load_mesh
from pymcfs.spd_solve import cholmod_available, solve_spd_ata

MESH_PRESETS = {
    "sindorelax": ROOT / "fixtures" / "parity" / "sindorelax" / "input.off",
    "ts1": ROOT / "data" / "mesh" / "TS1.obj",
}


def _profile_contract(driver: MeanCurvatureFlowSkeletonization) -> dict[str, float]:
    """Time sub-phases of one ``contract()`` call (seconds)."""
    times: dict[str, float] = {}

    driver._sync_pole_valid()
    wL, wH, wM = driver._update_constraint_weights()

    t0 = time.perf_counter()
    L = mcfs_cotangent_laplacian(driver.V, driver.F).tocsr()
    times["laplacian"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    diag = np.asarray(L.diagonal()).ravel()
    L_off = L - sp.diags(diag, format="csr", shape=L.shape)
    L_weighted = (sp.diags(wL) @ L_off) + sp.diags(diag, format="csr")
    WH = sp.diags(wH, format="csr")
    WP = sp.diags(wM, format="csr")
    A = sp.vstack([L_weighted, WH, WP], format="csc")
    rhs = np.vstack(
        [
            np.zeros_like(driver.V),
            wH[:, None] * driver.V,
            wM[:, None] * driver.poles,
        ]
    )
    AtA = (A.T @ A).tocsc()
    At_rhs = A.T @ rhs
    times["ata_assemble"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    solve_spd_ata(AtA, At_rhs, use_cholmod=driver._use_cholmod)
    times["solve"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    driver.contract_geometry()
    times["geometry"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    driver.collapse_edges()
    times["collapse"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_splits = driver.split_faces()
    times["split"] = time.perf_counter() - t0

    driver._constraint_fixed = driver.fixed.copy()
    driver._constraint_split = driver.is_split.copy()
    if new_splits:
        driver._constraint_split[-new_splits:] = False

    t0 = time.perf_counter()
    driver.detect_degeneracies()
    times["degen"] = time.perf_counter() - t0

    return times


def _print_profile(times: dict[str, float]) -> None:
    geom_detail = times["laplacian"] + times["ata_assemble"] + times["solve"]
    contract_total = (
        times["geometry"]
        + times["collapse"]
        + times["split"]
        + times["degen"]
    )
    total = contract_total
    print("profile (one contract, ms):")
    rows = [
        ("geometry", times["geometry"]),
        ("  laplacian", times["laplacian"]),
        ("  ata_assemble", times["ata_assemble"]),
        ("  solve", times["solve"]),
        ("collapse", times["collapse"]),
        ("split", times["split"]),
        ("degen", times["degen"]),
        ("total", total),
    ]
    for name, sec in rows:
        pct = 100.0 * sec / total if total > 0 else 0.0
        print(f"  {name:16s}: {1000.0 * sec:8.1f} ms ({pct:5.1f}%)")
    if geom_detail > 0 and abs(times["geometry"] - geom_detail) > 1e-6:
        print(
            f"  (geometry detail sum {1000.0 * geom_detail:.1f} ms "
            f"vs contract_geometry {1000.0 * times['geometry']:.1f} ms)"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mesh",
        type=Path,
        default=ROOT / "fixtures" / "parity" / "sindorelax" / "input.off",
        help="Mesh path, or preset name: sindorelax, ts1",
    )
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--w_H", type=float, default=0.5)
    ap.add_argument("--w_M", type=float, default=5.0)
    ap.add_argument(
        "--no-cholmod",
        action="store_true",
        help="Force SciPy SuperLU even if CHOLMOD is available",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Print per-phase timing breakdown for one contract() call",
    )
    args = ap.parse_args()

    mesh_path = MESH_PRESETS.get(str(args.mesh).lower(), args.mesh)
    if not mesh_path.is_file():
        print(f"mesh not found: {mesh_path}", file=sys.stderr)
        print(f"presets: {', '.join(sorted(MESH_PRESETS))}", file=sys.stderr)
        return 1

    mesh = load_mesh(mesh_path)
    print(f"loading done: {mesh_path.name} n={len(mesh.vertices)} f={len(mesh.faces)}")
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

    if args.profile:
        _print_profile(_profile_contract(driver))

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
