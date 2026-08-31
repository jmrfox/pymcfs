#!/usr/bin/env python3
"""Micro-benchmark for MCFS iteration cost.

Usage:
  uv run python dev/scripts/bench_mcfs_iter.py
  uv run python dev/scripts/bench_mcfs_iter.py --mesh dev/fixtures/parity/sindorelax/input.off --iters 5
  uv run python dev/scripts/bench_mcfs_iter.py --profile --iters 1
  uv run python dev/scripts/bench_mcfs_iter.py --mesh toric_spines/data/mesh/TS1.obj --profile --iters 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcfs import mcfs as mcfs_module
from pymcfs.laplacian import mcfs_cotangent_laplacian
from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.parity import load_mesh
from pymcfs.spd_solve import cholmod_available, solve_spd_ata

MESH_PRESETS = {
    "sindorelax": ROOT / "dev" / "fixtures" / "parity" / "sindorelax" / "input.off",
    "ts1": ROOT / "toric_spines" / "data" / "mesh" / "TS1.obj",
}


class _GatingProbe:
    """Charge pole containment to its own phase instead of the calling remesh.

    Gating runs inside ``collapse_edges`` / ``split_faces``, so timing those
    calls as wholes hides the point-in-mesh cost inside the remesh phases.
    This patches the driver's containment entry point, measures it separately,
    and the caller subtracts it back out.
    """

    def __init__(self) -> None:
        self.original = mcfs_module.points_inside_mesh
        self.seconds = 0.0
        self.calls = 0
        self.points = 0
        mcfs_module.points_inside_mesh = self

    def __call__(self, mesh, points, *, fast=False):
        t0 = time.perf_counter()
        result = self.original(mesh, points, fast=fast)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        self.points += len(points)
        return result

    def restore(self) -> None:
        mcfs_module.points_inside_mesh = self.original

    def take(self) -> tuple[float, int, int]:
        """Return and reset the accumulated (seconds, calls, points)."""
        stats = (self.seconds, self.calls, self.points)
        self.seconds, self.calls, self.points = 0.0, 0, 0
        return stats


def _profile_contract(driver: MeanCurvatureFlowSkeletonization) -> dict[str, float]:
    """Time sub-phases of one ``contract()`` call (seconds)."""
    times: dict[str, float] = {}
    probe = _GatingProbe()

    driver._sync_pole_valid()
    wL, wH, wM = driver._update_constraint_weights()

    t0 = time.perf_counter()
    L = mcfs_cotangent_laplacian(driver.V, driver.F).tocsr()
    times["laplacian"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    row_of = np.repeat(np.arange(L.shape[0]), np.diff(L.indptr))
    row_scale = wL[row_of]
    row_scale[L.indices == row_of] = 1.0
    L_weighted = sp.csr_matrix(
        (L.data * row_scale, L.indices, L.indptr), shape=L.shape
    )
    AtA = (
        (L_weighted.T @ L_weighted)
        + sp.diags(wH * wH, format="csr")
        + sp.diags(wM * wM, format="csr")
    ).tocsc()
    At_rhs = wH[:, None] * (wH[:, None] * driver.V) + wM[:, None] * (
        wM[:, None] * driver.poles
    )
    times["ata_assemble"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    solve_spd_ata(AtA, At_rhs, use_cholmod=driver._use_cholmod)
    times["solve"] = time.perf_counter() - t0

    probe.take()
    gating_s = 0.0
    gating_calls = 0
    gating_points = 0

    t0 = time.perf_counter()
    driver.contract_geometry()
    times["geometry"] = time.perf_counter() - t0

    # Each remesh phase is reported net of the gating it triggered internally.
    for phase, run in (("collapse", driver.collapse_edges), ("split", driver.split_faces)):
        t0 = time.perf_counter()
        result = run()
        elapsed = time.perf_counter() - t0
        gate_s, gate_calls, gate_points = probe.take()
        times[phase] = max(elapsed - gate_s, 0.0)
        gating_s += gate_s
        gating_calls += gate_calls
        gating_points += gate_points
        if phase == "split":
            new_splits = result

    driver._constraint_fixed = driver.fixed.copy()
    driver._constraint_split = driver.is_split.copy()
    if new_splits:
        driver._constraint_split[-new_splits:] = False

    t0 = time.perf_counter()
    driver.detect_degeneracies()
    times["degen"] = time.perf_counter() - t0

    times["gating"] = gating_s
    times["gating_calls"] = float(gating_calls)
    times["gating_points"] = float(gating_points)
    probe.restore()
    return times


def _print_profile(times: dict[str, float]) -> None:
    geom_detail = times["laplacian"] + times["ata_assemble"] + times["solve"]
    total = (
        times["geometry"]
        + times["collapse"]
        + times["split"]
        + times["degen"]
        + times["gating"]
    )
    print("profile (one contract, ms):")
    rows = [
        ("geometry", times["geometry"]),
        ("  laplacian", times["laplacian"]),
        ("  ata_assemble", times["ata_assemble"]),
        ("  solve", times["solve"]),
        ("collapse", times["collapse"]),
        ("split", times["split"]),
        ("degen", times["degen"]),
        ("gating (contains)", times["gating"]),
        ("total", total),
    ]
    for name, sec in rows:
        pct = 100.0 * sec / total if total > 0 else 0.0
        print(f"  {name:18s}: {1000.0 * sec:8.1f} ms ({pct:5.1f}%)")
    print(
        f"  (gating: {int(times['gating_calls'])} contains calls, "
        f"{int(times['gating_points'])} points; collapse/split shown net of it)"
    )
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
        default=ROOT / "dev" / "fixtures" / "parity" / "sindorelax" / "input.off",
        help="Mesh path, or preset name: sindorelax, ts1",
    )
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument(
        "--attraction-weight",
        type=float,
        default=0.5,
        help="Attraction weight (legacy name: w_H)",
    )
    ap.add_argument(
        "--medial-weight",
        type=float,
        default=5.0,
        help="Medial-centering weight (legacy name: w_M)",
    )
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
        attraction_weight=args.attraction_weight,
        medial_weight=args.medial_weight,
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
    probe = _GatingProbe()
    t0 = time.perf_counter()
    for i in range(args.iters):
        t_i = time.perf_counter()
        driver.contract()
        print(
            f"  iter {i + 1}: {(time.perf_counter() - t_i) * 1000:.1f} ms "
            f"n={driver.V.shape[0]} f={driver.F.shape[0]}"
        )
    elapsed = time.perf_counter() - t0
    gate_s, gate_calls, gate_points = probe.take()
    probe.restore()
    mean_ms = 1000.0 * elapsed / max(args.iters, 1)
    print(f"iters={args.iters} mean_ms={mean_ms:.2f} total_s={elapsed:.3f}")
    print(
        f"contains_calls={driver._n_contains} "
        f"(+{driver._n_contains - n_contains_before} during contracts) "
        f"gating={gate_s:.3f}s ({100.0 * gate_s / max(elapsed, 1e-9):.1f}%) "
        f"points={gate_points} in {gate_calls} calls"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
