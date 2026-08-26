#!/usr/bin/env python3
"""Compare pymcfs vs Starlab stage dumps under fixtures/parity/<case>/.

Usage:
  uv run python scripts/compare_starlab_parity.py --case sindorelax
  uv run python scripts/compare_starlab_parity.py --case sindorelax --stage poles
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcfs.parity import (
    FIXTURES_ROOT,
    compare_curves,
    compare_point_clouds,
    compare_poles,
    find_input_mesh,
    fixture_dir,
    iter_meso_snapshots,
    list_parity_cases,
    load_curve_graph,
    load_mesh,
    load_meso_vertices,
    read_starlab_poles_off,
)


def _load_pymcfs_poles(side: Path) -> np.ndarray:
    npy = side / "poles.npy"
    if npy.is_file():
        return np.load(npy)
    off = side / "poles.off"
    if off.is_file():
        # plain OFF of pole positions
        from pymcfs.parity import load_mesh as _lm

        return np.asarray(_lm(off).vertices, dtype=float)
    raise FileNotFoundError(f"no pymcfs poles in {side}")


def _load_starlab_poles(side: Path) -> np.ndarray:
    off = side / "poles.off"
    if not off.is_file():
        raise FileNotFoundError(f"no starlab poles.off in {side}")
    # Prefer medial nOFF parser; fall back to plain OFF.
    text = off.read_text(encoding="utf-8", errors="replace")[:200].upper()
    if "NOFF" in text or "MEDIAL" in text:
        return read_starlab_poles_off(off)
    from pymcfs.parity import load_mesh as _lm

    return np.asarray(_lm(off).vertices, dtype=float)


def compare_case(
    case: str,
    *,
    root: Path,
    stages: set[str],
    pole_rel_mean: float,
    pole_rel_max: float,
    meso_chamfer_rel: float,
    curve_onesided: float,
) -> list[str]:
    """Return a list of failure messages (empty => pass)."""
    failures: list[str] = []
    case_dir = fixture_dir(case, root=root)
    star = case_dir / "starlab"
    py = case_dir / "pymcfs"
    mesh = load_mesh(find_input_mesh(case_dir))
    V0 = np.asarray(mesh.vertices, dtype=float)
    diag = float(np.linalg.norm(V0.max(0) - V0.min(0))) or 1.0

    print(f"\n=== {case} (bbox_diag={diag:.6g}) ===")

    if "poles" in stages:
        try:
            p_star = _load_starlab_poles(star)
            p_py = _load_pymcfs_poles(py)
            r = compare_poles(p_py, p_star, surface_points=V0)
            print(
                f"poles: mean={r.mean:.6g} max={r.max:.6g} "
                f"frac>thr={r.frac_above:.4f} thr={r.threshold:.6g}"
            )
            if r.mean > pole_rel_mean * r.bbox_diag:
                failures.append(
                    f"{case} poles mean {r.mean:.6g} > {pole_rel_mean}*diag "
                    f"({pole_rel_mean * r.bbox_diag:.6g})"
                )
            if r.max > pole_rel_max * r.bbox_diag:
                failures.append(
                    f"{case} poles max {r.max:.6g} > {pole_rel_max}*diag "
                    f"({pole_rel_max * r.bbox_diag:.6g})"
                )
        except FileNotFoundError as e:
            print(f"poles: SKIP ({e})")

    if "meso" in stages:
        star_meso = {n: p for n, p in iter_meso_snapshots(star)}
        py_meso = {n: p for n, p in iter_meso_snapshots(py)}
        for side, bucket in ((star, star_meso), (py, py_meso)):
            for name in ("meso_final.off", "meso_final.obj", "meso_final.npz"):
                p = side / name
                if p.is_file():
                    bucket[10**9] = p  # sentinel key for final
                    break
        common = sorted(set(star_meso) & set(py_meso))
        if not common:
            print("meso: SKIP (need matching meso_N**** / meso_final on both sides)")
        for n in common:
            label = "final" if n >= 10**9 else f"N{n:04d}"
            Va = load_meso_vertices(py_meso[n])
            Vb = load_meso_vertices(star_meso[n])
            r = compare_point_clouds(Va, Vb, bbox_ref=V0)
            print(
                f"meso_{label}: chamfer={r.chamfer:.6g} "
                f"({r.chamfer / diag:.4g}x diag) "
                f"n={r.n_a}/{r.n_b} "
                f"a->b={r.mean_a_to_b:.6g} b->a={r.mean_b_to_a:.6g}"
            )
            if r.chamfer > meso_chamfer_rel * diag:
                failures.append(
                    f"{case} meso_{label} chamfer {r.chamfer:.6g} > "
                    f"{meso_chamfer_rel}*diag ({meso_chamfer_rel * diag:.6g})"
                )

    if "curve" in stages:
        star_cg = star / "skeleton.cg"
        py_cg = py / "skeleton.cg"
        if not star_cg.is_file() or not py_cg.is_file():
            print("curve: SKIP (need starlab/skeleton.cg and pymcfs/skeleton.cg)")
        else:
            Ga = load_curve_graph(py_cg)
            Gb = load_curve_graph(star_cg)
            r = compare_curves(Ga, Gb, bbox_ref=V0)
            print(
                f"curve: onesided py->star={r.one_sided_a_to_b:.6g} "
                f"star->py={r.one_sided_b_to_a:.6g} "
                f"leaves={r.n_leaves_a}/{r.n_leaves_b} "
                f"junc={r.n_junctions_a}/{r.n_junctions_b} "
                f"cyc={r.cyclomatic_a}/{r.cyclomatic_b}"
            )
            if max(r.one_sided_a_to_b, r.one_sided_b_to_a) > curve_onesided:
                failures.append(
                    f"{case} curve onesided max "
                    f"{max(r.one_sided_a_to_b, r.one_sided_b_to_a):.6g} > {curve_onesided}"
                )

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument(
        "--stage",
        action="append",
        default=None,
        help="stage name or comma-list: poles,meso,curve (repeatable)",
    )
    ap.add_argument("--fixtures-root", type=Path, default=FIXTURES_ROOT)
    ap.add_argument("--pole-rel-mean", type=float, default=0.01)
    ap.add_argument("--pole-rel-max", type=float, default=0.05)
    ap.add_argument("--meso-chamfer-rel", type=float, default=0.05)
    ap.add_argument("--curve-onesided", type=float, default=0.05)
    args = ap.parse_args()

    root = args.fixtures_root
    cases = list_parity_cases(root) if args.all else (args.case or ["sindorelax"])
    if args.stage:
        stages: set[str] = set()
        for item in args.stage:
            for tok in str(item).split(","):
                tok = tok.strip().lower()
                if tok not in {"poles", "meso", "curve"}:
                    raise SystemExit(f"invalid stage {tok!r} (choose poles, meso, curve)")
                stages.add(tok)
    else:
        stages = {"poles", "meso", "curve"}

    all_failures: list[str] = []
    for case in cases:
        all_failures.extend(
            compare_case(
                case,
                root=root,
                stages=stages,
                pole_rel_mean=args.pole_rel_mean,
                pole_rel_max=args.pole_rel_max,
                meso_chamfer_rel=args.meso_chamfer_rel,
                curve_onesided=args.curve_onesided,
            )
        )

    if all_failures:
        print("\nFAILURES:")
        for msg in all_failures:
            print(f"  - {msg}")
        return 1
    print("\nAll compared stages within tolerances.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
