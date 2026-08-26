#!/usr/bin/env python3
"""Dump pymcfs MCFS stages into fixtures/parity/<case>/pymcfs/.

Uses the Starlab parity profile by default (w_H=0.1, w_M=0.2, ungated poles).
Pass --gate-poles to enable CGAL-style exterior pole gating.

Usage:
  uv run python scripts/dump_pymcfs_parity.py --case sindorelax
  uv run python scripts/dump_pymcfs_parity.py --case cylinder --iters 1,10,final
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcfs.cg_io import write_cg
from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.medial import compute_voronoi_poles
from pymcfs.parity import FIXTURES_ROOT, find_input_mesh, fixture_dir, load_mesh


def _parse_iters(spec: str, max_iterations: int) -> list[int | str]:
    out: list[int | str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "final":
            out.append("final")
        else:
            out.append(int(tok))
    if not out:
        out = [1, max(1, max_iterations // 2), "final"]
    return out


def dump_case(
    case: str,
    *,
    w_H: float,
    w_M: float,
    max_iterations: int,
    timeout_seconds: float | None,
    iters_spec: str,
    root: Path,
    gate_exterior_poles: bool = False,
) -> Path:
    case_dir = fixture_dir(case, root=root)
    out_dir = case_dir / "pymcfs"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = find_input_mesh(case_dir)
    mesh = load_mesh(mesh_path)
    print(f"[{case}] input={mesh_path.name} V={len(mesh.vertices)} F={len(mesh.faces)}")

    # Stage 1 — poles
    poles, weights = compute_voronoi_poles(mesh)
    np.save(out_dir / "poles.npy", poles)
    np.save(out_dir / "pole_weights.npy", weights)
    # Also write a simple OFF of pole positions for easy inspection.
    with (out_dir / "poles.off").open("w", encoding="utf-8") as f:
        f.write("OFF\n")
        f.write(f"{poles.shape[0]} 0 0\n")
        for p in poles:
            f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
    print(f"[{case}] wrote poles.npy ({poles.shape[0]})")

    # Stage 2 — meso snapshots
    wanted = _parse_iters(iters_spec, max_iterations)
    numeric = [x for x in wanted if isinstance(x, int)]
    want_final = any(x == "final" for x in wanted)

    import time

    driver = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=float(w_H),
        w_M=float(w_M),
        gate_exterior_poles=bool(gate_exterior_poles),
        max_iterations=int(max_iterations),
        timeout_seconds=timeout_seconds,
        validate=True,
        verbose=False,
    )
    if timeout_seconds is not None and timeout_seconds > 0:
        driver._deadline = time.monotonic() + float(timeout_seconds)
    else:
        driver._deadline = None

    snapshots_done: set[int] = set()

    def _write_meso(tag: str) -> None:
        mesh_out = out_dir / f"meso_{tag}.off"
        driver.meso_skeleton_mesh().export(str(mesh_out))
        np.savez_compressed(
            out_dir / f"meso_{tag}.npz",
            V=driver.V.copy(),
            F=driver.F.copy(),
            fixed=driver.fixed.copy(),
            is_split=driver.is_split.copy(),
            poles=driver.poles.copy(),
            area=np.array([driver._surface_area()], dtype=float),
            n=np.array([driver.V.shape[0]], dtype=int),
            fcount=np.array([driver.F.shape[0]], dtype=int),
        )
        print(
            f"[{case}] meso_{tag}: n={driver.V.shape[0]} f={driver.F.shape[0]} "
            f"area={driver._surface_area():.6g}"
        )

    prev_area = driver._surface_area()
    converged = False
    for n in range(1, int(max_iterations) + 1):
        if driver._timed_out():
            print(f"[{case}] timeout before iter {n}")
            break
        driver._iter = n
        driver.contract()
        if n in numeric and n not in snapshots_done:
            _write_meso(f"N{n:04d}")
            snapshots_done.add(n)
        area = driver._surface_area()
        if prev_area > 0 and abs(prev_area - area) < driver.area_variation_factor * max(
            driver._area0, 1e-30
        ):
            converged = True
            print(f"[{case}] converged at iter {n}")
            break
        prev_area = area
        if driver.F.shape[0] == 0:
            converged = True
            break
        # Stop early once all requested numeric snapshots are done and final not needed
        if not want_final and numeric and set(numeric).issubset(snapshots_done):
            break

    if want_final:
        _write_meso("final")
    elif converged and numeric and max(numeric) not in snapshots_done:
        # Convergence before a requested snapshot: still record last state under that N
        pass

    # Stage 3 — raw curve (no refine)
    skel = driver.convert_to_skeleton(refine=False)
    write_cg(out_dir / "skeleton.cg", skel.nodes, skel.edges)
    np.savez_compressed(
        out_dir / "skeleton.npz",
        nodes=skel.nodes,
        edges=skel.edges,
    )
    print(
        f"[{case}] skeleton.cg nodes={skel.nodes.shape[0]} edges={skel.edges.shape[0]}"
    )
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--case",
        type=str,
        action="append",
        default=None,
        help="parity case name under fixtures/parity (repeatable)",
    )
    ap.add_argument("--all", action="store_true", help="dump all cases with an input mesh")
    ap.add_argument("--w_H", type=float, default=0.1, help="Starlab parity default")
    ap.add_argument("--w_M", type=float, default=0.2, help="Starlab parity default")
    ap.add_argument(
        "--gate-poles",
        action="store_true",
        help="CGAL-style exterior pole gating (default: off for Starlab parity)",
    )
    ap.add_argument("--max-iterations", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=300.0, help="0 = unlimited")
    ap.add_argument(
        "--iters",
        type=str,
        default="1,final",
        help="comma list of contract counts and/or 'final' (default: 1,final)",
    )
    ap.add_argument(
        "--fixtures-root",
        type=Path,
        default=FIXTURES_ROOT,
        help="fixtures/parity root",
    )
    args = ap.parse_args()

    root = args.fixtures_root
    if args.all:
        from pymcfs.parity import list_parity_cases

        cases = list_parity_cases(root)
    else:
        cases = args.case or ["sindorelax"]

    timeout = None if args.timeout <= 0 else float(args.timeout)
    for case in cases:
        dump_case(
            case,
            w_H=args.w_H,
            w_M=args.w_M,
            max_iterations=args.max_iterations,
            timeout_seconds=timeout,
            iters_spec=args.iters,
            root=root,
            gate_exterior_poles=bool(args.gate_poles),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
