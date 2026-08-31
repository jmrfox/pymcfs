#!/usr/bin/env python3
"""Dump pymcfs MCFS stages into dev/fixtures/parity/<case>/pymcfs/.

Uses the Starlab parity profile by default (attraction_weight=0.1,
medial_weight=0.2, ungated poles). Pass --gate-poles to enable CGAL-style
exterior pole gating.

Usage:
  uv run python dev/scripts/dump_pymcfs_parity.py --case sindorelax
  uv run python dev/scripts/dump_pymcfs_parity.py --case cylinder --iters 1,10,final
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcfs.cg_io import write_cg
from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.medial import compute_voronoi_poles
from pymcfs.parity import FIXTURES_ROOT, find_input_mesh, fixture_dir, load_mesh

logger = logging.getLogger(__name__)


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
    attraction_weight: float,
    medial_weight: float,
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
    logger.info(
        "[%s] input=%s V=%d F=%d",
        case,
        mesh_path.name,
        len(mesh.vertices),
        len(mesh.faces),
    )

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
    logger.info("[%s] wrote poles.npy (%d)", case, poles.shape[0])

    # Stage 2 — meso snapshots
    wanted = _parse_iters(iters_spec, max_iterations)
    numeric = [x for x in wanted if isinstance(x, int)]
    want_final = any(x == "final" for x in wanted)

    driver = MeanCurvatureFlowSkeletonization(
        mesh,
        attraction_weight=float(attraction_weight),
        medial_weight=float(medial_weight),
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
        logger.info(
            "[%s] meso_%s: n=%d f=%d area=%.6g",
            case,
            tag,
            driver.V.shape[0],
            driver.F.shape[0],
            driver._surface_area(),
        )

    prev_area = driver._surface_area()
    converged = False
    for n in range(1, int(max_iterations) + 1):
        if driver._timed_out():
            logger.info("[%s] timeout before iter %d", case, n)
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
            logger.info("[%s] converged at iter %d", case, n)
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

    # Stage 3 — raw curve (no resample)
    skel = driver.convert_to_skeleton(resample=False)
    write_cg(out_dir / "skeleton.cg", skel.nodes, skel.edges)
    np.savez_compressed(
        out_dir / "skeleton.npz",
        nodes=skel.nodes,
        edges=skel.edges,
    )
    logger.info(
        "[%s] skeleton.cg nodes=%d edges=%d",
        case,
        skel.nodes.shape[0],
        skel.edges.shape[0],
    )
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--case",
        type=str,
        action="append",
        default=None,
        help="parity case name under dev/fixtures/parity (repeatable)",
    )
    ap.add_argument("--all", action="store_true", help="dump all cases with an input mesh")
    ap.add_argument(
        "--attraction-weight",
        type=float,
        default=0.1,
        help="Starlab parity default (legacy name: w_H)",
    )
    ap.add_argument(
        "--medial-weight",
        type=float,
        default=0.2,
        help="Starlab parity default (legacy name: w_M)",
    )
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
        help="dev/fixtures/parity root",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

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
            attraction_weight=args.attraction_weight,
            medial_weight=args.medial_weight,
            max_iterations=args.max_iterations,
            timeout_seconds=timeout,
            iters_spec=args.iters,
            root=root,
            gate_exterior_poles=bool(args.gate_poles),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
