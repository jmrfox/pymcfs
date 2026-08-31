#!/usr/bin/env python3
"""Attempt to dump Starlab reference stages via starterm.exe (Windows demo).

The bundled ``mcfskel-v1.1-win32`` demo can run ``Voronoi based MAT``, but
``MCF Skeletonization`` currently access-violates on this machine (likely
missing CHOLMOD runtime deps). This script still automates what works and
documents the expected layout for hand-copied meso/curve dumps.

Usage:
  uv run python dev/scripts/dump_starlab_parity.py --case sindorelax
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STARTERM = (
    ROOT
    / "_ref_starlab-mcfskel"
    / "Downloads"
    / "StarlabPackageRelease"
    / "starterm.exe"
)


def _run_starterm(starterm: Path, filters: list[str], mesh: Path, *, overwrite: bool) -> int:
    cmd = [str(starterm)]
    for f in filters:
        cmd.append(f"--filter={f}")
    cmd.append("--save-overwrite" if overwrite else "--save")
    cmd.append(str(mesh))
    print(">", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(starterm.parent))
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True)
    ap.add_argument("--starterm", type=Path, default=DEFAULT_STARTERM)
    ap.add_argument(
        "--fixtures-root",
        type=Path,
        default=ROOT / "dev" / "fixtures" / "parity",
    )
    ap.add_argument(
        "--mcf-iters",
        type=int,
        default=1,
        help="number of MCF Skeletonization applies (each is one iteration)",
    )
    args = ap.parse_args()

    if not args.starterm.is_file():
        print(f"starterm not found: {args.starterm}", file=sys.stderr)
        print(
            "Download mcfskel-v1.1-win32.zip from ataiya/starlab-mcfskel Downloads "
            "and extract to _ref_starlab-mcfskel/Downloads/",
            file=sys.stderr,
        )
        return 2

    case_dir = args.fixtures_root / args.case
    star_dir = case_dir / "starlab"
    star_dir.mkdir(parents=True, exist_ok=True)

    # Prefer existing committed poles; only regenerate work mesh for MCF attempts.
    work = case_dir / "_starlab_work"
    work.mkdir(parents=True, exist_ok=True)
    src = None
    for name in ("input.off", "input.obj"):
        if (case_dir / name).is_file():
            src = case_dir / name
            break
    if src is None:
        print(f"no input mesh in {case_dir}", file=sys.stderr)
        return 2

    mesh = work / f"mesh{src.suffix}"
    shutil.copy2(src, mesh)

    rc = _run_starterm(args.starterm, ["Voronoi based MAT"], mesh, overwrite=True)
    if rc != 0:
        print(f"Voronoi based MAT failed with code {rc}", file=sys.stderr)
        return rc

    # Poles: keep committed sindorelax/starlab/poles.off when present.
    if not (star_dir / "poles.off").is_file():
        print(
            "Note: starterm OFF save does not write medial nOFF poles; "
            "copy sindorelax_poles.off manually or keep the committed fixture.",
            file=sys.stderr,
        )

    for i in range(1, int(args.mcf_iters) + 1):
        rc = _run_starterm(args.starterm, ["MCF Skeletonization"], mesh, overwrite=True)
        if rc != 0:
            print(
                f"MCF Skeletonization crashed/failed at iter {i} (code {rc}). "
                "Place meso_N****.off into starlab/ manually from a working Starlab build.",
                file=sys.stderr,
            )
            return rc
        dest = star_dir / f"meso_N{i:04d}{mesh.suffix}"
        shutil.copy2(mesh, dest)
        print(f"wrote {dest}")

    print(
        "Curve stage: in Starlab GUI run 'SurfaceMesh to Skeleton' and save skeleton.cg "
        f"into {star_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
