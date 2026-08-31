#!/usr/bin/env python3
"""
Demo: load a closed mesh, run MCFS, export polylines, print quality report.

Usage:
  uv run python toric_spines/scripts/demo_mcfs.py [--mesh PATH] [--out PATH] [--backend plotly|matplotlib]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import trimesh as tm

from pymcfs import MeshManager, analyze_skeleton, contract_mesh, load_and_repair, skeletonize

logger = logging.getLogger(__name__)


def find_default_mesh() -> str | None:
    root = Path(__file__).resolve().parents[2]
    search_dirs = [
        root / "toric_spines" / "data" / "mesh",
        root / "toric_spines" / "data" / "mesh" / "processed",
    ]
    exts = ("*.obj", "*.ply", "*.stl", "*.off")
    for search_root in search_dirs:
        if not search_root.is_dir():
            continue
        for pat in exts:
            files = sorted(search_root.glob(pat))
            for f in files:
                if "TS2" in f.name:
                    return str(f)
            if files:
                return str(files[0])
    return None


def ensure_outdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def overlay_skeleton_matplotlib(fig, nodes: np.ndarray, edges: np.ndarray) -> None:
    if nodes.size == 0 or edges.size == 0 or not fig.axes:
        return
    ax = fig.axes[0]
    for u, v in edges:
        p, q = nodes[int(u)], nodes[int(v)]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="crimson", linewidth=2)


def overlay_skeleton_plotly(fig, nodes: np.ndarray, edges: np.ndarray) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        return
    if nodes.size == 0 or edges.size == 0:
        return
    xs, ys, zs = [], [], []
    for u, v in edges:
        p, q = nodes[int(u)], nodes[int(v)]
        xs += [p[0], q[0], None]
        ys += [p[1], q[1], None]
        zs += [p[2], q[2], None]
    fig.add_trace(
        go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="crimson", width=4), name="Skel")
    )


def main():
    ap = argparse.ArgumentParser(description="pymcfs demo: MCFS + quality report")
    ap.add_argument("--mesh", type=str, default=None)
    ap.add_argument("--out", type=str, default="outputs/demo")
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "plotly", "matplotlib"])
    ap.add_argument(
        "--attraction-weight",
        type=float,
        default=0.1,
        help="Attraction weight (legacy name: w_H)",
    )
    ap.add_argument(
        "--medial-weight",
        type=float,
        default=0.2,
        help="Medial-centering weight (legacy name: w_M)",
    )
    ap.add_argument("--max-iterations", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=120.0, help="wall-clock seconds (0=unlimited)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    mesh_path = args.mesh or find_default_mesh()
    if mesh_path is None:
        logger.error("No mesh found under toric_spines/data/mesh/. Pass --mesh PATH.")
        sys.exit(1)

    outdir = ensure_outdir(args.out)
    m = load_and_repair(mesh_path)
    mm = MeshManager(m, verbose=args.verbose)
    logger.info("Loaded mesh: %d vertices, %d faces", len(m.vertices), len(m.faces))

    fig = mm.visualize_mesh_3d(title=f"Input: {Path(mesh_path).name}", backend=args.backend)

    timeout = None if args.timeout <= 0 else float(args.timeout)
    skel = skeletonize(
        m,
        attraction_weight=args.attraction_weight,
        medial_weight=args.medial_weight,
        max_iterations=args.max_iterations,
        timeout_seconds=timeout,
        resample=True,
        validate=False,
        verbose=args.verbose,
    )
    logger.info("Skeleton: %d nodes, %d edges", skel.nodes.shape[0], skel.edges.shape[0])

    report = analyze_skeleton(m, skel)
    logger.info("Quality: %s", report.summary())

    if fig is not None:
        if args.backend in ("auto", "plotly") and fig.__class__.__module__.startswith("plotly"):
            overlay_skeleton_plotly(fig, skel.nodes, skel.edges)
            out_html = outdir / "skeleton_plotly.html"
            fig.write_html(str(out_html))
            logger.info("Wrote %s", out_html)
        else:
            overlay_skeleton_matplotlib(fig, skel.nodes, skel.edges)
            out_png = outdir / "skeleton_matplotlib.png"
            fig.savefig(str(out_png), dpi=150)
            logger.info("Wrote %s", out_png)

    skel.write_polylines(str(outdir / "skeleton.polylines.txt"))
    logger.info("Wrote polylines under %s", outdir)

    try:
        Vt, Ft = contract_mesh(
            m,
            attraction_weight=args.attraction_weight,
            medial_weight=args.medial_weight,
            max_iterations=min(40, args.max_iterations),
            validate=False,
        )
        mt = tm.Trimesh(vertices=Vt, faces=Ft, process=False)
        mm_thin = MeshManager(mt, verbose=False)
        fig2 = mm_thin.visualize_mesh_3d(title="Meso-skeleton", backend=args.backend)
        if fig2 is not None and not fig2.__class__.__module__.startswith("plotly"):
            fig2.savefig(str(outdir / "thinned_matplotlib.png"), dpi=150)
    except Exception as e:
        logger.warning("Contract-mesh viz skipped: %s", e)


if __name__ == "__main__":
    main()
