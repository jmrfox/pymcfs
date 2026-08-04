#!/usr/bin/env python3
"""
Demo: load a closed mesh, run MCFS, export SWC + polylines, print quality report.

Usage:
  python scripts/demo_mcfs.py [--mesh PATH] [--outdir PATH] [--backend plotly|matplotlib]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh as tm

from pymcfs.mesh import MeshManager
from pymcfs.skeleton import skeletonize, thin_mesh
from pymcfs.quality import analyze_skeleton


def find_default_mesh() -> str | None:
    root = Path(__file__).resolve().parents[1]
    search_dirs = [root / "data" / "mesh", root / "data" / "mesh" / "processed"]
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
    ap.add_argument("--outdir", type=str, default="outputs/demo")
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "plotly", "matplotlib"])
    ap.add_argument("--w_H", type=float, default=0.1, help="quality_speed_tradeoff")
    ap.add_argument("--w_M", type=float, default=0.2, help="medially_centered_speed_tradeoff")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--timeout", type=float, default=120.0, help="wall-clock seconds (0=unlimited)")
    args = ap.parse_args()

    mesh_path = args.mesh or find_default_mesh()
    if mesh_path is None:
        print("No mesh found under data/mesh/. Pass --mesh PATH.")
        sys.exit(1)

    outdir = ensure_outdir(args.outdir)
    mm = MeshManager(verbose=True)
    m = mm.load_mesh(mesh_path, validate_mcfs=True)
    print(f"Loaded mesh: {len(m.vertices)} vertices, {len(m.faces)} faces")

    fig = mm.visualize_mesh_3d(title=f"Input: {Path(mesh_path).name}", backend=args.backend)

    timeout = None if args.timeout <= 0 else float(args.timeout)
    skel = skeletonize(
        m,
        w_H=args.w_H,
        w_M=args.w_M,
        max_iterations=args.iters,
        timeout_seconds=timeout,
        compress_chains=True,
        verbose=True,
    )
    print(f"Skeleton: {skel.nodes.shape[0]} nodes, {skel.edges.shape[0]} edges")

    report = analyze_skeleton(m, skel)
    print("Quality:", report.summary())

    if fig is not None:
        if args.backend in ("auto", "plotly") and fig.__class__.__module__.startswith("plotly"):
            overlay_skeleton_plotly(fig, skel.nodes, skel.edges)
            out_html = outdir / "skeleton_plotly.html"
            fig.write_html(str(out_html))
            print(f"Wrote {out_html}")
        else:
            overlay_skeleton_matplotlib(fig, skel.nodes, skel.edges)
            out_png = outdir / "skeleton_matplotlib.png"
            fig.savefig(str(out_png), dpi=150)
            print(f"Wrote {out_png}")

    skel.write_swc(str(outdir / "skeleton.swc"))
    skel.write_polylines(str(outdir / "skeleton.polylines.txt"))
    print(f"Wrote SWC + polylines under {outdir}")

    try:
        Vt, Ft = thin_mesh(m, w_H=args.w_H, w_M=args.w_M, max_iterations=min(40, args.iters))
        mt = tm.Trimesh(vertices=Vt, faces=Ft, process=False)
        mm_thin = MeshManager(mt, verbose=False)
        fig2 = mm_thin.visualize_mesh_3d(title="Meso-skeleton", backend=args.backend)
        if fig2 is not None and not fig2.__class__.__module__.startswith("plotly"):
            fig2.savefig(str(outdir / "thinned_matplotlib.png"), dpi=150)
    except Exception as e:
        print(f"Thin-mesh viz skipped: {e}")


if __name__ == "__main__":
    main()
