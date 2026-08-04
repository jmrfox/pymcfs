#!/usr/bin/env python3
"""
Demo script for pymcfs: load a mesh, run CGAL-style MCFS, visualize, export SWC.

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


def find_default_mesh() -> str | None:
    root = Path(__file__).resolve().parents[1]
    search_dirs = [
        root / "data" / "mesh",
        root / "data" / "mesh" / "processed",
    ]
    exts = ("*.obj", "*.ply", "*.stl", "*.off", "*.glb", "*.gltf")
    for search_root in search_dirs:
        if not search_root.is_dir():
            continue
        for pat in exts:
            files = sorted(search_root.glob(pat))
            # Prefer TS2 if present
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


def overlay_skeleton_plotly(fig, nodes: np.ndarray, edges: np.ndarray, name_prefix: str = "Skel") -> None:
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
        go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="crimson", width=4), name=f"{name_prefix} Edges",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode="markers",
            marker=dict(size=3, color="black"), name=f"{name_prefix} Nodes",
        )
    )


def overlay_skeleton_matplotlib(fig, nodes: np.ndarray, edges: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception:
        return
    if nodes.size == 0 or edges.size == 0 or not fig.axes:
        return
    ax = fig.axes[0]
    for u, v in edges:
        p, q = nodes[int(u)], nodes[int(v)]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="crimson", linewidth=2)


def main():
    ap = argparse.ArgumentParser(description="pymcfs demo: MCFS skeletonization + SWC export")
    ap.add_argument("--mesh", type=str, default=None, help="Path to input mesh")
    ap.add_argument("--outdir", type=str, default="outputs/demo", help="Output directory")
    ap.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "plotly", "matplotlib"],
        help="Visualization backend",
    )
    ap.add_argument("--guidance", type=str, default="voronoi", choices=["none", "voronoi"])
    ap.add_argument("--iters", type=int, default=40, help="Max MCFS iterations")
    ap.add_argument("--compress", action="store_true", help="Compress degree-2 chains")
    ap.add_argument("--resample", type=float, default=0.0, help="Resample spacing (0 disables)")
    args = ap.parse_args()

    mesh_path = args.mesh or find_default_mesh()
    if mesh_path is None:
        print("No mesh found under data/mesh/. Pass --mesh PATH.")
        sys.exit(1)

    outdir = ensure_outdir(args.outdir)
    mm = MeshManager(verbose=True)
    m = mm.load_mesh(mesh_path)
    print(f"Loaded mesh: {len(m.vertices)} vertices, {len(m.faces)} faces")

    fig = mm.visualize_mesh_3d(title=f"Input Mesh: {Path(mesh_path).name}", backend=args.backend)

    guidance_type = None if args.guidance == "none" else "voronoi"
    skel = skeletonize(
        m,
        mcf_iters=args.iters,
        guidance_type=guidance_type,
        omega_L=1.0,
        omega_H=0.1,
        omega_P=0.2,
        compress_chains=bool(args.compress) or True,
        resample_spacing=(args.resample if args.resample > 0 else None),
        verbose=True,
    )
    print(f"Skeleton: {skel.nodes.shape[0]} nodes, {skel.edges.shape[0]} edges")

    if fig is not None:
        if args.backend in ("auto", "plotly") and fig.__class__.__module__.startswith("plotly"):
            overlay_skeleton_plotly(fig, skel.nodes, skel.edges)
            out_html = outdir / "skeleton_plotly.html"
            try:
                fig.write_html(str(out_html))
                print(f"Wrote visualization: {out_html}")
            except Exception as e:
                print(f"Failed to write plotly HTML: {e}")
        else:
            overlay_skeleton_matplotlib(fig, skel.nodes, skel.edges)
            out_png = outdir / "skeleton_matplotlib.png"
            try:
                fig.savefig(str(out_png), dpi=150)
                print(f"Wrote visualization: {out_png}")
            except Exception as e:
                print(f"Failed to write matplotlib PNG: {e}")

    out_swc = outdir / "skeleton.swc"
    try:
        skel.write_swc(str(out_swc), break_cycles="mst", annotate=True)
        print(f"Wrote SWC: {out_swc}")
    except Exception as e:
        print(f"Failed to write SWC: {e}")

    try:
        Vt, Ft = thin_mesh(m, mcf_iters=15, guidance_type=guidance_type)
        mt = tm.Trimesh(vertices=Vt, faces=Ft, process=False)
        mm_thin = MeshManager(mt, verbose=False)
        fig2 = mm_thin.visualize_mesh_3d(title="Meso-skeleton surface", backend=args.backend)
        if fig2 is not None:
            if args.backend in ("auto", "plotly") and fig2.__class__.__module__.startswith("plotly"):
                out_html2 = outdir / "thinned_plotly.html"
                fig2.write_html(str(out_html2))
                print(f"Wrote visualization: {out_html2}")
            else:
                out_png2 = outdir / "thinned_matplotlib.png"
                fig2.savefig(str(out_png2), dpi=150)
                print(f"Wrote visualization: {out_png2}")
    except Exception as e:
        print(f"Thin-mesh visualization failed: {e}")


if __name__ == "__main__":
    main()
