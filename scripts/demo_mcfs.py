#!/usr/bin/env python3
"""
Demo script for pymcfs: load a processed mesh, run MCF-based skeletonization,
visualize with MeshManager utilities, and export an SWC.

Usage:
  python scripts/demo_mcfs.py [--mesh PATH] [--outdir PATH] [--backend plotly|matplotlib]

If --mesh is not provided, the script searches data/mesh/processed/ for a mesh file.
"""
from __future__ import annotations

import argparse
import os
import sys
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh as tm

from pymcfs.mesh import MeshManager
from pymcfs.skeleton import skeletonize, thin_mesh


def find_default_mesh() -> Optional[str]:
    # root = Path(__file__).resolve().parents[1]
    # search_root = root / "data" / "mesh" / "processed"
    # exts = ("*.obj", "*.ply", "*.stl", "*.off", "*.glb", "*.gltf")
    # for pat in exts:
    #     files = sorted(search_root.glob(pat))
    #     if files:
    #         return str(files[0])
    # return None
    return "C:/Users/MainUser/Documents/GitHub/pymcfs/data/mesh/processed/TS2_wrapped.obj"


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
    # Edge segments
    xs, ys, zs = [], [], []
    for u, v in edges:
        p, q = nodes[int(u)], nodes[int(v)]
        xs += [p[0], q[0], None]
        ys += [p[1], q[1], None]
        zs += [p[2], q[2], None]
    fig.add_trace(
        go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="crimson", width=4), name=f"{name_prefix} Edges")
    )
    fig.add_trace(
        go.Scatter3d(
            x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode="markers", marker=dict(size=3, color="black"), name=f"{name_prefix} Nodes"
        )
    )


def overlay_skeleton_matplotlib(fig, nodes: np.ndarray, edges: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception:
        return
    if nodes.size == 0 or edges.size == 0:
        return
    if not fig.axes:
        return
    ax = fig.axes[0]
    for u, v in edges:
        p, q = nodes[int(u)], nodes[int(v)]
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="crimson", linewidth=2)


def main():
    ap = argparse.ArgumentParser(description="pymcfs demo: skeletonization + visualization + SWC export")
    ap.add_argument("--mesh", type=str, default=None, help="Path to input mesh. If omitted, use first file in data/mesh/processed/")
    ap.add_argument("--outdir", type=str, default="outputs/demo", help="Directory to write outputs")
    ap.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "plotly", "matplotlib"],
        help="Visualization backend for MeshManager",
    )
    ap.add_argument("--guidance", type=str, default="voronoi", choices=["none", "voronoi"], help="MCF guidance type")
    ap.add_argument("--compress", action="store_true", help="Compress degree-2 chains in skeleton")
    ap.add_argument("--resample", type=float, default=0.0, help="Uniform edge resampling spacing (0 disables)")
    args = ap.parse_args()

    mesh_path = args.mesh or find_default_mesh()
    if mesh_path is None:
        print("No mesh specified and none found under data/mesh/processed/. Please pass --mesh PATH.")
        sys.exit(1)

    outdir = ensure_outdir(args.outdir)

    # Load mesh via MeshManager
    mm = MeshManager(verbose=True)
    m = mm.load_mesh(mesh_path)
    print(f"Loaded mesh: {len(m.vertices)} vertices, {len(m.faces)} faces")

    # Visualize input mesh
    fig = mm.visualize_mesh_3d(title=f"Input Mesh: {Path(mesh_path).name}", backend=args.backend)

    # Run MCF + skeletonization
    guidance_type = None if args.guidance == "none" else "voronoi"

    skel = skeletonize(
        m,
        mcf_dt=2e-2,
        mcf_iters=30,
        laplacian_type="cotangent",
        guidance_type=guidance_type,
        guidance_weight=1.0 if guidance_type == "voronoi" else 0.0,
        build_graph="mesh",
        knn=12,
        length_quantile=0.7,
        collapse_passes=2,
        collapse_mode="pq_heap",
        collapse_ratio=0.2,
        collapse_domain="graph",
        compress_chains=bool(args.compress),
        resample_spacing=(args.resample if args.resample > 0 else None),
        medial_protect=True if guidance_type == "voronoi" else False,
        medial_protect_threshold=0.5,
        closest_pole_policy=True if guidance_type == "voronoi" else False,
        verbose=True,
    )

    print(f"Skeleton: {skel.nodes.shape[0]} nodes, {skel.edges.shape[0]} edges")

    # Overlay skeleton on mesh visualization and save
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

    # Export SWC
    out_swc = outdir / "skeleton.swc"
    try:
        skel.write_swc(str(out_swc), break_cycles="mst", annotate=True)
        print(f"Wrote SWC: {out_swc}")
    except Exception as e:
        print(f"Failed to write SWC: {e}")

    # Optional: show thinned surface mesh
    try:
        Vt, Ft = thin_mesh(m, mcf_dt=1e-2, mcf_iters=10, collapse_passes=1, collapse_mode="pq", collapse_ratio=0.2)
        mt = tm.Trimesh(vertices=Vt, faces=Ft, process=False)
        mm_thin = MeshManager(mt, verbose=False)
        fig2 = mm_thin.visualize_mesh_3d(title="Thinned Surface", backend=args.backend)
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
