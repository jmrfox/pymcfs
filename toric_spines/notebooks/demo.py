# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # pymcfs demo: detailed MCFS run
#
# Walk through mean-curvature-flow skeletonization for one closed triangle mesh:
#
# 1. load / validate the input
# 2. inspect Voronoi medial poles
# 3. contract with the Starlab-style MCFS driver (geometry + remesh + pinning)
# 4. convert the meso-skeleton surface to a curve graph
# 5. optional curve resample (density); refine phase also includes prune/extend
# 6. quality check and export
#
# Change `MESH_PATH` (or use `example_mesh`) below to try another shape.

# %%
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import trimesh as tm

from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.medial import compute_voronoi_poles
from pymcfs.mesh import MeshManager, example_mesh
from pymcfs.quality import analyze_skeleton
from pymcfs.skeleton import Skeleton, resample_skeleton, skeletonize
from pymcfs.validate import validate_mcfs_mesh

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pymcfs.demo")

ROOT = Path("../..").resolve() if Path.cwd().name == "notebooks" else Path.cwd()
DATA = ROOT / "toric_spines" / "data" / "mesh"
OUT = ROOT / "outputs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

# Pick one mesh: a file under toric_spines/data/mesh/, or set MESH_PATH = None and use EXAMPLE.
MESH_PATH: Path | None = DATA / "cylinder.obj"
EXAMPLE: str | None = None  # e.g. "cylinder" / "torus" when MESH_PATH is None

# MCFS weights (Starlab / CGAL defaults)
ATTRACTION_WEIGHT = 0.1   # attraction to current positions
MEDIAL_WEIGHT = 0.2       # attraction to Voronoi poles (0 disables medial term)
MAX_ITERS = 500
TIMEOUT_S = 120.0
SNAPSHOT_EVERY = 25  # save meso-surface every N contraction iterations

print(f"ROOT={ROOT}")
print(f"OUT={OUT}")

# %% [markdown]
# ## 1. Load and validate

# %%
if MESH_PATH is not None:
    mesh_path = Path(MESH_PATH)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mm = MeshManager(verbose=True)
    mesh = mm.load_mesh(str(mesh_path), validate_mcfs=True)
    mesh_name = mesh_path.stem
    print(f"Loaded {mesh_path}")
else:
    kind = EXAMPLE or "cylinder"
    mesh = example_mesh(kind)
    mm = MeshManager(mesh, verbose=True)
    mesh_name = kind
    print(f"Built example_mesh({kind!r})")

validate_mcfs_mesh(mesh)  # raises if not a closed watertight triangle mesh
mm.print_mesh_analysis()
print(
    f"n={len(mesh.vertices)} f={len(mesh.faces)} "
    f"watertight={mesh.is_watertight} volume={float(mesh.volume):.4g}"
)

# %%
fig_input = mm.visualize_mesh_3d(title=f"Input: {mesh_name}", backend="plotly")
fig_input

# %% [markdown]
# ## 2. Voronoi medial poles
#
# Each vertex gets an inner Voronoi pole used by the `medial_weight` term. MCFS stores
# these on the driver; here we compute them once for visualization.

# %%
poles, pole_weights = compute_voronoi_poles(mesh)
offsets = np.linalg.norm(poles - np.asarray(mesh.vertices), axis=1)
print(
    f"poles: shape={poles.shape} weight_range=[{pole_weights.min():.3g}, {pole_weights.max():.3g}] "
    f"offset mean={offsets.mean():.4g} max={offsets.max():.4g}"
)

# Subsample pole segments so the plot stays readable.
rng = np.random.default_rng(0)
n_show = min(200, len(mesh.vertices))
idx = rng.choice(len(mesh.vertices), size=n_show, replace=False)
V = np.asarray(mesh.vertices, dtype=float)
F = np.asarray(mesh.faces, dtype=int)

xs, ys, zs = [], [], []
for i in idx:
    p, q = V[i], poles[i]
    xs += [p[0], q[0], None]
    ys += [p[1], q[1], None]
    zs += [p[2], q[2], None]

fig_poles = go.Figure(
    data=[
        go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            color="#AAAAAA", opacity=0.25, name="mesh", flatshading=True,
        ),
        go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="#d62728", width=2), name="pole rays",
        ),
        go.Scatter3d(
            x=poles[idx, 0], y=poles[idx, 1], z=poles[idx, 2],
            mode="markers",
            marker=dict(size=2, color="#ff7f0e"), name="poles",
        ),
    ]
)
fig_poles.update_layout(title=f"Voronoi poles ({n_show}/{len(V)} shown)", scene_aspectmode="data")
fig_poles

# %% [markdown]
# ## 3. MCFS contraction (step by step)
#
# Each `contract()` iteration is:
#
# `contract_geometry → collapse_edges → split_faces → detect_degeneracies`
#
# We run the loop ourselves so we can snapshot area / topology along the way.
# (`contract_until_convergence()` does the same stopping criteria internally.)

# %%
driver = MeanCurvatureFlowSkeletonization(
    mesh,
    attraction_weight=ATTRACTION_WEIGHT,
    medial_weight=MEDIAL_WEIGHT,
    max_iterations=MAX_ITERS,
    timeout_seconds=TIMEOUT_S,
    verbose=True,
    log=log,
)

history: list[dict] = []
prev_area = float(driver._surface_area())
area0 = float(driver._area0)
print(
    f"start: n={driver.V.shape[0]} f={driver.F.shape[0]} "
    f"area0={area0:.4g} min_edge={driver._min_edge:.4g} "
    f"attraction_weight={ATTRACTION_WEIGHT} medial_weight={MEDIAL_WEIGHT}"
)

if driver.timeout_seconds is not None and driver.timeout_seconds > 0:
    import time as _time
    driver._deadline = _time.monotonic() + float(driver.timeout_seconds)
else:
    driver._deadline = None

last_it = 0
for it in range(1, int(driver.max_iterations) + 1):
    last_it = it
    driver._iter = it
    if driver._timed_out():
        print(f"timeout at iter {it - 1}")
        break

    driver.contract()
    area = float(driver._surface_area())
    record = {
        "iter": it,
        "n": int(driver.V.shape[0]),
        "f": int(driver.F.shape[0]),
        "area": area,
        "d_area": abs(prev_area - area),
        "fixed": int(driver.fixed.sum()),
        "split": int(driver.is_split.sum()),
    }
    history.append(record)

    if it == 1 or it % SNAPSHOT_EVERY == 0:
        print(
            f"iter {it:4d}: n={record['n']:5d} f={record['f']:5d} "
            f"area={area:.4g} dA={record['d_area']:.4g} "
            f"fixed={record['fixed']} split={record['split']}"
        )

    if prev_area > 0 and abs(prev_area - area) < driver.area_variation_factor * max(area0, 1e-30):
        print(f"converged at iter {it} area={area:.4g}")
        break
    prev_area = area
    if driver.F.shape[0] == 0:
        break

print(f"finished after {last_it} iterations; meso n={driver.V.shape[0]} f={driver.F.shape[0]}")

# %%
iters = [h["iter"] for h in history]
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=iters, y=[h["area"] for h in history], name="area"))
fig_hist.add_trace(go.Scatter(x=iters, y=[h["n"] for h in history], name="#vertices", yaxis="y2"))
fig_hist.add_trace(go.Scatter(x=iters, y=[h["fixed"] for h in history], name="#fixed", yaxis="y2"))
fig_hist.update_layout(
    title="Contraction progress",
    xaxis_title="iteration",
    yaxis=dict(title="surface area"),
    yaxis2=dict(title="count", overlaying="y", side="right"),
    legend=dict(orientation="h"),
)
fig_hist

# %% [markdown]
# ## 4. Meso-skeleton surface
#
# After convergence the mesh is a thin “sheet” around the medial axis (still a surface).

# %%
meso = driver.meso_skeleton_mesh()
print(f"meso-skeleton: n={len(meso.vertices)} f={len(meso.faces)}")
mm_meso = MeshManager(meso, verbose=False)
fig_meso = mm_meso.visualize_mesh_3d(title="Meso-skeleton surface", backend="plotly")
fig_meso

# %% [markdown]
# ## 5. Convert to curve skeleton (raw)
#
# Starlab collapses remaining face-bearing edges in length-priority order. Survivors
# sit at the centroid of collapsed meso-skeleton vertices.
#
# Refinement is **off** here so you see the dense, often irregular raw curve graph.

# %%
def _summarize_skeleton(skel: Skeleton, label: str) -> dict:
    G = skel.graph
    deg = Counter(dict(G.degree()).values())
    n_cc = nx.number_connected_components(G) if G.number_of_nodes() else 0
    cyclomatic = (
        int(G.number_of_edges() - G.number_of_nodes() + n_cc) if G.number_of_nodes() else 0
    )
    lengths = (
        np.array([float(d.get("weight", 0.0)) for _, _, d in G.edges(data=True)], dtype=float)
        if G.number_of_edges()
        else np.zeros(0)
    )
    info = {
        "label": label,
        "nodes": skel.nodes.shape[0],
        "edges": skel.edges.shape[0],
        "components": n_cc,
        "cyclomatic": cyclomatic,
        "degree": dict(sorted(deg.items())),
        "leaves": deg.get(1, 0),
        "junctions": sum(c for d, c in deg.items() if d >= 3),
        "edge_len_mean": float(lengths.mean()) if lengths.size else 0.0,
        "edge_len_std": float(lengths.std()) if lengths.size else 0.0,
        "edge_len_min": float(lengths.min()) if lengths.size else 0.0,
        "edge_len_max": float(lengths.max()) if lengths.size else 0.0,
        "lengths": lengths,
    }
    print(
        f"{label}: nodes={info['nodes']} edges={info['edges']} "
        f"cc={n_cc} cyclomatic={cyclomatic}"
    )
    print(f"  degree histogram: {info['degree']}")
    print(
        f"  leaves={info['leaves']} junctions={info['junctions']} | "
        f"edge length mean={info['edge_len_mean']:.4g} std={info['edge_len_std']:.4g} "
        f"[{info['edge_len_min']:.4g}, {info['edge_len_max']:.4g}]"
    )
    return info


skel_raw = driver.convert_to_skeleton(resample=False)
raw_info = _summarize_skeleton(skel_raw, "raw")

# %%
fig_raw = skel_raw.plot_3d(
    mesh,
    show_nodes=True,
    node_size=3,
    mesh_opacity=0.15,
    title=f"Raw skeleton: {mesh_name}",
    autoshow=False,
)
fig_raw

# %% [markdown]
# ## 6. Optional resample
#
# MCFS often produces more sample points than needed, with uneven spacing along the
# medial axis. **Resample** is the curve-density step (`resample=False` by default on
# conversion / `skeletonize`). The broader **refine phase** also includes prune and
# extend; this section only shows resampling:
#
# | option | behavior |
# |---|---|
# | `resample=True` / `"uniform"` | arc-length resample chains between junctions/leaves |
# | `resample_spacing=...` | absolute target segment length |
# | `resample_spacing_frac=...` | spacing as a fraction of skeleton bbox diagonal |
# | `resample="compress"` | keep only junctions and leaves (drop all degree-2 nodes) |
#
# Default uniform spacing is `2 × median edge length` (mild downsample + evening).
# Junctions/leaves stay fixed; curvature along chains is preserved.

# %%
# Mild default resample (also available as skeletonize(..., resample=True)).
skel_uniform = resample_skeleton(skel_raw, mode="uniform")
uniform_info = _summarize_skeleton(skel_uniform, "uniform (default spacing)")

# Explicit spacing relative to bbox diagonal (~2% → coarser).
skel_frac = resample_skeleton(skel_raw, mode="uniform", spacing_frac=0.02)
frac_info = _summarize_skeleton(skel_frac, "uniform (spacing_frac=0.02)")

# Junction-only compress (aggressive).
skel_compress = resample_skeleton(skel_raw, mode="compress")
compress_info = _summarize_skeleton(skel_compress, "compress")

# Use the default uniform resample for the rest of the notebook.
skel = skel_uniform

# %%
# Edge-length histograms: raw vs resampled.
fig_len = go.Figure()
for info, color in (
    (raw_info, "#1f77b4"),
    (uniform_info, "#2ca02c"),
    (frac_info, "#ff7f0e"),
):
    if info["lengths"].size == 0:
        continue
    fig_len.add_trace(
        go.Histogram(
            x=info["lengths"],
            name=info["label"],
            opacity=0.55,
            marker_color=color,
            nbinsx=30,
        )
    )
fig_len.update_layout(
    barmode="overlay",
    title="Edge length distribution: raw vs resampled",
    xaxis_title="edge length",
    yaxis_title="count",
    legend=dict(orientation="h"),
)
fig_len

# %%
def _overlay_skeletons(
    mesh: tm.Trimesh,
    skels: list[tuple[Skeleton, str, str]],
    *,
    title: str,
) -> go.Figure:
    """Plot several skeletons on the same mesh (different colors)."""
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    traces: list = [
        go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=F[:, 0], j=F[:, 1], k=F[:, 2],
            color="#CCCCCC", opacity=0.12, name="mesh", flatshading=True,
        )
    ]
    for sk, name, color in skels:
        P = np.asarray(sk.nodes, dtype=float)
        E = np.asarray(sk.edges, dtype=int)
        xs, ys, zs = [], [], []
        for a, b in E:
            pa, pb = P[int(a)], P[int(b)]
            xs += [float(pa[0]), float(pb[0]), None]
            ys += [float(pa[1]), float(pb[1]), None]
            zs += [float(pa[2]), float(pb[2]), None]
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=color, width=5), name=f"{name} edges",
            )
        )
        traces.append(
            go.Scatter3d(
                x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                marker=dict(size=3, color=color), name=f"{name} nodes",
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(title=title, scene_aspectmode="data", legend=dict(orientation="h"))
    return fig


fig_compare = _overlay_skeletons(
    mesh,
    [
        (skel_raw, "raw", "#1f77b4"),
        (skel_uniform, "uniform", "#2ca02c"),
    ],
    title=f"Raw vs uniform resample: {mesh_name}",
)
fig_compare

# %%
fig_compress = skel_compress.plot_3d(
    mesh,
    show_nodes=True,
    node_size=5,
    edge_color="#d62728",
    mesh_opacity=0.15,
    title=f"compress mode (junctions/leaves only): {mesh_name}",
    autoshow=False,
)
fig_compress

# %% [markdown]
# ## 7. Quality report and export
#
# Exports use the default **uniform** resampled skeleton from above.

# %%
report = analyze_skeleton(mesh, skel)
print(report.summary())

out_dir = OUT / mesh_name
out_dir.mkdir(parents=True, exist_ok=True)
skel_raw.write_polylines(str(out_dir / "skeleton_raw.polylines.txt"))
skel.write_polylines(str(out_dir / "skeleton.polylines.txt"))
meso.export(str(out_dir / "meso_skeleton.obj"))
fig_compare.write_html(str(out_dir / "skeleton_resample_compare.html"))
print(f"wrote outputs under {out_dir}")

# %% [markdown]
# ## 8. One-liner equivalent
#
# `skeletonize(..., resample=True)` runs contraction + conversion + default uniform
# resample in one call.

# %%
skel_quick = skeletonize(
    mesh,
    attraction_weight=ATTRACTION_WEIGHT,
    medial_weight=MEDIAL_WEIGHT,
    max_iterations=MAX_ITERS,
    timeout_seconds=TIMEOUT_S,
    resample=True,
    verbose=False,
)
print(
    f"skeletonize(resample=True): nodes={skel_quick.nodes.shape[0]} edges={skel_quick.edges.shape[0]} "
    f"(notebook uniform: {skel.nodes.shape[0]} / {skel.edges.shape[0]}; "
    f"raw: {skel_raw.nodes.shape[0]} / {skel_raw.edges.shape[0]})"
)
skel_quick.plot_3d(
    mesh,
    show_nodes=True,
    title=f"skeletonize(resample=True): {mesh_name}",
    autoshow=False,
)
