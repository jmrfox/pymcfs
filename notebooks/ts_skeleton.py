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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TS mesh MCFS skeletonization
#
# Skeletonize one `TS*.obj` mesh from `data/mesh/` and inspect intermediate stages:
#
# 1. choose / load / validate (repair if needed)
# 2. Voronoi medial poles
# 3. MCFS contraction with progress curves + meso snapshots
# 4. raw curve skeleton + topology stats
# 5. optional refinement
# 6. quality report and export
#
# Change `TS_NAME` below (e.g. `"TS2"`, `"TS24"`).

# %%
from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import trimesh as tm

from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.medial import compute_voronoi_poles
from pymcfs.mesh import MeshManager
from pymcfs.quality import analyze_skeleton
from pymcfs.skeleton import Skeleton, refine_skeleton
from pymcfs.validate import validate_mcfs_mesh

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pymcfs.ts_skeleton")

ROOT = Path("..").resolve() if Path.cwd().name == "notebooks" else Path.cwd()
DATA = ROOT / "data" / "mesh"
OUT = ROOT / "outputs" / "polylines"
OUT.mkdir(parents=True, exist_ok=True)

# --- choose mesh ---
TS_NAME = "TS3" 
REPAIR_IF_NEEDED = True

# MCFS weights: robust defaults (0.5 / 5.0), or set USE_ORACLE=True for
# mesh-conditioned proposals (see pymcfs.params.propose_mcfs_params).
# branching="sparse" (default) snaps TS1-like meshes to robust and prefers
# fewer junctions on thick fragments — best for neuroscience centerlines.
USE_ORACLE = True
ORACLE_BRANCHING = "sparse"  # "sparse" | "balanced" | "dense"
W_H = 0.5
W_M = 5.0
GATE_EXTERIOR_POLES = True  # CGAL: w_M only when pole is inside the mesh
MAX_ITERS = 500
TIMEOUT_S = 180.0
SNAPSHOT_EVERY = 10  # print + keep meso mesh every N iters (and always iter 1)

print(f"ROOT={ROOT}")
print(f"DATA={DATA}")
print(f"OUT={OUT}")

# %% [markdown]
# ## 0. Available TS meshes

# %%
ts_files = sorted(DATA.glob("TS*.obj"))
if not ts_files:
    raise FileNotFoundError(f"No TS*.obj under {DATA}")

rows = []
for p in ts_files:
    try:
        m = tm.load(str(p), force="mesh", process=False)
        rows.append(
            {
                "name": p.stem,
                "path": p.name,
                "V": len(m.vertices),
                "F": len(m.faces),
                "watertight": bool(m.is_watertight),
                "size_kb": round(p.stat().st_size / 1024, 1),
            }
        )
    except Exception as e:
        rows.append({"name": p.stem, "path": p.name, "error": str(e)})

for r in rows:
    if "error" in r:
        print(f"  {r['name']}: ERROR {r['error']}")
    else:
        mark = " <-- selected" if r["name"] == TS_NAME else ""
        print(
            f"  {r['name']}: V={r['V']} F={r['F']} "
            f"watertight={r['watertight']} ({r['size_kb']} KB){mark}"
        )

# %% [markdown]
# ## 1. Load and validate

# %%
mesh_path = DATA / f"{TS_NAME}.obj"
if not mesh_path.is_file():
    raise FileNotFoundError(
        f"Mesh not found: {mesh_path}\nAvailable: {[p.stem for p in ts_files]}"
    )

mm = MeshManager(verbose=True)
mesh = mm.load_mesh(str(mesh_path), validate_mcfs=False)
mesh_name = mesh_path.stem

try:
    validate_mcfs_mesh(mesh)
    print("validate_mcfs_mesh: OK")
except ValueError as e:
    print(f"validate_mcfs_mesh: FAILED ({e})")
    if not REPAIR_IF_NEEDED:
        raise
    print("Attempting MeshManager.repair_mesh() …")
    mesh = mm.repair_mesh(mesh)
    mm = MeshManager(mesh, verbose=True)
    validate_mcfs_mesh(mesh)
    print("validate_mcfs_mesh: OK after repair")

mm.print_mesh_analysis()
V0 = np.asarray(mesh.vertices, dtype=float)
F0 = np.asarray(mesh.faces, dtype=int)
diag0 = float(np.linalg.norm(V0.max(0) - V0.min(0)))
print(
    f"n={len(V0)} f={len(F0)} watertight={mesh.is_watertight} "
    f"area={float(mesh.area):.4g} volume={float(mesh.volume):.4g} bbox_diag={diag0:.4g}"
)

case_out = OUT / mesh_name
case_out.mkdir(parents=True, exist_ok=True)

# %%
fig_input = mm.visualize_mesh_3d(title=f"Input: {mesh_name}", backend="plotly")
fig_input

# %% [markdown]
# ## 2. Voronoi medial poles
#
# Inner Voronoi poles feed the `w_M` attraction term (set `W_M=0` to disable).

# %%
poles, pole_weights = compute_voronoi_poles(mesh)
offsets = np.linalg.norm(poles - V0, axis=1)
print(
    f"poles: shape={poles.shape} "
    f"weight_range=[{pole_weights.min():.3g}, {pole_weights.max():.3g}] "
    f"offset mean={offsets.mean():.4g} max={offsets.max():.4g} "
    f"({100 * offsets.mean() / max(diag0, 1e-30):.2f}% of bbox diag)"
)

rng = np.random.default_rng(0)
n_show = min(250, len(V0))
idx = rng.choice(len(V0), size=n_show, replace=False)

xs, ys, zs = [], [], []
for i in idx:
    p, q = V0[i], poles[i]
    xs += [p[0], q[0], None]
    ys += [p[1], q[1], None]
    zs += [p[2], q[2], None]

fig_poles = go.Figure(
    data=[
        go.Mesh3d(
            x=V0[:, 0],
            y=V0[:, 1],
            z=V0[:, 2],
            i=F0[:, 0],
            j=F0[:, 1],
            k=F0[:, 2],
            color="#AAAAAA",
            opacity=0.22,
            name="mesh",
            flatshading=True,
        ),
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#d62728", width=2),
            name="pole rays",
        ),
        go.Scatter3d(
            x=poles[idx, 0],
            y=poles[idx, 1],
            z=poles[idx, 2],
            mode="markers",
            marker=dict(size=2, color="#ff7f0e"),
            name="poles",
        ),
    ]
)
fig_poles.update_layout(
    title=f"Voronoi poles ({n_show}/{len(V0)} shown)",
    scene_aspectmode="data",
)
fig_poles

# %%
fig_off = go.Figure(
    data=[
        go.Histogram(x=offsets, nbinsx=40, marker_color="#ff7f0e", name="pole offset"),
    ]
)
fig_off.update_layout(
    title="Pole offset distribution (distance vertex → pole)",
    xaxis_title="offset",
    yaxis_title="count",
)
fig_off

# %% [markdown]
# ## 3. MCFS contraction (step by step)
#
# Each `contract()` is:
#
# `contract_geometry → collapse_edges → split_faces → detect_degeneracies`
#
# Exterior Voronoi poles get `w_M = 0` when `GATE_EXTERIOR_POLES` is True
# (CGAL `Side_of_triangle_mesh` / `ON_BOUNDED_SIDE`). That prevents medial
# attraction from pulling branches outside the surface on complex TS meshes.
#
# We drive the loop manually to record area / vertex count / pinning and keep
# occasional meso snapshots (N=1 is especially useful for Starlab parity).

# %%
if USE_ORACLE:
    from pymcfs import propose_mcfs_params

    _proposed = propose_mcfs_params(mesh, branching=ORACLE_BRANCHING)
    W_H, W_M = float(_proposed.w_H), float(_proposed.w_M)
    GATE_EXTERIOR_POLES = bool(_proposed.gate_exterior_poles)
    print(f"oracle: {_proposed.summary()}")

driver = MeanCurvatureFlowSkeletonization(
    mesh,
    w_H=W_H,
    w_M=W_M,
    gate_exterior_poles=GATE_EXTERIOR_POLES,
    max_iterations=MAX_ITERS,
    timeout_seconds=TIMEOUT_S,
    verbose=True,
    log=log,
)

history: list[dict] = []
meso_snapshots: dict[int, tm.Trimesh] = {}
prev_area = float(driver._surface_area())
area0 = float(driver._area0)
n_poles_valid = int(driver.pole_valid.sum()) if driver.pole_valid.size else 0
print(
    f"start: n={driver.V.shape[0]} f={driver.F.shape[0]} "
    f"area0={area0:.4g} min_edge={driver._min_edge:.4g} "
    f"w_H={W_H} w_M={W_M} gate_poles={GATE_EXTERIOR_POLES} "
    f"poles_valid={n_poles_valid}/{driver.V.shape[0]}"
)

if driver.timeout_seconds is not None and driver.timeout_seconds > 0:
    driver._deadline = time.monotonic() + float(driver.timeout_seconds)
else:
    driver._deadline = None

t0 = time.perf_counter()
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
        "area_frac": area / max(area0, 1e-30),
        "d_area": abs(prev_area - area),
        "fixed": int(driver.fixed.sum()),
        "split": int(driver.is_split.sum()),
    }
    history.append(record)

    if it == 1 or it % SNAPSHOT_EVERY == 0:
        meso_snapshots[it] = driver.meso_skeleton_mesh()
        n_gated = (
            int((~driver.pole_valid).sum())
            if GATE_EXTERIOR_POLES and driver.pole_valid.shape[0] == driver.V.shape[0]
            else 0
        )
        print(
            f"iter {it:4d}: n={record['n']:5d} f={record['f']:5d} "
            f"area={area:.4g} ({100 * record['area_frac']:.2f}% of area0) "
            f"dA={record['d_area']:.4g} fixed={record['fixed']} split={record['split']} "
            f"poles_gated={n_gated}"
        )

    if prev_area > 0 and abs(prev_area - area) < driver.area_variation_factor * max(
        area0, 1e-30
    ):
        print(f"converged at iter {it} area={area:.4g}")
        break
    prev_area = area
    if driver.F.shape[0] == 0:
        break

elapsed = time.perf_counter() - t0
meso_snapshots[last_it] = driver.meso_skeleton_mesh()
print(
    f"finished after {last_it} iterations in {elapsed:.2f}s; "
    f"meso n={driver.V.shape[0]} f={driver.F.shape[0]}"
)

# %%
iters = [h["iter"] for h in history]
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=iters, y=[h["area"] for h in history], name="area"))
fig_hist.add_trace(
    go.Scatter(x=iters, y=[h["n"] for h in history], name="#vertices", yaxis="y2")
)
fig_hist.add_trace(
    go.Scatter(x=iters, y=[h["f"] for h in history], name="#faces", yaxis="y2")
)
fig_hist.add_trace(
    go.Scatter(x=iters, y=[h["fixed"] for h in history], name="#fixed", yaxis="y2")
)
fig_hist.update_layout(
    title=f"Contraction progress — {mesh_name}",
    xaxis_title="iteration",
    yaxis=dict(title="surface area"),
    yaxis2=dict(title="count", overlaying="y", side="right"),
    legend=dict(orientation="h"),
)
fig_hist

# %% [markdown]
# ## 4. Meso-skeleton surface
#
# After convergence the mesh is a thin sheet around the medial axis (still a surface).
# Snapshot at N=1 is the best stage for Starlab parity checks.

# %%
meso = driver.meso_skeleton_mesh()
meso.export(str(case_out / "meso_final.obj"))
if 1 in meso_snapshots:
    meso_snapshots[1].export(str(case_out / "meso_N0001.obj"))
print(f"meso-skeleton: n={len(meso.vertices)} f={len(meso.faces)}")
print(f"wrote {case_out / 'meso_final.obj'}")

mm_meso = MeshManager(meso, verbose=False)
fig_meso = mm_meso.visualize_mesh_3d(
    title=f"Meso-skeleton (final) — {mesh_name}",
    backend="plotly",
)
fig_meso

# %%
# Overlay input (transparent) with final meso.
fig_meso_overlay = go.Figure(
    data=[
        go.Mesh3d(
            x=V0[:, 0],
            y=V0[:, 1],
            z=V0[:, 2],
            i=F0[:, 0],
            j=F0[:, 1],
            k=F0[:, 2],
            color="#BBBBBB",
            opacity=0.12,
            name="input",
            flatshading=True,
        ),
        go.Mesh3d(
            x=meso.vertices[:, 0],
            y=meso.vertices[:, 1],
            z=meso.vertices[:, 2],
            i=meso.faces[:, 0],
            j=meso.faces[:, 1],
            k=meso.faces[:, 2],
            color="#ff7f0e",
            opacity=0.55,
            name="meso",
            flatshading=True,
        ),
    ]
)
fig_meso_overlay.update_layout(
    title=f"Input vs meso — {mesh_name}",
    scene_aspectmode="data",
)
fig_meso_overlay

# %% [markdown]
# ## 5. Convert to curve skeleton (raw)
#
# Face-bearing edge collapse in length-priority order (Starlab-compatible).
# Refinement is **off** so you see the dense raw curve graph.

# %%
def _summarize_skeleton(skel: Skeleton, label: str) -> dict:
    G = skel.graph
    deg = Counter(dict(G.degree()).values())
    n_cc = nx.number_connected_components(G) if G.number_of_nodes() else 0
    cyclomatic = (
        int(G.number_of_edges() - G.number_of_nodes() + n_cc) if G.number_of_nodes() else 0
    )
    lengths = (
        np.array(
            [float(d.get("weight", 0.0)) for _, _, d in G.edges(data=True)],
            dtype=float,
        )
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
        "total_length": float(lengths.sum()) if lengths.size else 0.0,
        "lengths": lengths,
    }
    print(
        f"{label}: nodes={info['nodes']} edges={info['edges']} "
        f"cc={n_cc} cyclomatic={cyclomatic}"
    )
    print(f"  degree histogram: {info['degree']}")
    print(
        f"  leaves={info['leaves']} junctions={info['junctions']} | "
        f"total_len={info['total_length']:.4g} "
        f"edge mean={info['edge_len_mean']:.4g} std={info['edge_len_std']:.4g} "
        f"[{info['edge_len_min']:.4g}, {info['edge_len_max']:.4g}]"
    )
    return info


skel_raw = driver.convert_to_skeleton(refine=False)
raw_info = _summarize_skeleton(skel_raw, "raw")
skel_raw.write_cg(str(case_out / "skeleton_raw.cg"))
skel_raw.write_polylines(str(case_out / "skeleton_raw.polylines.txt"))

# %%
fig_raw = skel_raw.plot_3d(
    mesh,
    show_nodes=True,
    node_size=3,
    mesh_opacity=0.12,
    title=f"Raw skeleton — {mesh_name}",
    autoshow=False,
)
fig_raw

# %% [markdown]
# ## 6. Optional refinement
#
# Non-core post-step (`refine=False` by default on `skeletonize`). Useful for
# evenly spaced samples along chains; junctions/leaves stay fixed.

# %%
skel_uniform = refine_skeleton(skel_raw, mode="uniform")
uniform_info = _summarize_skeleton(skel_uniform, "uniform (default spacing)")

skel_frac = refine_skeleton(skel_raw, mode="uniform", spacing_frac=0.02)
frac_info = _summarize_skeleton(skel_frac, "uniform (spacing_frac=0.02)")

skel_compress = refine_skeleton(skel_raw, mode="compress")
compress_info = _summarize_skeleton(skel_compress, "compress")

skel = skel_uniform
skel.write_cg(str(case_out / "skeleton_uniform.cg"))
skel.write_polylines(str(case_out / "skeleton_uniform.polylines.txt"))

# %%
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
    title="Edge length distribution: raw vs refined",
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
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    traces: list = [
        go.Mesh3d(
            x=V[:, 0],
            y=V[:, 1],
            z=V[:, 2],
            i=F[:, 0],
            j=F[:, 1],
            k=F[:, 2],
            color="#CCCCCC",
            opacity=0.12,
            name="mesh",
            flatshading=True,
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
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=5),
                name=f"{name} edges",
            )
        )
        traces.append(
            go.Scatter3d(
                x=P[:, 0],
                y=P[:, 1],
                z=P[:, 2],
                mode="markers",
                marker=dict(size=3, color=color),
                name=f"{name} nodes",
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(title=title, scene_aspectmode="data", legend=dict(orientation="h"))
    return fig


fig_compare = _overlay_skeletons(
    mesh,
    [
        (skel_raw, "raw", "#1f77b4"),
        (skel_uniform, "uniform", "#d62728"),
        (skel_compress, "compress", "#2ca02c"),
    ],
    title=f"Skeleton refine compare — {mesh_name}",
)
fig_compare

# %% [markdown]
# ## 7. Quality analysis

# %%
report = analyze_skeleton(mesh, skel)
print(report.summary())

# Degree / component table for the chosen refined skeleton
G = skel.graph
print(
    f"\nchosen skeleton: nodes={skel.nodes.shape[0]} edges={skel.edges.shape[0]} "
    f"cc={nx.number_connected_components(G)}"
)
print(f"degree histogram: {dict(sorted(Counter(dict(G.degree()).values()).items()))}")

# %%
fig_final = skel.plot_3d(
    mesh,
    show_nodes=True,
    node_size=3,
    mesh_opacity=0.12,
    title=f"Final (uniform refine) — {mesh_name}",
    autoshow=False,
)
fig_final

# %% [markdown]
# ## 8. Export summary
#
# Outputs are under `outputs/ts_skeleton/<TS_NAME>/`.

# %%
exports = sorted(case_out.glob("*"))
print(f"Wrote {len(exports)} files to {case_out}:")
for p in exports:
    print(f"  {p.name:28s} {p.stat().st_size / 1024:8.1f} KB")

print(
    f"\nDone: {mesh_name} | iters={last_it} | "
    f"raw nodes={raw_info['nodes']} | uniform nodes={uniform_info['nodes']} | "
    f"leaves={uniform_info['leaves']} junctions={uniform_info['junctions']}"
)
