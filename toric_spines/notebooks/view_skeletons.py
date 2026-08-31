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
# # View skeleton results vs input mesh
#
# Load a previously computed curve skeleton from `outputs/polylines/` and
# overlay it on the matching `toric_spines/data/mesh/TS*.obj` surface.
#
# Typical sources (from `toric_spines/scripts/batch_ts_skeletonize.py` or `ts_skeleton`):
#
# - `skeleton_sparse.cg` — batch sparse-oracle result
# - `skeleton_raw.cg` / `skeleton_uniform.cg` — notebook exports
#
# Change `CASE` / `SKELETON` below, then re-run.

# %%
from __future__ import annotations

from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import trimesh as tm

from pymcfs.cg_io import graph_from_cg, read_cg
from pymcfs.quality import analyze_skeleton
from pymcfs.skeleton import Skeleton

ROOT = Path("../..").resolve() if Path.cwd().name == "notebooks" else Path.cwd()
DATA = ROOT / "toric_spines" / "data" / "mesh"
OUT = ROOT / "outputs" / "polylines"

# --- choose what to view ---
CASE = "TS1"  # e.g. "TS2", "TS3", "TS76"
SKELETON = "skeleton_sparse"  # stem without extension; prefers .cg over .polylines.txt
SHOW_NODES = True
MESH_OPACITY = 0.15
RUN_ANALYZE = True  # recompute analyze_skeleton (uses mesh.contains)

print(f"ROOT={ROOT}")
print(f"DATA={DATA}")
print(f"OUT={OUT}")


# %% [markdown]
# ## 0. Available cases

# %%
def get_file_stem(path: Path) -> str:
    return path.name.replace(".polylines.txt", "").removesuffix(".cg")

cases: list[dict] = []
for case_dir in sorted(OUT.glob("TS*")):
    if not case_dir.is_dir():
        continue
    skels = sorted(
        {get_file_stem(p) for p in case_dir.glob("skeleton*")}
    )
    mesh_path = DATA / f"{case_dir.name}.obj"
    cases.append(
        {
            "case": case_dir.name,
            "mesh": mesh_path.is_file(),
            "skeletons": skels,
            "quality": (case_dir / "quality.txt").is_file(),
        }
    )

if not cases:
    print(f"No results under {OUT}. Run toric_spines/scripts/batch_ts_skeletonize.py first.")
else:
    for c in cases:
        mark = " <-- selected" if c["case"] == CASE else ""
        print(
            f"  {c['case']}: mesh={c['mesh']} quality={c['quality']} "
            f"skeletons={c['skeletons']}{mark}"
        )

# %% [markdown]
# ## 1. Load mesh + skeleton

# %%
def _skeleton_from_cg(path: Path) -> Skeleton:
    nodes, edges = read_cg(path)
    G = graph_from_cg(path)
    return Skeleton(nodes=nodes, edges=edges, graph=G)


def _load_skeleton(case_dir: Path, stem: str) -> tuple[Skeleton, Path]:
    cg = case_dir / f"{stem}.cg"
    if cg.is_file():
        return _skeleton_from_cg(cg), cg
    raise FileNotFoundError(
        f"Missing {cg}. Available: {[p.name for p in case_dir.glob('skeleton*')]}"
    )


def _summarize(skel: Skeleton) -> None:
    G = skel.graph
    deg = Counter(dict(G.degree()).values())
    n_cc = nx.number_connected_components(G) if G.number_of_nodes() else 0
    cyclomatic = (
        int(G.number_of_edges() - G.number_of_nodes() + n_cc) if G.number_of_nodes() else 0
    )
    print(
        f"skeleton: nodes={skel.nodes.shape[0]} edges={skel.edges.shape[0]} "
        f"cc={n_cc} cyclomatic={cyclomatic}"
    )
    print(f"  degree histogram: {dict(sorted(deg.items()))}")
    print(
        f"  leaves={deg.get(1, 0)} junctions={sum(c for d, c in deg.items() if d >= 3)}"
    )


mesh_path = DATA / f"{CASE}.obj"
case_dir = OUT / CASE
if not mesh_path.is_file():
    raise FileNotFoundError(f"Mesh not found: {mesh_path}")
if not case_dir.is_dir():
    raise FileNotFoundError(f"Result dir not found: {case_dir}")

mesh = tm.load(str(mesh_path), force="mesh", process=False)
skel, skel_path = _load_skeleton(case_dir, SKELETON)
print(f"mesh: {mesh_path.name} V={len(mesh.vertices)} F={len(mesh.faces)}")
print(f"loaded: {skel_path.relative_to(ROOT)}")
_summarize(skel)

quality_path = case_dir / "quality.txt"
if quality_path.is_file():
    print("\n--- quality.txt ---")
    print(quality_path.read_text(encoding="utf-8").rstrip())

# %% [markdown]
# ## 2. Overlay: input mesh + skeleton

# %%
fig = skel.plot_3d(
    mesh,
    show_nodes=SHOW_NODES,
    node_size=1,
    mesh_opacity=MESH_OPACITY,
    title=f"{CASE} — {SKELETON}",
    autoshow=False,
)
fig

# %% [markdown]
# ## 3. Optional quality recompute
#
# Re-runs `analyze_skeleton` against the current mesh (may differ slightly from
# a stored `quality.txt` if the mesh or prune settings changed).

# %%
if RUN_ANALYZE:
    report = analyze_skeleton(mesh, skel)
    print(report.summary())
else:
    print("RUN_ANALYZE=False — skipped")

# %% [markdown]
# ## 4. Gallery — all cases with a matching skeleton stem
#
# One figure per available case (same `SKELETON` stem). Useful after a batch run.

# %%
from IPython.display import display

SHOW_GALLERY = True
GALLERY_OPACITY = 0.12

if SHOW_GALLERY:
    n_shown = 0
    for c in cases:
        stem_dir = OUT / c["case"]
        cg = stem_dir / f"{SKELETON}.cg"
        mpath = DATA / f"{c['case']}.obj"
        if not cg.is_file() or not mpath.is_file():
            continue
        m = tm.load(str(mpath), force="mesh", process=False)
        s = _skeleton_from_cg(cg)
        f = s.plot_3d(
            m,
            show_nodes=False,
            mesh_opacity=GALLERY_OPACITY,
            title=f"{c['case']} — {SKELETON}",
            autoshow=False,
        )
        print(f"gallery: {c['case']} nodes={s.nodes.shape[0]} edges={s.edges.shape[0]}")
        display(f)
        n_shown += 1
    if n_shown == 0:
        print(f"No {SKELETON}.cg files found under {OUT}")
else:
    print("SHOW_GALLERY=False — skipped")

# %% [markdown]
# Tip: set `CASE` / `SKELETON` at the top and re-run from cell 1 for a single
# focused comparison. Batch summary CSV (if present):
# `outputs/polylines/batch_summary.csv`.
