# Quality and sweeps

## Refine phase

After MCFS contraction, `skeletonize` / `convert_to_skeleton` apply an optional
**refine phase** (in order):

1. **Exterior prune** — dangling tips outside the mesh (`prune_exterior`)
2. **Short-leaf prune** — mild micro-spurs with `L < short_leaf_scale × r_junc`
   (`prune_short_leaves`; not the primary volume-star fix)
3. **Thick-hub prune** — at thick high-degree hubs, keep only the longest
   `keep_hub_branches` leaf arms (`prune_thick_hubs`; default on)
4. **Tip extension** — grow unfinished tips toward lobe ends (`extend_tips`;
   default off; max travel = `tip_extend_scale × bbox_diag`)
5. Optional **resample** — curve density only (`resample=`; see below)

| Knob | Default | Notes |
|------|---------|-------|
| `prune_thick_hubs` | `True` | Volume-star prune at thick hubs |
| `keep_hub_branches` | `2` | Leaf arms kept per thick hub |
| `hub_degree_min` | `4` | Ignores ordinary Y-junctions |
| `hub_radius_frac` | `0.015` | Min `|sd|/diag` for “thick” |
| `extend_tips` | `False` | Useful when tips should reach surface end-caps |
| `tip_extend_scale` | `1.0` | Travel budget vs bbox diagonal |
| `tip_clearance_frac` | `0.01` | Stop near surface |
| `tip_cone_deg` | `40` | Direction-search cone |

Example with tip extension (application-specific; e.g. unfinished tubular tips):

```python
skel = skeletonize(
    mesh,
    profile="auto",
    branching="sparse",
    extend_tips=True,
    tip_extend_scale=1.0,
)
```

**Refine phase** = prune / extend / resample. **`resample`** controls curve
density only (`resample=True` / `"uniform"` / `"compress"`); it does not change
connectivity beyond node spacing.

## Analyze a skeleton

```python
from pymcfs import analyze_skeleton, score_skeleton

report = analyze_skeleton(mesh, skel)
print(report.summary())
# nodes, junctions, leaves, nodes/edges inside, genus vs cyclomatic, …

score = score_skeleton(report)
print(score.summary())
```

`score_skeleton` ranks candidates for parameter selection:

1. Hard reject — remesh growth, non-finite verts, area overshoot
2. Topology — prefer `cyclomatic == mesh_genus` (penalize missing *and* excess cycles)
3. Containment — nodes/edges inside; exterior nodes heavily penalized
4. Compactness — fewer junctions, then fewer nodes/leaves

## Parameter search vs resample

**`parameter_search`** re-contracts the mesh with a few nearby
`(attraction_weight, medial_weight)` values and tries a small set of refine-phase
settings, then returns the best skeleton (~4× contraction cost). **`resample`**
only post-processes an already extracted curve (uniform spacing / compress); it
does not change contraction weights.

```python
from pymcfs import skeletonize, search_mcfs_params

# Convenience: same API, returns Skeleton
skel = skeletonize(mesh, profile="auto", parameter_search=True)

# Full result (weights, score, trial count)
result = search_mcfs_params(mesh, profile="auto", extend_tips=True)
print(result.attraction_weight, result.medial_weight, result.score.summary(), result.n_contracts)
skel = result.skeleton
```

Search ranking starts from `score_skeleton`, then applies light penalties for
deep tips and excess leaf arms at thick hubs. For a dense grid over weights,
use the sweep script below instead.

## Parameter sweep script

Research sweeps live on **`main`** under `toric_spines/` (not on `release`).

```bash
git checkout main
uv run python toric_spines/scripts/sweep_mcfs_params.py --mesh ts2
uv run python toric_spines/scripts/sweep_mcfs_params.py --mesh ts2 --mesh ts1 --top-k 3
```

Sweeps `(attraction_weight, ratio)` with
`medial_weight = ratio · attraction_weight`, early-aborts remesh blow-ups,
writes CSV + top polylines under `outputs/sweeps/`.
