# Parameter oracle

`propose_mcfs_params(mesh)` suggests `(w_H, w_M)` from mesh features. Used by
`profile="auto"`.

## Features

Two thickness signals (both normalized by bbox diagonal):

- `ρ` = mean Voronoi-pole offset / bbox diagonal
- `r_char` = characteristic radius / bbox diagonal

The oracle scale is `max(ρ/ρ_ref, r_char/r_ref)` with refs from the TS1 band
(`ρ ≈ 1.6%`, `r_char ≈ 2.5%`). Using only mean `ρ` mis-classifies meshes like
**TS3**: thin processes keep mean `ρ` near-ref while a bulky compartment has
elevated `r_char`, and robust `(0.5, 5)` then fills that volume with spurious
junctions.

Thick / compact fragments (e.g. TS2, high `ρ`) need a **lower medial ratio** or
remesh growth blows up under robust weights.

## Branching preference

Default is **`branching="sparse"`** — prefer fewer junctions and no spurious
volume branches (neuroscience priority).

| Mode | Near TS1 band (both signals) | Thick (high `ρ` or `r_char`) |
|------|------------------------------|------------------------------|
| `sparse` (default) | Exact robust `(0.5, 5.0)` | Lowest safe medial ratio |
| `balanced` | Exact robust `(0.5, 5.0)` | Geometry base ratio |
| `dense` | Slightly stronger medial | Higher ratio (still capped) |

```python
from pymcfs import propose_mcfs_params, skeletonize

params = propose_mcfs_params(mesh, branching="sparse")
print(params.summary())

skel = skeletonize(mesh, profile="auto", branching="sparse")
```

## Inspect features

```python
from pymcfs import mesh_mcfs_features

feats = mesh_mcfs_features(mesh)
print(feats.summary())
```

## Grid search

For systematic tuning on a mesh:

```bash
uv run python scripts/sweep_mcfs_params.py --mesh ts2 --mesh ts1
```

Results land under `outputs/sweeps/<name>/` (CSV + top-k polylines). Scoring
favors topology match, containment, then compactness — see
[Quality and sweeps](quality.md).
