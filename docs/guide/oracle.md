# Parameter proposals

`propose_mcfs_params(mesh)` suggests `(attraction_weight, medial_weight)` from
mesh features. Used by `profile="auto"`.

## Features

Two thickness signals (both normalized by bbox diagonal):

- `ρ` = mean Voronoi-pole offset / bbox diagonal
- `r_char` = characteristic radius / bbox diagonal

The proposal scale is `max(ρ/ρ_ref, r_char/r_ref)` with refs from a slender-tube
band (`ρ ≈ 1.6%`, `r_char ≈ 2.5%`). Using only mean `ρ` mis-classifies meshes
with thin processes but a bulky compartment: mean `ρ` stays near-ref while
`r_char` is elevated, and robust `(0.5, 5)` then fills that volume with spurious
junctions.

Thick / compact fragments (high `ρ`) need a **lower medial ratio** or remesh
growth blows up under robust weights.

## Branching preference

Default is **`branching="sparse"`** — prefer fewer junctions and no spurious
volume branches.

| Mode | Near slender-tube band (both signals) | Thick (high `ρ` or `r_char`) |
|------|---------------------------------------|------------------------------|
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

## Next: automatic search

If a single proposal is still off, try a small set of nearby weights and refine
settings with [`parameter_search=True`](search.md) or
[`search_mcfs_params`](search.md) (~4× contraction cost). For scoring and
refine-phase knobs, see [Refine and quality](quality.md).
