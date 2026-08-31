# Parameter search

When a single profile or proposal is close but not quite right,
`parameter_search` / `search_mcfs_params` try a **small set of nearby**
`(attraction_weight, medial_weight)` values and refine-phase settings, then
return the best skeleton.

Cost is roughly **~4×** a normal contraction (capped by `max_search_contracts`,
default 4). Prefer this over hand-tuning when you want a one-shot improvement
on a new mesh class.

## Convenience: flag on skeletonize

```python
from pymcfs import skeletonize

skel = skeletonize(mesh, profile="auto", parameter_search=True)
```

Same public kwargs as a normal call; search uses the resolved base weights as
the center of the trial grid.

## Full result

```python
from pymcfs import search_mcfs_params

result = search_mcfs_params(mesh, profile="auto", extend_tips=True)
print(result.attraction_weight, result.medial_weight)
print(result.score.summary())
print(result.n_contracts, result.n_converts)
skel = result.skeleton
```

`McfsSearchResult` also carries `gate_exterior_poles`, winning refine knobs
(`keep_hub_branches`, `tip_extend_scale`), and optional per-trial summaries
when `return_trials=True`.

## What it tries

1. Weight candidates around the base `(attraction, medial)` (clamped to safe
   bands), up to `max_search_contracts` contractions.
2. For each contracted meso-surface, a few refine variants (hub keep count /
   tip-extend scale).
3. Ranking starts from `score_skeleton`, then applies light penalties for deep
   tips and excess leaf arms at thick hubs.

## Search vs resample

| | `parameter_search` | `resample` |
|--|--------------------|------------|
| Re-contracts mesh | yes | no |
| Changes weights | yes (nearby trials) | no |
| Changes connectivity | via better contraction / refine | only node spacing |
| Cost | ~4× | cheap post-process |

```python
# Search for better weights (expensive)
skel = skeletonize(mesh, profile="auto", parameter_search=True)

# Only change curve density after a good skeleton (cheap)
skel = skeletonize(mesh, profile="auto", resample="uniform")
```

See [Refine and quality](quality.md) for scoring details and
[API: Search](../api/search.md) for signatures.
