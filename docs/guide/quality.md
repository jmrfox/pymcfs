# Quality and sweeps

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

## Parameter sweep script

```bash
uv run python scripts/sweep_mcfs_params.py --mesh ts2
uv run python scripts/sweep_mcfs_params.py --mesh ts2 --mesh ts1 --top-k 3
```

Sweeps `(w_H, ratio)` with `w_M = ratio · w_H`, early-aborts remesh blow-ups,
writes CSV + top polylines under `outputs/sweeps/`.
