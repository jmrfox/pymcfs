# Research notebooks

Application notebooks and TS workflows live on the **`main`** branch under
`toric_spines/notebooks/` (not on this public `release` branch).

```bash
git checkout main
uv run jupytext --sync toric_spines/notebooks/ts_skeleton.py
```

On `release`, use the package API directly from Python or the [quick start](../getting-started/quickstart.md).
