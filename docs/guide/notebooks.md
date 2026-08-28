# Notebooks

Notebooks live under `notebooks/` as **Jupytext** percent scripts (reviewable in git).

```bash
# Edit the .py source
uv run jupytext --sync notebooks/ts_skeleton.py

# Dev stack (jupyter + viz)
uv sync --group dev
```

## `ts_skeleton`

Interactive MCFS on one `data/mesh/TS*.obj`:

- Set `TS_NAME`, weights (`W_H` / `W_M`), or `USE_ORACLE = True`
- `ORACLE_BRANCHING = "sparse"` | `"balanced"` | `"dense"`
- Writes under `outputs/polylines/<case>/` (gitignored)

## Outputs

Keep generated meshes, polylines, and sweeps out of git (`outputs/` is ignored).
