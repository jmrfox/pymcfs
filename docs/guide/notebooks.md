# Research notebooks (main branch)

Notebooks live under `toric_spines/notebooks/` as **Jupytext** percent scripts
(reviewable in git). This tree is on **`main` only** — not on `release`.

```bash
uv run jupytext --sync toric_spines/notebooks/ts_skeleton.py
```

## `ts_skeleton`

Interactive MCFS on one `toric_spines/data/mesh/TS*.obj` (application example):

```bash
uv run python toric_spines/scripts/batch_ts_skeletonize.py --mesh TS1
```

- Writes under `outputs/polylines/<case>/` (gitignored)

## `view_skeletons`

- Reads `.cg` under `outputs/polylines/<CASE>/`

## `demo`

Package walkthrough with optional mesh under `toric_spines/data/mesh/`.

Keep generated meshes, polylines, and sweeps out of git (`outputs/` is ignored).
