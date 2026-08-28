# pymcfs — path to v1.0

Goal: a small, installable library for MCFS skeletonization of closed triangle
meshes (especially TS biological surfaces). Users import and call; no CLI.

Version is `0.1.0` in `pyproject.toml`. Tag `v1.0.0` when the remaining
correctness gates below are signed off.

```bash
uv sync --extra cholmod --group dev
uv run pytest -m "not e2e" -q
```

---

## 1. Dependency diet — done

Runtime: `numpy`, `scipy`, `trimesh`, `networkx`, `numba`, `rtree`.

Extras / groups:

```bash
uv sync                          # core
uv sync --extra cholmod          # faster solves (recommended on Linux/WSL)
uv sync --extra viz              # plotly / matplotlib
uv sync --extra embree           # optional fast_gating
uv sync --group dev              # pytest, jupytext, notebooks, viz libs
```

Removed unused `pymeshlab`. `pytest` / Jupyter / viz are no longer default deps.

---

## 2. API freeze — done (keep stable)

Public surface: `skeletonize`, `thin_mesh`, `MeanCurvatureFlowSkeletonization`,
`Skeleton`, `propose_mcfs_params` / `profile="auto"`, `analyze_skeleton` /
`score_skeleton`, `validate_mcfs_mesh`, `MeshManager` (optional utilities).

- [x] Thread `use_cholmod` through `skeletonize` / `thin_mesh`
- [x] Soft-import plotly/matplotlib via `pymcfs.viz` (`pymcfs[viz]`)
- [x] Keep `MeshManager` public but document as optional
- [x] `__version__` matches `pyproject.toml` (`0.1.0`)

Do **not** expand the API (no CLI, no GPU, no silent normalization).

---

## 3. Correctness gates — remaining before tag

- [ ] Re-run `notebooks/ts_skeleton.py` on TS1 + TS2 (oracle or documented weights)
- [x] E2E regression (`uv run pytest -m e2e -q`) — passed
- [ ] Spot-check `analyze_skeleton` on each TS mesh you ship examples for
- [x] Parity Stage 1 green (`tests/test_parity.py`)
- [x] Remesh-growth abort + oracle on TS2 confirmed in earlier sweeps
      (robust `0.5/5` aborts; oracle / low ratio works)

---

## 4. Docs — MkDocs site

```bash
uv sync --group docs
uv run mkdocs serve
uv run mkdocs build --strict
```

Pages under `docs/`; config in `mkdocs.yml`. README stays a short landing page.

- [x] MkDocs Material + mkdocstrings API pages
- [x] Split algorithm / guide / getting-started from the old README monolith
- [x] CI builds docs with `--strict`
- [x] Read the Docs (`.readthedocs.yaml`, same uv/`docs` pattern as MASCAF)
      (URL: https://pymcfs.readthedocs.io/en/latest/)
---

## 5. Packaging / release

- [x] Lean `pyproject.toml`, version `0.1.0`
- [x] Wheel contains only `pymcfs/`; sdist excludes `data/`, `notebooks/`, fixtures
- [x] CI workflow: `.github/workflows/ci.yml` (`pytest -m "not e2e"`)
- [ ] Tag `v1.0.0` after §3 gates
- [ ] Optional: publish to PyPI

---

## 6. Explicit non-goals (v1)

- Built-in normalize → skeletonize → rescale
- Incremental `AᵀA` refactor / iterative CG solvers
- JAX / GPU
- Windows-native CHOLMOD (use WSL)
- Changing Laplacian weight `w_L` (stays 1)
- Per-vertex `w_M` from unused Voronoi pole weights (follow-up)

---

## Quick commands

```bash
uv sync --extra cholmod --group dev
uv run pytest -m "not e2e" -q
uv run pytest -m e2e -q
uv run python scripts/bench_mcfs_iter.py --mesh ts1 --iters 5 --profile
uv run python scripts/sweep_mcfs_params.py --mesh ts2
uv run python scripts/compare_starlab_parity.py --case sindorelax
```
