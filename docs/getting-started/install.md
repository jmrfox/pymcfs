# Install

## Core library

```bash
uv sync
```

Runtime dependencies: `numpy`, `scipy`, `trimesh`, `networkx`, `numba`, `rtree`.

## Recommended: CHOLMOD

Faster SPD solves for the contraction system (Linux/WSL):

```bash
sudo apt install libsuitesparse-dev
uv sync --extra cholmod
```

Pass `use_cholmod=True|False|None` on `skeletonize` / `contract_mesh` / the driver
(`None` = use CHOLMOD when importable).

!!! note "Nearly singular warning"
    CHOLMOD may report `rcond ~ 1e-15`. Pinned vertices use
    `attraction_weight = 1/pinned_attraction_floor` (≈`1e7`), so `AᵀA`
    diagonals reach `1e14`. Expected conditioning noise, not a failed solve.

## Optional extras

| Extra | Install | Purpose |
|-------|---------|---------|
| `cholmod` | `uv sync --extra cholmod` | SuiteSparse CHOLMOD via scikit-sparse |
| `viz` | `uv sync --extra viz` | plotly / matplotlib for plot helpers |
| `embree` | `uv sync --extra embree` | Fast pole gating with `fast_gating=True` |

```bash
uv sync --group dev    # pytest, jupytext, notebooks, viz libs
```

!!! warning "Embree / float32"
    Embree traces in single precision. Only enable `fast_gating=True` for
    unit-ish meshes near the origin. Large-coordinate meshes (e.g. µm-scale
    neuron surfaces far from the origin) need the default exact float64
    gating path.

## From a wheel / editable

```bash
pip install .
pip install ".[cholmod,viz]"
```
