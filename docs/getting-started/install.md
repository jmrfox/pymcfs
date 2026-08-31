# Install

## Library install

From a checkout of this repository:

```bash
pip install .
pip install ".[cholmod,viz]"   # recommended extras
```

Or with [uv](https://docs.astral.sh/uv/) in the repo:

```bash
uv sync                     # core
uv sync --extra cholmod     # recommended on Linux/WSL
uv sync --extra viz         # plotly / matplotlib
uv sync --extra embree      # optional fast_gating
```

Runtime dependencies: `numpy`, `scipy`, `trimesh`, `networkx`, `numba`, `rtree`.

## Recommended: CHOLMOD

Faster SPD solves for the contraction system (Linux/WSL):

```bash
sudo apt install libsuitesparse-dev
pip install ".[cholmod]"
# or: uv sync --extra cholmod
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
| `cholmod` | `pip install ".[cholmod]"` | SuiteSparse CHOLMOD via scikit-sparse |
| `viz` | `pip install ".[viz]"` | plotly / matplotlib for plot helpers |
| `embree` | `pip install ".[embree]"` | Fast pole gating with `fast_gating=True` |

!!! warning "Embree / float32"
    Embree traces in single precision. Only enable `fast_gating=True` for
    unit-ish meshes near the origin. Large-coordinate meshes (e.g. µm-scale
    surfaces far from the origin) need the default exact float64 gating path.

## Contributors

```bash
uv sync --group docs    # mkdocs + API autodoc
uv sync --group dev     # pytest and related tooling
```
