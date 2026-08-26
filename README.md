Mean-curvature flow skeletonization (MCFS) of 3D surface meshes in Python.

Inspired by CGAL Triangulated Surface Mesh Skeletonization and "Mean Curvature Skeletons" (Tagliasacchi 2012).

**Defaults** (CGAL-app / complex meshes): `w_H=0.5`, `w_M=5.0`, and
`gate_exterior_poles=True` (medial pull only for Voronoi poles inside the mesh).
Use `skeletonize(..., profile="starlab")` for Starlab parity weights without gating.

## Optional CHOLMOD (faster SPD solve)

MCFS factors the normal equations `AᵀA` each iteration. Install SuiteSparse CHOLMOD
via scikit-sparse for a large speedup on bigger meshes (Linux/WSL: `apt install
libsuitesparse-dev` then `uv sync --extra cholmod`). Set `use_cholmod=False` on the
driver to force SciPy SuperLU.

```bash
uv sync --extra cholmod
```

## Starlab parity

Stage-wise comparison against Starlab `mcfskel` dumps lives under [`fixtures/parity/`](fixtures/parity/README.md).
Parity dumps use the Starlab profile (`w_H=0.1`, `w_M=0.2`, ungated poles).

```bash
uv run python scripts/dump_pymcfs_parity.py --case sindorelax --iters 1,final
uv run python scripts/compare_starlab_parity.py --case sindorelax
uv run pytest tests/test_parity.py
```

