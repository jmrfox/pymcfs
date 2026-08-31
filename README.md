# pymcfs

[![Documentation](https://readthedocs.org/projects/pymcfs/badge/?version=latest)](https://pymcfs.readthedocs.io/en/latest/)

Mean-curvature flow skeletonization (MCFS) of closed 3D triangle meshes in Python.
Inspired by CGAL Triangulated Surface Mesh Skeletonization and Tagliasacchi et al.
(SGP 2012). Import and call — there is no CLI.

**Version 0.1.0** — early preview for collaborators; API is mostly stable but not
yet tagged v1.0.

**Branches:** `main` is the full tree (including research/dev). The public
`release` branch is auto-synced from `main` without those trees.

**Docs:** [pymcfs.readthedocs.io](https://pymcfs.readthedocs.io/en/latest/)
(or locally: `uv sync --group docs && uv run mkdocs serve`).

## Install

```bash
pip install .
pip install ".[cholmod,viz]"   # recommended extras

# or with uv in this repo:
uv sync                     # core
uv sync --extra cholmod     # recommended on Linux/WSL
uv sync --extra viz         # plotly / matplotlib
uv sync --group docs        # mkdocs + API autodoc
```

System dep for CHOLMOD: `sudo apt install libsuitesparse-dev`.

## Quick start

```python
from pymcfs import load_and_repair, skeletonize

mesh = load_and_repair("mesh.obj")
skel = skeletonize(mesh)                    # robust defaults
# skel = skeletonize(mesh, profile="auto")  # mesh-conditioned weights
skel.write_polylines("skeleton.polylines.txt")
```

Input meshes should be closed and watertight. Defaults already run a refine phase
(prune exterior tips, short leaves, and thick hubs). Use `profile="auto"` to
propose contraction weights from mesh shape.

## Profiles

| Profile | `attraction_weight` | `medial_weight` | Gate poles | Use |
|---------|---------------------|-----------------|------------|-----|
| `robust` (default) | 0.5 | 5.0 | yes | Complex tubular meshes |
| `starlab` | 0.1 | 0.2 | no | Starlab parity |
| `auto` | from mesh | from mesh | yes | Proposed (`branching="sparse"` by default) |

`attraction_weight` resists moving vertices from their current positions;
`medial_weight` pulls toward interior centerline targets (medial poles).

Coordinates are **not** normalized. See the docs for parameter proposals, quality
scoring, CHOLMOD warnings, and Embree caveats.

## Tests

```bash
uv run pytest -m "not e2e" -q
uv run pytest -m e2e -q
uv run mkdocs build --strict
```

## Documentation site

```bash
uv sync --group docs
uv run mkdocs serve    # http://127.0.0.1:8000
uv run mkdocs build    # site/ (gitignored)
```
