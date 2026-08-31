# pymcfs

Mean-curvature flow skeletonization (MCFS) of closed 3D triangle meshes in Python.
Inspired by [CGAL Triangulated Surface Mesh Skeletonization](https://doc.cgal.org/latest/Surface_mesh_skeletonization/)
and Tagliasacchi et al. (SGP 2012).

Import and call — there is no CLI.

```python
import trimesh as tm
from pymcfs import skeletonize

mesh = tm.load("mesh.obj", force="mesh", process=False)
skel = skeletonize(mesh)
skel.write_polylines("skeleton.polylines.txt")
```

## What it does

1. Contract the surface with a weighted mean-curvature solve (optional Voronoi-pole medial term).
2. Remesh locally (collapse short edges, split obtuse faces).
3. Pin formed branch tips.
4. Convert the thin meso-surface into a 1D curve skeleton.
5. Optionally refine (prune / tip extension / resample) and score the result.

## Where to go

- [Install](getting-started/install.md) — core, CHOLMOD, viz, Embree
- [Quick start](getting-started/quickstart.md) — first skeleton in a few lines
- [Meshes and validation](guide/meshes.md) — input requirements and repair
- [Profiles and weights](guide/profiles.md) — `robust` / `starlab` / `auto`
- [Parameter proposals](guide/oracle.md) — mesh-conditioned weights
- [Refine and quality](guide/quality.md) — prune, score, analyze
- [Parameter search](guide/search.md) — automatic nearby-weight trials
- [Export and I/O](guide/export.md) — polylines, `.cg`, plotting
- [Algorithm](algorithm/index.md) — how MCFS is implemented here
- [API reference](api/index.md) — public functions and classes

## Design notes

- Coordinates are **not** normalized; skeletons stay in the input frame.
- Default profile (`robust`) targets complex tubular surfaces with pole gating.
- `propose_mcfs_params` / `profile="auto"` defaults to `branching="sparse"` (fewer junctions).
- Exact float64 point-in-mesh gating is the default; Embree is opt-in only.
