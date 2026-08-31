# Quick start

```python
from pymcfs import load_and_repair, skeletonize, analyze_skeleton

mesh = load_and_repair("mesh.obj")  # load + repair + validate
skel = skeletonize(mesh)            # robust defaults
print(analyze_skeleton(mesh, skel).summary())

skel.write_polylines("skeleton.polylines.txt")
skel.write_cg("skeleton.cg")
```

If the mesh is already closed and watertight, you can skip repair:

```python
import trimesh as tm
from pymcfs import skeletonize

mesh = tm.load("mesh.obj", force="mesh", process=False)
skel = skeletonize(mesh)
```

## Common variants

```python
# Mesh-conditioned weights (sparse branching by default)
skel = skeletonize(mesh, profile="auto")

# Try a few nearby weights + refine settings (~4× contraction cost)
skel = skeletonize(mesh, profile="auto", parameter_search=True)

# Starlab parity weights (ungated poles)
skel = skeletonize(mesh, profile="starlab")

# Meso-surface only (no curve conversion)
from pymcfs import contract_mesh
V, F = contract_mesh(mesh, max_iterations=100)

# Step-through driver
from pymcfs import MeanCurvatureFlowSkeletonization
driver = MeanCurvatureFlowSkeletonization(mesh, verbose=True)
driver.contract_until_convergence()
skel = driver.convert_to_skeleton(resample=False)
```

## Input requirements

- Closed, manifold triangle mesh (watertight preferred)
- Exactly one connected component
- Coordinates stay in the **input frame** (no built-in normalize/rescale)
- Default remesh length is `0.002 × bbox_diagonal`

See [Meshes and validation](../guide/meshes.md).

## Next steps

- [Profiles and weights](../guide/profiles.md)
- [Parameter search](../guide/search.md)
- [Export and I/O](../guide/export.md)
- [API reference](../api/index.md)
