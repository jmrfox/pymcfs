# Quick start

```python
import trimesh as tm
from pymcfs import skeletonize, analyze_skeleton

mesh = tm.load("mesh.obj", force="mesh", process=False)
# Optional: MeshManager.repair_mesh() if the surface is not watertight

skel = skeletonize(mesh)  # robust defaults: attraction_weight=0.5, medial_weight=5.0, gated poles
print(analyze_skeleton(mesh, skel).summary())

skel.write_polylines("skeleton.polylines.txt")
```

## Common variants

```python
# Mesh-conditioned weights (sparse branching by default)
skel = skeletonize(mesh, profile="auto")

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
- Coordinates stay in the **input frame** (no built-in normalize/rescale)
- Default remesh length is `0.002 × bbox_diagonal`

## Export

```python
# One polyline per chain (junction-to-leaf / cycle)
skel.write_polylines("out.polylines.txt")

# Optional Plotly figure (needs pymcfs[viz])
fig = skel.plot_3d(mesh=mesh, autoshow=False)
```

See [Profiles and weights](../guide/profiles.md) and the [API reference](../api/index.md).
