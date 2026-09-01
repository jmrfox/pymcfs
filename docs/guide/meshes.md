# Meshes and validation

MCFS expects a **closed, manifold triangle mesh** with a single connected
component. Coordinates are **not** normalized — the skeleton stays in the
input frame. Remesh thresholds (e.g. `min_edge_length`) scale with the
bounding-box diagonal, so large-coordinate meshes work without hand-tuning.

## Requirements

| Check | Default |
|-------|---------|
| Triangular faces `(m, 3)`, vertices `(n, 3)` | required |
| Non-empty | required |
| Exactly one connected component | required |
| Watertight (no boundary edges) | required when `validate=True` |

`skeletonize` / the driver run validation by default (`validate=True`).

## Validate and repair

```python
from pymcfs import validate_mcfs_mesh, load_and_repair

# Soft-validate → repair → re-validate (path or Trimesh)
mesh = load_and_repair("mesh.obj")

# Or check a mesh you already hold
validate_mcfs_mesh(mesh)  # raises ValueError on failure
```

`load_and_repair` loads with `process=False`, validates, and **only repairs
when needed**. Already-watertight MCFS inputs are returned unchanged.
`repair_mesh` never runs proximity vertex welding (`Trimesh.process` /
`merge_vertices`), which can create non-manifold edges at pinch/contact
seams. Prefer fixing the source mesh when possible.

## MeshManager (optional)

```python
from pymcfs import MeshManager, example_mesh

mgr = MeshManager()
mgr.load_mesh("mesh.obj", validate_mcfs=False)
mesh = mgr.repair_mesh()
validate_mcfs_mesh(mesh)

# Synthetic closed shape for smoke tests
cyl = example_mesh("cylinder")
```

Most application code only needs `trimesh` + `load_and_repair` /
`skeletonize`. Treat `MeshManager` as optional utilities.

## Coordinate frame

There is no built-in normalize → skeletonize → rescale path. Pass meshes in
the units your application already uses; exported polylines and `.cg` files
match that frame.
