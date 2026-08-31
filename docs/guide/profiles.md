# Profiles and weights

MCFS solves a stacked least-squares system each iteration with three soft weights:

| Symbol | Code | Role |
|--------|------|------|
| `ω_L` | fixed `1` | Laplacian (mean-curvature) term |
| `ω_H` | `attraction_weight` | Attraction to current vertex positions |
| `ω_P` | `medial_weight` | Attraction to Voronoi poles |

Absolute scale vs `ω_L=1` matters: `(0.5, 5)` and `(0.1, 1)` share ratio 10 but
behave differently.

## Built-in profiles

| Profile | `attraction_weight` | `medial_weight` | Gate exterior poles | Use |
|---------|---------------------|-----------------|---------------------|-----|
| `robust` (default) | 0.5 | 5.0 | yes | Complex tubular meshes |
| `starlab` | 0.1 | 0.2 | no | Parity with Starlab dumps |
| `auto` | from mesh | from mesh | yes | Mesh-conditioned proposal |

```python
skeletonize(mesh)                       # robust
skeletonize(mesh, profile="starlab")
skeletonize(mesh, profile="auto")
skeletonize(mesh, profile="auto", branching="dense")
```

## Pole gating

With `gate_exterior_poles=True` (default for `robust` / `auto`), medial weights
apply only when the Voronoi pole lies **inside** the input mesh. Exterior poles
get `medial_weight = 0` so they cannot pull branches outside the surface.

Even with gating, a few meso vertices can still drift slightly outside during
contraction. By default `:func:`~pymcfs.skeletonize`` /
`:meth:`~pymcfs.MeanCurvatureFlowSkeletonization.convert_to_skeleton`` prune
**dangling exterior tips** (`prune_exterior=True`) so those leaks do not become
long through-surface leaf branches. Set `prune_exterior=False` to keep the raw
curve graph.

## Remesh and pinning

- `min_edge_length` defaults to `0.002 × bbox_diagonal`
- Contraction aborts if vertex count exceeds `max_vertex_growth × n0` (default `4.0`)
- Pinned branch tips use `attraction_weight = 1/pinned_attraction_floor` (very large)

See [Robustness](../algorithm/robustness.md) and
[parameter proposals](oracle.md).
