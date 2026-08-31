# Robustness

## Input

- Prefer closed, manifold, watertight meshes
- Use `load_and_repair` or `MeshManager.repair_mesh` when needed
- No built-in coordinate normalization — skeletons remain in the input frame
- `min_edge_length` scales with bbox diagonal, so large-coordinate meshes work
  without hand-tuning a time step

See [Meshes and validation](../guide/meshes.md).

## Pole gating and ray backends

Exterior-pole gating needs a point-in-mesh test. `trimesh` may switch
`mesh.contains` to Embree when `embreex` is installed; **Embree is float32**.
On meshes far from the origin that flips many containment answers.

`pymcfs` therefore uses an exact float64 traverser by default
(`points_inside_mesh`). Opt into Embree only with `fast_gating=True` and
`pymcfs[embree]` on unit-ish meshes near the origin.

Gating is typically the dominant cost per iteration (often ~60–80% of wall
clock after other speedups). Prefer the default exact path unless you have
measured a safe Embree case.

## Remesh growth abort

Aggressive medial weights can create many obtuse faces; splits then outpace
collapses and vertex count explodes. The driver stops when

`n > max_vertex_growth × n0` (default `4.0`).

Successful runs on complex tubular meshes often reach ~2×; catastrophic blow-ups
are 10–100×. If you hit the abort, lower `medial_weight` / medial ratio or use
`profile="auto"` / [parameter search](../guide/search.md).

## CHOLMOD conditioning

Pinned tips use huge `attraction_weight`, so `AᵀA` looks nearly singular. The
solve usually still succeeds; treat the warning as expected. Install
`pymcfs[cholmod]` on Linux/WSL for faster SPD solves; set `use_cholmod=False`
to force SuperLU.

## Performance tips

Same discrete algorithm as the reference MCFS pipeline; practical speedups:

- Carry pole validity through remesh by index (only new split poles are re-tested)
- Numba collapse / topology hashing and direct `AᵀA` assembly
- Optional CHOLMOD for the SPD solve
- Avoid `fast_gating=True` on large-coordinate meshes (wrong answers beat speed)

For interactive debugging, use `verbose=True` or a standard-library `logging`
handler on the `pymcfs` loggers rather than guessing iteration cost.
