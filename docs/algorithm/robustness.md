# Robustness

## Input

- Prefer closed, manifold, watertight meshes
- Use `MeshManager.repair_mesh` when needed
- No built-in coordinate normalization — skeletons remain in the input frame
- `min_edge_length` scales with bbox diagonal, so large-coordinate meshes work
  without hand-tuning a time step

## Pole gating and ray backends

Exterior-pole gating needs a point-in-mesh test. `trimesh` may switch
`mesh.contains` to Embree when `embreex` is installed; **Embree is float32**.
On meshes far from the origin that flips many containment answers.

`pymcfs` therefore uses an exact float64 traverser by default
(`points_inside_mesh`). Opt into Embree only with `fast_gating=True` and
`pymcfs[embree]` on unit-ish meshes near the origin.

Gating is typically 60–80% of an iteration after other speedups — see
[Benchmarks](../benchmarks.md).

## Remesh growth abort

Aggressive medial weights can create many obtuse faces; splits then outpace
collapses and vertex count explodes. The driver stops when

`n > max_vertex_growth × n0` (default `4.0`).

Successful runs on complex tubular meshes often reach ~2×; catastrophic blow-ups
are 10–100×.

## CHOLMOD conditioning

Pinned tips use huge `attraction_weight`, so `AᵀA` looks nearly singular. The
solve usually still succeeds; treat the warning as expected.

## Performance (same discrete algorithm)

Pole validity is carried through remesh by index (only new split poles are
re-tested); Numba collapse / topology hashing; direct `AᵀA` assembly; optional
CHOLMOD. Net ~7× per iteration on TS1 and ~27× on sindorelax versus the
pre-fix baseline — details in [Benchmarks](../benchmarks.md).
