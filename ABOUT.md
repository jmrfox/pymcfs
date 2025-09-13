# pymcfs: Algorithms and Design

This document explains the main algorithms implemented in `pymcfs`, with an emphasis on the mean-curvature-flow (MCF) contraction and the skeletonization pipeline that follows. It is intended for readers comfortable with linear algebra and numerical methods, but not necessarily familiar with computational geometry.

The references anchor the design to well-known discrete differential geometry (DDG) operators and to the Mean Curvature Skeletons method (Tagliasacchi et al., SGP 2012).

## Contents

- Overview
- Discrete surface operators
  - Mesh notation
  - Cotangent Laplacian
  - Mean-value Laplacian
  - Lumped mass matrix
  - Properties and robustness ("secure" mode)
- Mean Curvature Flow (MCF)
  - PDE and time discretization
  - Linear system and guidance
  - Voronoi medial guidance
  - Stability tips and parameter choices
- Skeletonization pipeline
  - Step 1: Surface contraction (MCF)
  - Step 2: Node candidates and optional downsampling
  - Step 3: Graph construction (mesh-edge vs kNN)
  - Step 4: Priority edge-collapse graph thinning
  - Step 5: Pruning short leaves
  - Step 6: Compressing degree-2 chains
  - Step 7: Uniform resampling (optional)
  - Output and SWC export
- Surface thinning variant (mesh-edge collapses)
- Robustness and practical considerations
- Complexity and performance notes
- Relationship to the original MCF skeletonization (Starlab)
- API quick reference
- References

---

## Overview

At a high level, `pymcfs` turns a watertight triangle mesh into a 1D curve network (a "skeleton"). The process has two phases:

1) Contract the surface with mean curvature flow (MCF). This pulls the surface toward its medial axis while smoothing.

2) Collapse and prune connectivity on the contracted geometry to produce a sparse curve graph (the skeleton), optionally exporting to SWC.

Internally, we rely on standard discrete differential geometry operators assembled on triangle meshes:

- Discrete Laplacians: cotangent and mean-value variants
- Lumped mass matrix
- A stable implicit-Euler time stepping for MCF
- Optional soft positional guidance (uniform or per-vertex) to bias the contraction

## Discrete surface operators

### Mesh notation

Let `V` be an `(n, 3)` array of vertex positions and `F` an `(m, 3)` array of triangular faces (each row is a vertex index triple). Define `E` to be the set of undirected edges of the mesh.

### Cotangent Laplacian

The cotangent Laplacian is a standard discretization of the Laplace–Beltrami operator on triangle meshes. For an interior edge `(i, j)` shared by two triangles, the off-diagonal weight is

- `L[i, j] = -1/2 * (cot α + cot β)`

where `α` and `β` are the angles opposite edge `(i, j)` in the two incident triangles. Row sums are set to zero by defining the diagonal as

- `L[i, i] = -Σ_{j ≠ i} L[i, j]`.

In `pymcfs.laplacian.cotangent_laplacian(V, F)`, we compute cotangents from edge lengths using Heron's formula for the triangle areas. The result is a symmetric sparse matrix with zero row-sum.

Robustness option: `secure=True` clamps negative cotangents to zero. This mirrors a common practice in robust geometry processing where obtuse angles can generate negative weights that may cause artifacts on poor-quality meshes. With `secure=True` we still maintain symmetry and zero row-sum.

### Mean-value Laplacian

The mean-value Laplacian (Floater, 2003) uses weights of the form `w_ij = Σ_t tan(θ_t/2) / ||v_i - v_j||` across the two triangles sharing the edge. In practice it behaves similarly to cotangent but can be more numerically forgiving in certain configurations. We assemble a symmetric matrix and set the diagonal to enforce zero row-sum. See `pymcfs.laplacian.mean_value_laplacian(V, F)`.

### Lumped mass matrix

For implicit time-stepping, we use a diagonal ("lumped") mass matrix

- `M[i, i] = (1/3) * Σ_{faces incident to i} area(face)`.

See `pymcfs.laplacian.lumped_mass_matrix(V, F)`.

### Properties and robustness

- Symmetry: Both Laplacians are symmetric by construction.
- Zero row-sum: Implies that adding a constant vector field does not change Laplacian; global translation-invariance.
- Boundaries: All algorithms are intended for closed (watertight), manifold meshes. If your mesh has boundaries or degeneracies, consider repairing it first (see `pymcfs.mesh.MeshManager.repair_mesh`).
- Secure cotangent: `secure=True` clamps negative cot weights. This makes the operator more robust on meshes with obtuse triangles or skinny elements.

## Mean Curvature Flow (MCF)

### PDE and time discretization

We evolve vertex positions `V(t)` by the (discrete) mean-curvature flow:

- `dV/dt = -L V`

where `L` is the discrete Laplacian. We use implicit Euler time-stepping with step size `dt`:

- `V^{k+1} = V^{k} + dt * (-L V^{k+1})`.

Rearranging gives the linear system:

- `(M + dt * L) V^{k+1} = M V^{k}`

where `M` is the (lumped) mass matrix. This is solved at each iteration with sparse linear algebra. See `pymcfs.mcf.mean_curvature_flow`.

### Linear system with soft guidance

We optionally include soft positional guidance toward target positions `T` via a diagonal matrix `W ≥ 0`:

- `(M + dt * L + W) V^{k+1} = M V^{k} + W T`.

Interpretation:

- The `(M + dt*L)` term performs curvature-driven shrinkage in a stable, implicit fashion.
- The `W T` term softly pulls vertices toward targets `T`. When `W = w I` with scalar `w`, you get a uniform pull; when `W = diag(w_i)`, per-vertex strengths are possible.

Conveniences in `mean_curvature_flow`:

- `guidance_type="centroid"`: use the current centroid as `T` (a global pull).  
- `guidance_type="original"`: use the original `V^0` as `T` to resist over-contraction.  
- `guidance_targets`/`guidance_diag`: provide explicit `T` and per-vertex diagonal weights.
- `laplacian_type={"cotangent", "mean_value"}` and `laplacian_secure=True` for robustness.


### Voronoi medial guidance

`pymcfs.medial.compute_voronoi_poles(mesh)` returns per-vertex targets and weights derived from the (approximate) Voronoi poles (a standard construction approximating medial geometry).

- Targets `T`: points along the pole directions.  
- Weights `w_i ∈ [0, 1]`: higher near medial features, lower elsewhere.

Use as:

- `guidance_targets=targets`
- `guidance_diag=weights * scale`

This biases MCF toward medial structures, stabilizing skeletonization and helping avoid collapsing away important branches too quickly.


### Stability tips and parameter choices

- `dt`: Typical values are `1e-3` to `5e-2`. Larger `dt` contracts faster but risks feature loss.  
- `iterations`: 10–100 often suffice, depending on model scale and desired contraction.  
- Laplacian: Cotangent is a standard default; mean-value is a useful alternative if you observe artifacts on challenging meshes.  
- Guidance: `guidance_type="original"` often helps preserve shape while still shrinking. Voronoi guidance is powerful for emphasizing medial structures.


## Skeletonization pipeline

The top-level function `pymcfs.skeleton.skeletonize` implements a Skelcollapse-inspired workflow which operates partly on the surface and partly in a graph built from contracted points.

### Step 1: Surface contraction (MCF)

- Run `mean_curvature_flow` with chosen Laplacian, step size, iterations, and optional guidance.  
- Output is the contracted vertex set `X` (same connectivity as the input mesh).

### Step 2: Node candidates and optional downsampling

Two options are supported for the node set that seeds the graph:

- Mesh-derived: use all contracted vertices `X` as node candidates.
- kNN-derived: voxel-downsample `X` to produce fewer nodes, suitable for sparser graphs.

### Step 3: Graph construction (mesh-edge vs kNN)

- Mesh-edge graph: edges correspond to unique edges of the input mesh; edge weights are Euclidean lengths on `X`.
- kNN graph: connect each node to `k` nearest neighbors; edge weights are Euclidean distances.

### Step 4: Priority edge-collapse graph thinning

We iteratively collapse short edges to decrease graph complexity while preserving branching structure.

Selection modes:

- `mode="percentile"`: collapse edges whose lengths fall below a per-pass length percentile.
- `mode="pq"`: collapse up to a fixed ratio of the globally shortest edges.
- `mode="pq_heap"`: similar to `pq` but with a heap that re-pushes edge lengths after local updates (better local ordering).

Constraints and safety checks:

- Preserve high-degree nodes: avoid collapsing at vertices with degree ≥ `preserve_branch_degree` (protects junctions/branches).
- Link-condition / degeneracy checks: avoid collapses that would create invalid local configurations.
- Medial protection (optional): if Voronoi guidance is active, edges incident to nodes with high pole weights may be skipped, to keep medial branches.
- Closest-pole policy (optional): prefer collapses that do not move nodes away from their pole targets beyond a tolerance factor.

After each pass, we may rebuild a reduced kNN graph among the surviving nodes to maintain good local connectivity and discourage long shortcuts.

### Step 5: Pruning short leaves

Remove leaf edges shorter than a length quantile. This trims small stubs while preserving main branches.

### Step 6: Compressing degree-2 chains

Compress consecutive degree-2 nodes into a single edge between junctions. This simplifies paths without changing overall topology.

### Step 7: Uniform resampling (optional)

Optionally resample edges to near-uniform spacing along each curve, which can help downstream applications that assume roughly uniform sampling.

### Output and SWC export

The result is a `Skeleton`:

- `nodes`: `(k, 3)` array of 3D points
- `edges`: `(e, 2)` array of undirected edges indexing into `nodes`
- `graph`: `networkx.Graph` with positions stored per-node

You can write to SWC with `Skeleton.write_swc` which breaks cycles using an MST or BFS to produce a spanning forest and annotates removed edges in the file header. `Skeleton.plot_3d` visualizes the skeleton and (optionally) overlays the original mesh.


## Surface thinning variant (mesh-edge collapses)

`pymcfs.skeleton.thin_mesh` runs MCF on the surface and then collapses mesh edges directly (rather than switching to a graph in Euclidean space). The algorithm is similar in spirit to Step 4 above, but applied to the mesh connectivity, with repeated edge collapses and face compaction to thin the surface into a simpler mesh.

This is useful if you prefer to operate in the original mesh domain before converting to a curve graph (e.g., if you want to export a thinned surface as an intermediate product).


## Robustness and practical considerations

- Watertightness and manifoldness: The differential operators assume closed, manifold surfaces. Non-watertight input may produce artifacts or numerical failures. Consider `MeshManager.repair_mesh` to improve mesh quality (duplicate faces, holes, degenerate faces, inconsistent winding).
- Scaling: Absolute sizes affect the choice of `dt` and thresholds. The algorithms operate in world units; tune parameters accordingly.
- Cotangent `secure=True`: Recommended on poor-quality meshes to avoid negative weights from obtuse angles. This is similar to the "secure" mode used in reference C++ code.
- Numerical tolerances: When comparing lengths or checking staleness in priority queues, small tolerances (e.g., `1e-4`–`1e-3` relative) are used to be robust to floating-point noise.


## Complexity and performance notes

- Operator assembly (Laplacian, mass): `O(m)` where `m` is the number of faces.
- Implicit MCF step: solve a sparse SPD linear system `(M + dt L + W) V^{k+1} = ...`. The complexity depends on sparsity patterns and solver; typical sparse Cholesky-based solves are near-linear in practice for large meshes.
- Iterations: doing `T` implicit steps costs roughly `T` linear solves; since `A = (M + dt L + W)` is constant when connectivity is fixed, you can factor once and reuse if needed.
- Graph construction: kNN with KD-tree `O(n log n)` queries; mesh-edge graphs are linear in `|E|`.
- Edge-collapse thinning: dominated by sorting/heap operations `O(E log E)` per pass; rebuilding kNN costs `O(n log n)` per rebuild.


## Relationship to the original MCF skeletonization (Starlab)

The original Starlab/MCF Skelcollapse implementation (Tagliasacchi et al.) formulates each iteration as a single stacked least-squares problem with three blocks of rows:

- Laplacian rows (scaled by per-vertex `omega_L`) enforce contraction
- `omega_H` rows pull vertices toward their original positions
- `omega_P` rows pull vertices toward Voronoi pole targets

This is solved with normal equations and a sparse Cholesky (`SimplicialLDLT`).

`pymcfs` implements a closely related (and widely used) formulation via implicit Euler mean-curvature flow with optional soft guidance:

- `(M + dt L + W) V^{k+1} = M V^{k} + W T`

Both approaches shrink the surface with curvature and allow positional guidance. The least-squares formulation solves a single energy minimization per iteration; our implicit step mimics time integration of the curvature flow while respecting mass. In practice, both yield similar contracted shapes and can be combined with Voronoi guidance for robust skeletonization.


## API quick reference

- `pymcfs.laplacian.cotangent_laplacian(V, F, secure=False)`  
  Build symmetric cotangent Laplacian (zero row-sum). `secure=True` clamps negative cot weights.

- `pymcfs.laplacian.mean_value_laplacian(V, F)`  
  Build symmetric mean-value Laplacian.

- `pymcfs.laplacian.lumped_mass_matrix(V, F)`  
  Build barycentric lumped mass matrix.

- `pymcfs.mcf.mean_curvature_flow(mesh, dt, iterations, laplacian_type, laplacian_secure, guidance_type, guidance_targets, guidance_diag)`  
  Implicit MCF with optional guidance toward targets (`centroid`, `original`, or explicit).

- `pymcfs.medial.compute_voronoi_poles(mesh)`  
  Per-vertex (target, weight) pairs for medial guidance.

- `pymcfs.skeleton.skeletonize(mesh, ...) -> Skeleton`  
  Full Skelcollapse-inspired pipeline: MCF contraction, graph building, priority edge collapses, pruning, chain compression, optional resampling.

- `pymcfs.skeleton.thin_mesh(mesh, ...) -> (V_thin, F_thin)`  
  Surface thinning via mesh-edge collapses after MCF.

- `pymcfs.skeleton.curve_skeleton_from_mesh(V, F, ...) -> Skeleton`  
  Build a curve graph from a surface mesh, then compress/resample.

- `Skeleton.write_swc(path, ...)`  
  Write an SWC file, breaking cycles via MST/BFS.

- `Skeleton.plot_3d(mesh=None, ...) -> plotly.graph_objects.Figure`  
  Interactive 3D visualization of the skeleton with optional mesh overlay.

- `pymcfs.mesh.example_mesh(kind="cylinder"|"torus", ...) -> trimesh.Trimesh`  
  Create a simple demo mesh for notebooks and tests.

- `pymcfs.mesh.MeshManager`  
  Helpers for loading, transforming, analyzing, and repairing meshes.


## References

- A. Tagliasacchi, I. Alhashim, M. Olson, H. Zhang. "Mean Curvature Skeletons." Computer Graphics Forum (SGP), 2012.  
- M. S. Floater. "Mean Value Coordinates." Computer Aided Geometric Design, 2003.  
- U. Pinkall, K. Polthier. "Computing Discrete Minimal Surfaces and Their Conjugates." Experimental Mathematics, 1993.  
- M. Meyer, M. Desbrun, P. Schröder, A. H. Barr. "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds." Visualization and Mathematics III, 2003.
