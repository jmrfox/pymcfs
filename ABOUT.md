# pymcfs: Algorithms and Design

This document explains the main algorithms implemented in `pymcfs`, which follows the implementation of Mean Curvature Skeletons method (Tagliasacchi et al., SGP 2012) and was inspired by CGAL’s `Mean_curvature_flow_skeletonization`.

## Contents

- Overview
- Discrete surface operators
  - Mesh notation
  - Cotangent Laplacian
  - Mean-value Laplacian
  - Lumped mass matrix
  - Properties and robustness ("secure" mode)
- Mean Curvature Flow skeletonization (CGAL-style loop)
  - Contraction system
  - Local remeshing (collapse + split)
  - Degeneracy detection (branch pinning)
  - Convergence
  - Convert meso-skeleton to curve
- Voronoi medial guidance
- Robustness and practical considerations
- Relationship to Starlab / CGAL
- API quick reference
- References

---

## Overview

`pymcfs` turns a watertight triangle mesh into a 1D curve network called a "skeleton". 

1. **Contract** the surface with a weighted mean-curvature solve (optional Voronoi-pole medial term).
2. **Remesh locally**: collapse edges shorter than `min_edge_length`, then split faces with angles larger than `max_triangle_angle`.
3. **Detect degeneracies**: pin vertices whose local neighborhood is no longer a topological disk.
4. Repeat until the surface area change is small.
5. **Convert** the contracted meso-skeleton into a curve graph by collapsing remaining face-bearing edges.

Internally we use SciPy sparse solvers, trimesh for mesh I/O, NetworkX for the curve graph, and optional PyMeshLab helpers for preprocess.
---

## Discrete surface operators

### Mesh notation

Let `V` be an `(n, 3)` array of vertex positions and `F` an `(m, 3)` array of triangular faces. Define `E` to be the set of undirected edges of the mesh.

### Cotangent Laplacian

For an interior edge `(i, j)` shared by two triangles, the off-diagonal weight is

- `L[i, j] = -1/2 * (cot α + cot β)`

with diagonal `L[i, i] = -Σ_{j ≠ i} L[i, j]` (zero row-sum). See `pymcfs.laplacian.cotangent_laplacian(V, F)`. With `secure=True`, negative cotangents are clamped to zero.

### Mean-value Laplacian

See `pymcfs.laplacian.mean_value_laplacian(V, F)` (Floater, 2003).

### Lumped mass matrix

`M[i, i] = (1/3) * Σ_{faces incident to i} area(face)`. See `pymcfs.laplacian.lumped_mass_matrix(V, F)`.

### Properties and robustness

- Symmetry and zero row-sum for Laplacians.
- Algorithms assume closed, manifold meshes. Use `MeshManager.repair_mesh` when needed.

---

## Mean Curvature Flow skeletonization (CGAL-style loop)

The driver is `pymcfs.mcfs.MeanCurvatureFlowSkeletonization`. Each iteration (`contract()`) runs:

```
contract_geometry → collapse_edges → split_faces → detect_degeneracies
```

`contract_until_convergence()` repeats until

`|area_{k-1} - area_k| < area_variation_factor * area_0`

or `max_iterations` is reached. Then `convert_to_skeleton()` builds the curve graph.

### Contraction system

Per iteration we rebuild the cotangent Laplacian on the **current** `(V, F)` (connectivity changes after remesh) and solve a weighted system in the spirit of Tagliasacchi / CGAL / Starlab:

- `ω_L` — contraction (Laplacian) weight  
- `ω_H` — attraction to current positions  
- `ω_P` — attraction to Voronoi poles (medial centering)

Fixed (pinned) vertices use `ω_L = 0`, `ω_H = 1/zero_TH`, `ω_P = 0`. Split vertices from obtuse splits have `ω_P = 0`.

### Local remeshing

Implemented in `pymcfs.remesh`:

- **collapse_edges**: collapse edges with length `< min_edge_length` (default `0.002 × bbox_diagonal`) when the manifold link condition holds; midpoint merge; optional closest-pole update.
- **split_faces**: if a triangle angle exceeds `max_triangle_angle` (default 110°), project the obtuse vertex onto the opposite edge and split.

This is the consolidation step as the surface shrinks—**not** a post-hoc graph degree filter.

### Degeneracy detection (branch pinning)

Starlab/CGAL do **not** skip collapses based on mesh valence (typical valence is ~6, so a `preserve_branch_degree=3` gate would block nearly everything). Instead, when ≥2 incident ultra-short edges fail the link condition, the vertex is treated as a formed branch and **pinned**. That is `detect_degeneracies()`.

### Convert meso-skeleton to curve

After convergence, repeatedly collapse the shortest **face-bearing** edge until faces are gone (or residual surface edges remain as the curve). Edges that no longer bound faces are curve segments and are not collapsed. Optional post-process on the 1D NetworkX graph: compress degree-2 chains and uniform resample.

`skeletonize()` is a thin wrapper around this driver. `thin_mesh()` returns the meso-skeleton `(V, F)` without curve conversion.

---

## Voronoi medial guidance

`pymcfs.medial.compute_voronoi_poles(mesh)` returns per-vertex medial targets. When `is_medially_centered=True` (or `guidance_type="voronoi"`), those targets feed the `ω_P` term.

---

## Robustness and practical considerations

- Watertight manifold input is required for stable Laplacians and link-conditioned collapses.
- `min_edge_length` is relative to the bounding-box diagonal by default, so absolute mesh scale (e.g. µm neuron meshes) is handled without hand-tuning `dt`.
- Cotangent `secure=True` is the default in the MCFS driver.

---

## Relationship to Starlab / CGAL

| CGAL / Starlab | pymcfs |
|---|---|
| `contract_geometry` | `MeanCurvatureFlowSkeletonization.contract_geometry` |
| `collapse_edges` / TopologyJanitor | `pymcfs.remesh.collapse_short_edges` |
| `split_faces` | `pymcfs.remesh.split_obtuse_faces` |
| `detect_degeneracies` / `visfixed` | `detect_degeneracies` |
| `convert_to_skeleton` | `convert_to_skeleton` / `meso_surface_to_curve_graph` |
| Boost adjacency_list skeleton | NetworkX `Skeleton.graph` |

CGAL’s published C++ API is the algorithmic guide; this package reimplements it in Python because CGAL’s Python bindings do not expose MCFS.

---

## API quick reference

- `pymcfs.mcfs.MeanCurvatureFlowSkeletonization(mesh, ...)`  
  CGAL-style driver: `contract_geometry`, `collapse_edges`, `split_faces`, `detect_degeneracies`, `contract`, `contract_until_convergence`, `convert_to_skeleton`.

- `pymcfs.skeleton.skeletonize(mesh, ...) -> Skeleton`  
  Full MCFS pipeline wrapper.

- `pymcfs.skeleton.thin_mesh(mesh, ...) -> (V, F)`  
  Meso-skeleton surface after contraction + remesh.

- `pymcfs.skeleton.curve_skeleton_from_mesh(V, F, ...) -> Skeleton`  
  Face-bearing edge collapse to a curve graph.

- `pymcfs.laplacian.cotangent_laplacian` / `mean_value_laplacian` / `lumped_mass_matrix`

- `pymcfs.mcf.mean_curvature_flow`  
  Standalone implicit Euler MCF (no remesh); useful for experiments. The skeletonization path uses the CGAL-style driver above.

- `pymcfs.medial.compute_voronoi_poles(mesh)`

- `Skeleton.write_swc` / `Skeleton.plot_3d`

- `pymcfs.mesh.example_mesh` / `MeshManager`

---

## References

- A. Tagliasacchi, I. Alhashim, M. Olson, H. Zhang. "Mean Curvature Skeletons." Computer Graphics Forum (SGP), 2012.
- CGAL: Triangulated Surface Mesh Skeletonization (`CGAL::Mean_curvature_flow_skeletonization`).
- M. S. Floater. "Mean Value Coordinates." Computer Aided Geometric Design, 2003.
- U. Pinkall, K. Polthier. "Computing Discrete Minimal Surfaces and Their Conjugates." Experimental Mathematics, 1993.
- M. Meyer, M. Desbrun, P. Schröder, A. H. Barr. "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds." Visualization and Mathematics III, 2003.
