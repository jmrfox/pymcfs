# pymcfs: Mean Curvature Flow Skeletonization

Mean-curvature flow skeletonization (MCFS) of 3D surface meshes in Python.

Inspired by CGAL Triangulated Surface Mesh Skeletonization and "Mean Curvature Skeletons" (Tagliasacchi 2012).

**Defaults** (CGAL-app / complex meshes): `w_H=0.5`, `w_M=5.0`, and
`gate_exterior_poles=True` (medial pull only for Voronoi poles inside the mesh).
Use `skeletonize(..., profile="starlab")` for Starlab parity weights without gating.

## Optional CHOLMOD (faster SPD solve)

MCFS factors the normal equations `AᵀA` each iteration. Install SuiteSparse CHOLMOD
via scikit-sparse for a large speedup on bigger meshes (Linux/WSL: `apt install
libsuitesparse-dev` then `uv sync --extra cholmod`). Set `use_cholmod=False` on the
driver to force SciPy SuperLU.

```bash
uv sync --extra cholmod
```

## Starlab parity

Stage-wise comparison against Starlab `mcfskel` dumps lives under [`fixtures/parity/`](fixtures/parity/README.md).
Parity dumps use the Starlab profile (`w_H=0.1`, `w_M=0.2`, ungated poles).

```bash
uv run python scripts/dump_pymcfs_parity.py --case sindorelax --iters 1,final
uv run python scripts/compare_starlab_parity.py --case sindorelax
uv run pytest tests/test_parity.py
```

---

## Algorithms and design

The sections below explain the main algorithms implemented in `pymcfs`, which follows the implementation of Mean Curvature Skeletons method (Tagliasacchi et al., SGP 2012) and was inspired by CGAL’s `Mean_curvature_flow_skeletonization`.

### Contents

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

### Overview

`pymcfs` turns a watertight triangle mesh into a 1D curve network called a "skeleton". 

1. **Contract** the surface with a weighted mean-curvature solve (optional Voronoi-pole medial term).
2. **Remesh locally**: collapse edges shorter than `min_edge_length`, then split shared edges whose two opposite angles exceed `max_triangle_angle`.
3. **Detect degeneracies**: pin vertices incident to at least two ultra-short, non-collapsible edges.
4. Repeat until the surface area change is small.
5. **Convert** the contracted meso-skeleton into a curve graph by collapsing remaining face-bearing edges.

Internally we use SciPy sparse solvers, trimesh for mesh I/O, NetworkX for the curve graph, and optional PyMeshLab helpers for preprocess.

---

### Discrete surface operators

#### Mesh notation

Let `V` be an `(n, 3)` array of vertex positions and `F` an `(m, 3)` array of triangular faces. Define `E` to be the set of undirected edges of the mesh.

#### Cotangent Laplacian

For an interior edge `(i, j)` shared by two triangles, the off-diagonal weight is

- `L[i, j] = -1/2 * (cot α + cot β)`

with diagonal `L[i, i] = -Σ_{j ≠ i} L[i, j]` (zero row-sum). See `pymcfs.laplacian.cotangent_laplacian(V, F)`. With `secure=True`, negative cotangents are clamped to zero.

The MCFS contraction uses the separate Starlab-compatible operator
`starlab_cotangent_laplacian`: its off-diagonal weight is the unhalved
`cot α + cot β`, angle cosines are clamped to `[-0.999, 0.999]`, and only a
negative summed edge weight is clamped to zero.

#### Mean-value Laplacian

See `pymcfs.laplacian.mean_value_laplacian(V, F)` (Floater, 2003).

#### Lumped mass matrix

`M[i, i] = (1/3) * Σ_{faces incident to i} area(face)`. See `pymcfs.laplacian.lumped_mass_matrix(V, F)`.

#### Properties and robustness

- Symmetry and zero row-sum for Laplacians.
- Algorithms assume closed, manifold meshes. Use `MeshManager.repair_mesh` when needed.

---

### Mean Curvature Flow skeletonization (CGAL-style loop)

The driver is `pymcfs.mcfs.MeanCurvatureFlowSkeletonization`. Each iteration (`contract()`) runs:

```
contract_geometry → collapse_edges → split_faces → detect_degeneracies
```

`contract_until_convergence()` repeats until

`|area_{k-1} - area_k| < area_variation_factor * area_0`

or `max_iterations` is reached. Then `convert_to_skeleton()` builds the curve graph.

#### Contraction system

Per iteration we rebuild the Starlab cotangent Laplacian on the **current**
`(V, F)` and solve the same stacked least-squares system as
`EigenContractionHelper`:

```
        [ W_L L ]             [    0    ]
min_X ‖ [  W_H  ] X - [ W_H V_t ] ‖²
        [  W_P  ]             [ W_P P  ]
```

The implementation solves the normal equations `(AᵀA)X = AᵀB`, matching the
reference Eigen implementation.

- `ω_L` — contraction (Laplacian) weight (fixed at 1; CGAL uses the same scale)
- `ω_H` — attraction to current positions (application default `w_H=0.5`)
- `ω_P` / `w_M` — attraction to Voronoi poles (application default `w_M=5.0`)

When `gate_exterior_poles=True` (default), medial weights are applied only for
poles that lie inside the input mesh — matching CGAL
`Side_of_triangle_mesh` / `ON_BOUNDED_SIDE`. Exterior poles get `w_M = 0` so
they cannot pull branches outside the surface on complex TS-like meshes.

`skeletonize(..., profile="starlab")` selects Starlab parity weights
(`w_H=0.1`, `w_M=0.2`) with ungated poles for fixture dumps.

Fixed (pinned) vertices use `ω_L = 0`, `ω_H = 1/zero_TH`, `ω_P = 0`. Split vertices from obtuse splits have `ω_P = 0`.

#### Local remeshing

Implemented in `pymcfs.remesh`:

- **collapse_edges**: collapse edges with length `< min_edge_length` (default `0.002 × bbox_diagonal`) when the manifold link condition holds; midpoint merge; optional closest-pole update.
- **split_faces**: split an interior edge only when the angles opposite it in both incident triangles exceed `max_triangle_angle` (default 110°). The split updates both faces, preserving the closed manifold.

This is the consolidation step as the surface shrinks—**not** a post-hoc graph degree filter.

#### Degeneracy detection (branch pinning)

Starlab/CGAL do **not** skip collapses based on mesh valence (typical valence is ~6, so a `preserve_branch_degree=3` gate would block nearly everything). Instead, when ≥2 incident ultra-short edges fail the link condition, the vertex is treated as a formed branch and **pinned**. That is `detect_degeneracies()`.

#### Convert meso-skeleton to curve

After convergence, process edges in Starlab's length-priority order and collapse
an edge only while it still has incident faces. Surviving graph vertices are
placed at the centroid of the meso-skeleton vertices collapsed into them.

Optional, non-core post-processing can refine the curve graph (disabled by
default via `refine=False` on `skeletonize` / `convert_to_skeleton`):

- `refine=True` / `"uniform"` — arc-length resample each chain between
  junctions/leaves (and closed cycles) to a more even spacing. Default spacing
  is `2 × median edge length` (mild downsample). Override with absolute
  `refine_spacing` or relative `refine_spacing_frac` (fraction of skeleton bbox
  diagonal).
- `refine="compress"` — drop all degree-2 nodes; keep only junctions and leaves.

Legacy `compress_chains` / `resample_spacing` map onto this path. Prefer
chain-aware uniform refine over compress-then-chord-subdivide, which flattens
curved medial paths.

`skeletonize()` is a thin wrapper around this driver. `thin_mesh()` returns the meso-skeleton `(V, F)` without curve conversion.

---

### Voronoi medial guidance

`pymcfs.medial.compute_voronoi_poles(mesh)` returns per-vertex medial targets.
Those targets feed the `ω_P` / `w_M` term in the MCFS driver. CGAL only
assembles the medial block when the pole is on the bounded side of the mesh;
pymcfs mirrors that with `gate_exterior_poles` (on by default).

---

### Robustness and practical considerations

- Watertight manifold input is required for stable Laplacians and link-conditioned collapses.
- `min_edge_length` is relative to the bounding-box diagonal by default, so absolute mesh scale (e.g. µm neuron meshes) is handled without hand-tuning `dt`.
- MCFS uses `mcfs_cotangent_laplacian` (Starlab-weighted off-diagonals, unweighted diagonal).
- **Tier A speedups (same discrete algorithm):** cached `mesh.contains` for exterior-pole gating (recomputed only when poles are remapped); optional CHOLMOD via `pymcfs[cholmod]` / `use_cholmod` (~3× on the geometry slice, ~1% full iteration); faster MCFS Laplacian edge assembly (argsort reduce instead of `np.unique`); short-edge collapse uses face-walk order with incremental adjacency and Numba `link_condition_ok` / `apply_collapse_local` in `topology.py`; obtuse split apply pass uses preallocated buffers (see `docs/benchmarks.md`).

---

### Relationship to Starlab / CGAL

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

### API quick reference

- `pymcfs.mcfs.MeanCurvatureFlowSkeletonization(mesh, ...)`  
  CGAL-style driver: `contract_geometry`, `collapse_edges`, `split_faces`, `detect_degeneracies`, `contract`, `contract_until_convergence`, `convert_to_skeleton`. Defaults: `w_H=0.5`, `w_M=5.0`, `gate_exterior_poles=True`.

- `pymcfs.skeleton.skeletonize(mesh, ...) -> Skeleton`  
  Full MCFS pipeline wrapper. Optional `refine=True` / `"uniform"` / `"compress"`.
  Use `profile="starlab"` for Starlab parity weights without pole gating.

- `pymcfs.skeleton.refine_skeleton(skel, ...)` / `Skeleton.refine(...)`  
  Post-hoc curve-graph refinement (arc-length resample or compress).

- `pymcfs.skeleton.thin_mesh(mesh, ...) -> (V, F)`  
  Meso-skeleton surface after contraction + remesh.

- `pymcfs.skeleton.curve_skeleton_from_mesh(V, F, ...) -> Skeleton`  
  Face-bearing edge collapse to a curve graph.

- `pymcfs.laplacian.cotangent_laplacian` / `mean_value_laplacian` / `lumped_mass_matrix`

- `pymcfs.mcf.mean_curvature_flow`  
  Standalone implicit Euler MCF (no remesh); useful for experiments. The skeletonization path uses the CGAL-style driver above.

- `pymcfs.medial.compute_voronoi_poles(mesh)`

- `Skeleton.write_polylines` / `Skeleton.to_polylines` / `Skeleton.plot_3d`

- `pymcfs.mesh.example_mesh` / `MeshManager`

---

### References

- A. Tagliasacchi, I. Alhashim, M. Olson, H. Zhang. "Mean Curvature Skeletons." Computer Graphics Forum (SGP), 2012.
- CGAL: Triangulated Surface Mesh Skeletonization (`CGAL::Mean_curvature_flow_skeletonization`).
- M. S. Floater. "Mean Value Coordinates." Computer Aided Geometric Design, 2003.
- U. Pinkall, K. Polthier. "Computing Discrete Minimal Surfaces and Their Conjugates." Experimental Mathematics, 1993.
- M. Meyer, M. Desbrun, P. Schröder, A. H. Barr. "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds." Visualization and Mathematics III, 2003.
