# Algorithm overview

`pymcfs` follows Mean Curvature Skeletons (Tagliasacchi et al., SGP 2012) and
CGAL’s `Mean_curvature_flow_skeletonization`.

## Pipeline

1. **Contract** the surface with a weighted mean-curvature solve (optional Voronoi-pole medial term).
2. **Remesh locally**: collapse edges shorter than `min_edge_length`, then split shared edges whose two opposite angles exceed `max_triangle_angle`.
3. **Detect degeneracies**: pin vertices incident to at least two ultra-short, non-collapsible edges.
4. Repeat until the surface area change is small.
5. **Convert** the contracted meso-skeleton into a curve graph by collapsing remaining face-bearing edges.

Internally: SciPy sparse solvers, trimesh for mesh I/O, NetworkX for the curve graph.
Optional `MeshManager.repair_mesh` for preprocess.

## Discrete surface operators

### Mesh notation

Let `V` be an `(n, 3)` array of vertex positions and `F` an `(m, 3)` array of
triangular faces.

### Cotangent Laplacian

For an interior edge `(i, j)` shared by two triangles:

- `L[i, j] = -1/2 * (cot α + cot β)`
- `L[i, i] = -Σ_{j ≠ i} L[i, j]` (zero row-sum)

See `pymcfs.laplacian.cotangent_laplacian(V, F)`. With `secure=True`, negative
cotangents are clamped to zero.

MCFS contraction uses the Starlab-compatible
`mcfs_cotangent_laplacian`: off-diagonal weight is the unhalved `cot α + cot β`,
angle cosines clamped to `[-0.999, 0.999]`.

### Mean-value Laplacian / mass matrix

- `mean_value_laplacian(V, F)` — Floater (2003)
- `lumped_mass_matrix(V, F)` — `M[i,i] = (1/3) Σ area(incident faces)`

## MCFS contraction loop

Driver: `MeanCurvatureFlowSkeletonization`. Each `contract()` runs:

```text
contract_geometry → collapse_edges → split_faces → detect_degeneracies
```

`contract_until_convergence()` repeats until

`|area_{k-1} - area_k| < area_variation_factor * area_0`

or `max_iterations` / timeout / remesh-growth abort.

### Contraction system

Rebuild the Starlab cotangent Laplacian on the current `(V, F)` and solve:

```text
        [ W_L L ]             [    0    ]
min_X ‖ [  W_H  ] X - [ W_H V_t ] ‖²
        [  W_P  ]             [ W_P P  ]
```

via normal equations `(AᵀA)X = AᵀB`.

- `ω_L` — fixed at 1
- `ω_H` / `w_H` — attraction to current positions (default 0.5)
- `ω_P` / `w_M` — attraction to Voronoi poles (default 5.0)

Pinned vertices: `ω_L = 0`, `ω_H = 1/zero_TH`, `ω_P = 0`.
Split vertices (one step): `ω_P = 0`.

### Local remeshing

- **collapse_edges**: length `< min_edge_length` (default `0.002 × bbox_diag`) when the link condition holds
- **split_faces**: shared edge only if both opposite angles exceed `max_triangle_angle` (default 110°)

### Degeneracy detection

When ≥2 incident ultra-short edges fail the link condition, the vertex is
**pinned** as a formed branch (`detect_degeneracies`).

### Convert to curve

Collapse face-bearing edges in length-priority order. Optional `refine` on
`skeletonize` / `convert_to_skeleton`:

- `refine=True` / `"uniform"` — arc-length resample chains
- `refine="compress"` — keep only junctions and leaves

## Voronoi medial guidance

`compute_voronoi_poles(mesh)` supplies per-vertex medial targets for `w_M`.
With gating on, only poles inside the input mesh receive medial weight.

## Relationship to Starlab / CGAL

| CGAL / Starlab | pymcfs |
|---|---|
| `contract_geometry` | `MeanCurvatureFlowSkeletonization.contract_geometry` |
| `collapse_edges` | `pymcfs.remesh.collapse_short_edges` |
| `split_faces` | `pymcfs.remesh.split_obtuse_faces` |
| `detect_degeneracies` | `detect_degeneracies` |
| `convert_to_skeleton` | `convert_to_skeleton` |
| Boost adjacency_list | NetworkX `Skeleton.graph` |

## References

- A. Tagliasacchi, I. Alhashim, M. Olson, H. Zhang. "Mean Curvature Skeletons." Computer Graphics Forum (SGP), 2012.
- CGAL: Triangulated Surface Mesh Skeletonization (`CGAL::Mean_curvature_flow_skeletonization`).
- M. S. Floater. "Mean Value Coordinates." Computer Aided Geometric Design, 2003.
- U. Pinkall, K. Polthier. "Computing Discrete Minimal Surfaces and Their Conjugates." Experimental Mathematics, 1993.
- M. Meyer, M. Desbrun, P. Schröder, A. H. Barr. "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds." Visualization and Mathematics III, 2003.
