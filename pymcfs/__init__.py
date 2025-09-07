"""pymcfs: Mean Curvature Flow Skeletonization for triangle meshes.

Public API:
- cotangent_laplacian(V, F)
- lumped_mass_matrix(V, F)
- mean_curvature_flow(mesh, dt=1e-2, iterations=10, cotangent=True, constrain_boundary=False)
- skeletonize(mesh, *, mcf_dt=1e-2, mcf_iters=50, knn=10, length_quantile=0.7)

"""
from .laplacian import cotangent_laplacian, lumped_mass_matrix
from .mcf import mean_curvature_flow
from .skeleton import skeletonize, thin_mesh, curve_skeleton_from_mesh
from .mesh import MeshManager

__all__ = [
    "cotangent_laplacian",
    "lumped_mass_matrix",
    "mean_curvature_flow",
    "skeletonize",
    "thin_mesh",
    "curve_skeleton_from_mesh",
    "MeshManager",
]
