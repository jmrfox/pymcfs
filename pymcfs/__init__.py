"""pymcfs: Mean Curvature Flow Skeletonization for triangle meshes.

Public API mirrors CGAL's Mean_curvature_flow_skeletonization workflow:
contract → collapse short edges → split obtuse faces → detect degeneracies →
convert to a curve skeleton.
"""

from .laplacian import cotangent_laplacian, lumped_mass_matrix
from .mcf import mean_curvature_flow
from .mcfs import MeanCurvatureFlowSkeletonization
from .skeleton import skeletonize, thin_mesh, curve_skeleton_from_mesh, Skeleton
from .mesh import MeshManager, example_mesh

__all__ = [
    "cotangent_laplacian",
    "lumped_mass_matrix",
    "mean_curvature_flow",
    "MeanCurvatureFlowSkeletonization",
    "skeletonize",
    "thin_mesh",
    "curve_skeleton_from_mesh",
    "Skeleton",
    "MeshManager",
    "example_mesh",
]
