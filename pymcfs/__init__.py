"""pymcfs: Mean Curvature Flow Skeletonization for triangle meshes.

Primary entry point::

    from pymcfs import skeletonize
    skel = skeletonize(mesh, w_H=0.1, w_M=0.2)
"""

from .mcfs import MeanCurvatureFlowSkeletonization
from .skeleton import (
    skeletonize,
    thin_mesh,
    curve_skeleton_from_mesh,
    Skeleton,
    refine_skeleton,
)
from .mesh import MeshManager, example_mesh
from .quality import analyze_skeleton, SkeletonQualityReport
from .validate import validate_mcfs_mesh

# Advanced / experimental helpers
from .laplacian import cotangent_laplacian, lumped_mass_matrix
from .mcf import mean_curvature_flow

__all__ = [
    "skeletonize",
    "MeanCurvatureFlowSkeletonization",
    "Skeleton",
    "refine_skeleton",
    "analyze_skeleton",
    "SkeletonQualityReport",
    "validate_mcfs_mesh",
    "thin_mesh",
    "curve_skeleton_from_mesh",
    "MeshManager",
    "example_mesh",
    "cotangent_laplacian",
    "lumped_mass_matrix",
    "mean_curvature_flow",
]
