"""pymcfs: mean-curvature-flow skeletonization for triangle meshes.

User-facing paths::

    from pymcfs import skeletonize, thin_mesh, MeanCurvatureFlowSkeletonization

    skel = skeletonize(mesh)          # 1D curve skeleton
    V, F = thin_mesh(mesh)            # contracted meso-skeleton surface
    mcs = MeanCurvatureFlowSkeletonization(mesh)  # step-through driver
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
from .laplacian import cotangent_laplacian, lumped_mass_matrix, mcfs_cotangent_laplacian
from .mcf import mean_curvature_flow, MCFResult

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
    "mcfs_cotangent_laplacian",
    "cotangent_laplacian",
    "lumped_mass_matrix",
    "mean_curvature_flow",
    "MCFResult",
]
