"""pymcfs: mean-curvature-flow skeletonization for triangle meshes.

User-facing paths::

    from pymcfs import skeletonize, thin_mesh, MeanCurvatureFlowSkeletonization

    skel = skeletonize(mesh)          # 1D curve skeleton
    V, F = thin_mesh(mesh)            # contracted meso-skeleton surface
    mcs = MeanCurvatureFlowSkeletonization(mesh)  # step-through driver

Also public: ``propose_mcfs_params`` / ``profile=\"auto\"``, ``analyze_skeleton`` /
``score_skeleton``, ``validate_mcfs_mesh``, and optional ``MeshManager`` utilities.
"""

__version__ = "0.1.0"

from .mcfs import MeanCurvatureFlowSkeletonization
from .skeleton import (
    skeletonize,
    thin_mesh,
    curve_skeleton_from_mesh,
    Skeleton,
    refine_skeleton,
    prune_exterior_branches,
)
from .mesh import MeshManager, example_mesh
from .quality import analyze_skeleton, score_skeleton, SkeletonQualityReport, SkeletonScore
from .params import (
    mesh_mcfs_features,
    propose_mcfs_params,
    McfsParams,
    MeshMcfsFeatures,
    BranchingPreference,
)
from .validate import validate_mcfs_mesh

# Advanced / experimental helpers
from .laplacian import cotangent_laplacian, lumped_mass_matrix, mcfs_cotangent_laplacian
from .mcf import mean_curvature_flow, MCFResult
from .cg_io import read_cg, write_cg

__all__ = [
    "skeletonize",
    "MeanCurvatureFlowSkeletonization",
    "Skeleton",
    "refine_skeleton",
    "prune_exterior_branches",
    "analyze_skeleton",
    "score_skeleton",
    "SkeletonQualityReport",
    "SkeletonScore",
    "mesh_mcfs_features",
    "propose_mcfs_params",
    "McfsParams",
    "MeshMcfsFeatures",
    "BranchingPreference",
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
    "read_cg",
    "write_cg",
    "__version__",
]
