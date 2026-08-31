"""pymcfs: extract a 1D curve skeleton from a closed triangle mesh.

Mean-curvature flow contracts the surface toward the medial axis (Voronoi
poles), yielding a thin meso-skeleton surface that is then converted to a
curve graph and refined (prune / optional tip extension / resample).

User-facing paths::

    from pymcfs import skeletonize, contract_mesh, MeanCurvatureFlowSkeletonization

    skel = skeletonize(mesh)          # 1D curve skeleton
    V, F = contract_mesh(mesh)        # contracted meso-skeleton surface
    mcs = MeanCurvatureFlowSkeletonization(mesh)  # step-through driver

Also public: ``propose_mcfs_params`` / ``profile=\"auto\"``, ``analyze_skeleton`` /
``score_skeleton``, ``search_mcfs_params`` (or ``parameter_search=True`` on
``skeletonize``), ``validate_mcfs_mesh`` / ``load_and_repair``, settings
dataclasses, and optional ``MeshManager`` utilities.
"""

__version__ = "0.1.0"

from .mcfs import MeanCurvatureFlowSkeletonization
from .skeleton import (
    skeletonize,
    contract_mesh,
    curve_skeleton_from_mesh,
    Skeleton,
    resample_skeleton,
    prune_exterior_branches,
    prune_short_leaves,
    prune_thick_hubs,
    extend_tips,
)
from .config import (
    ContractionSettings,
    RefineSettings,
    SkeletonizeSettings,
)
from .mesh import MeshManager, example_mesh
from .quality import analyze_skeleton, score_skeleton, SkeletonQualityReport, SkeletonScore
from .search import McfsSearchResult, search_mcfs_params, score_skeleton_candidate
from .params import (
    mesh_mcfs_features,
    propose_mcfs_params,
    McfsParams,
    MeshMcfsFeatures,
    BranchingPreference,
)
from .validate import validate_mcfs_mesh, load_and_repair
from .cg_io import read_cg, write_cg

__all__ = [
    "skeletonize",
    "MeanCurvatureFlowSkeletonization",
    "Skeleton",
    "resample_skeleton",
    "prune_exterior_branches",
    "prune_short_leaves",
    "prune_thick_hubs",
    "extend_tips",
    "analyze_skeleton",
    "score_skeleton",
    "score_skeleton_candidate",
    "SkeletonQualityReport",
    "SkeletonScore",
    "search_mcfs_params",
    "McfsSearchResult",
    "mesh_mcfs_features",
    "propose_mcfs_params",
    "McfsParams",
    "MeshMcfsFeatures",
    "BranchingPreference",
    "ContractionSettings",
    "RefineSettings",
    "SkeletonizeSettings",
    "validate_mcfs_mesh",
    "load_and_repair",
    "contract_mesh",
    "curve_skeleton_from_mesh",
    "MeshManager",
    "example_mesh",
    "read_cg",
    "write_cg",
    "__version__",
]
