# API reference

Public entry points from `pymcfs`. Internals (`remesh`, `topology`, …) are
omitted unless you import them explicitly.

```python
from pymcfs import (
    skeletonize,
    contract_mesh,
    curve_skeleton_from_mesh,
    MeanCurvatureFlowSkeletonization,
    Skeleton,
    resample_skeleton,
    prune_exterior_branches,
    prune_short_leaves,
    prune_thick_hubs,
    extend_tips,
    propose_mcfs_params,
    mesh_mcfs_features,
    search_mcfs_params,
    analyze_skeleton,
    score_skeleton,
    validate_mcfs_mesh,
    load_and_repair,
    read_cg,
    write_cg,
    SkeletonizeSettings,
    ContractionSettings,
    RefineSettings,
    MeshManager,
    example_mesh,
)
```

| Module | Page |
|--------|------|
| High-level skeletonization | [skeletonize / Skeleton](skeleton.md) |
| Step-through driver | [MCFS driver](mcfs.md) |
| Settings dataclasses | [Settings](config.md) |
| Parameter proposals | [Parameters](params.md) |
| Parameter search | [Search](search.md) |
| Quality scoring | [Quality](quality.md) |
| Mesh prep utilities | [Mesh utilities](mesh.md) |
| Curve graph I/O | [Curve I/O](io.md) |
| Poles, Laplacians, MCF | [Geometry](geometry.md) |
