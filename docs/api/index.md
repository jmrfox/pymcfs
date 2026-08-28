# API reference

Public entry points. Internals (`remesh`, `topology`, …) are omitted unless you
import them explicitly.

```python
from pymcfs import (
    skeletonize,
    thin_mesh,
    MeanCurvatureFlowSkeletonization,
    Skeleton,
    propose_mcfs_params,
    analyze_skeleton,
    score_skeleton,
    validate_mcfs_mesh,
)
```

| Module | Page |
|--------|------|
| High-level skeletonization | [skeletonize / Skeleton](skeleton.md) |
| Step-through driver | [MCFS driver](mcfs.md) |
| Weight oracle | [Parameters](params.md) |
| Quality scoring | [Quality](quality.md) |
| Poles, Laplacians, MCF | [Geometry](geometry.md) |
