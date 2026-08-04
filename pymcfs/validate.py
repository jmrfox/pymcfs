"""Input mesh validation for MCF skeletonization."""
from __future__ import annotations

import numpy as np
import trimesh as tm


def validate_mcfs_mesh(mesh: tm.Trimesh, *, require_watertight: bool = True) -> tm.Trimesh:
    """Ensure ``mesh`` is a single-component closed watertight triangle mesh.

    Raises
    ------
    TypeError
        If ``mesh`` is not a ``trimesh.Trimesh``.
    ValueError
        If the mesh fails MCFS input preconditions.
    """
    if not isinstance(mesh, tm.Trimesh):
        raise TypeError(f"expected trimesh.Trimesh, got {type(mesh)!r}")

    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("mesh faces must be triangles with shape (m, 3)")
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError("mesh vertices must have shape (n, 3)")
    if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
        raise ValueError("mesh is empty")

    if require_watertight and not bool(mesh.is_watertight):
        raise ValueError(
            "mesh must be watertight (closed, no holes). "
            "Repair it first (e.g. MeshManager.repair_mesh) or fix the source mesh."
        )

    # Boundary edges: edges with multiplicity 1
    if require_watertight:
        edges = np.sort(np.asarray(mesh.faces[:, [0, 1, 1, 2, 2, 0]]).reshape(-1, 2), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        n_boundary = int(np.sum(counts == 1))
        if n_boundary > 0:
            raise ValueError(
                f"mesh has {n_boundary} boundary edges; "
                "MCFS requires a closed surface without boundary"
            )

    try:
        comps = mesh.split(only_watertight=False)
        n_comp = len(comps)
    except Exception:
        n_comp = 1
    if n_comp != 1:
        raise ValueError(
            f"mesh must have exactly one connected component (found {n_comp})"
        )

    return mesh
