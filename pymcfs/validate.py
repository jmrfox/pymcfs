"""Validate (and optionally load/repair) meshes before MCFS contraction."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import trimesh as tm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def validate_mcfs_mesh(mesh: tm.Trimesh, *, require_watertight: bool = True) -> tm.Trimesh:
    """Ensure ``mesh`` meets MCFS input preconditions.

    Always requires a non-empty triangle mesh with exactly one connected
    component. When ``require_watertight`` is True (default), also requires a
    closed watertight surface with no boundary edges.

    Parameters
    ----------
    mesh :
        Input ``trimesh.Trimesh``.
    require_watertight :
        If True (default), enforce watertightness and zero boundary edges.
        Set False when loading a mesh that will be repaired first.

    Returns
    -------
    trimesh.Trimesh
        The same ``mesh`` object if validation succeeds.

    Raises
    ------
    TypeError
        If ``mesh`` is not a ``trimesh.Trimesh``.
    ValueError
        If the mesh fails the selected preconditions.
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
            "Repair it first (e.g. load_and_repair / MeshManager.repair_mesh) "
            "or fix the source mesh."
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


def load_and_repair(
    mesh: tm.Trimesh | str,
    *,
    repair_kwargs: dict[str, Any] | None = None,
    file_format: str | None = None,
) -> tm.Trimesh:
    """Load (if needed) and prepare an MCFS-ready mesh via validate → repair → re-validate.

    Soft-validates without requiring watertightness, attempts common repairs,
    then re-validates with full MCFS preconditions.

    Parameters
    ----------
    mesh :
        A ``trimesh.Trimesh`` or a filesystem path to load.
    repair_kwargs :
        Forwarded to :meth:`pymcfs.mesh.MeshManager.repair_mesh`.
    file_format :
        Optional format hint when ``mesh`` is a path.

    Returns
    -------
    trimesh.Trimesh
        Repaired mesh that passes :func:`validate_mcfs_mesh`.

    Raises
    ------
    TypeError
        If ``mesh`` is neither a path nor a ``trimesh.Trimesh``.
    ValueError
        If the mesh still fails MCFS validation after repair.
    """
    from .mesh import MeshManager

    if isinstance(mesh, str):
        mgr = MeshManager()
        logger.info("load_and_repair: loading %s", mesh)
        mgr.load_mesh(mesh, file_format=file_format, validate_mcfs=False)
        tri = mgr.mesh
    elif isinstance(mesh, tm.Trimesh):
        mgr = MeshManager(mesh)
        tri = mesh
    else:
        raise TypeError("mesh must be a trimesh.Trimesh or a filesystem path string")

    validate_mcfs_mesh(tri, require_watertight=False)
    kwargs = dict(repair_kwargs or {})
    kwargs.setdefault("verbose", False)
    logger.info("load_and_repair: repairing mesh (n=%d f=%d)", len(tri.vertices), len(tri.faces))
    repaired = mgr.repair_mesh(**kwargs)
    validate_mcfs_mesh(repaired, require_watertight=True)
    logger.info(
        "load_and_repair: ready (n=%d f=%d watertight=%s)",
        len(repaired.vertices),
        len(repaired.faces),
        repaired.is_watertight,
    )
    return repaired
