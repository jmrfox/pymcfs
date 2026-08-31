"""Voronoi-pole (medial-axis target) helpers used during MCFS contraction.

Poles are per-vertex points near the medial axis inside the volume; medial
weight pulls surface vertices toward them so the meso-skeleton stays centered.
"""
from __future__ import annotations

import logging

import numpy as np
import trimesh as tm
from scipy.spatial import Voronoi

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def points_inside_mesh(
    mesh: tm.Trimesh, points: np.ndarray, *, fast: bool = False
) -> np.ndarray:
    """Boolean mask of ``points`` strictly inside ``mesh``.

    ``mesh.contains`` silently switches to Embree whenever ``embreex`` is
    importable, and Embree traces in single precision. On meshes whose
    coordinates sit far from the origin that loses enough precision to flip
    the majority of pole gating decisions, so the exact float64 traverser is
    the default and the fast backend has to be requested explicitly.

    Parameters
    ----------
    mesh :
        Closed triangle mesh used for containment tests.
    points : (k, 3) float
        Query points.
    fast :
        Use the mesh's own ray backend (Embree when installed). Much faster,
        but only trustworthy for meshes at unit-ish scale near the origin.

    Returns
    -------
    (k,) bool ndarray
        True where the point is strictly inside the mesh.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    if fast:
        return np.asarray(mesh.contains(points), dtype=bool)
    from trimesh.ray.ray_triangle import RayMeshIntersector

    intersector = RayMeshIntersector(mesh)
    return np.asarray(intersector.contains_points(points), dtype=bool)


def compute_voronoi_poles(mesh: tm.Trimesh, *, use_vertex_normals: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-vertex medial (inner Voronoi) poles for contraction guidance.

    A Voronoi pole is a point in the vertex's Voronoi cell chosen as a
    medial-axis target: the finite locus farthest inward along the surface
    normal. Compatible with Starlab ``mcfskel`` / ``QhullVoronoiHelper``.

    Parameters
    ----------
    mesh :
        Input triangle mesh.
    use_vertex_normals :
        Currently unused for branching (both paths use ``mesh.vertex_normals``);
        retained for API compatibility with older call sites.

    Returns
    -------
    targets : (n, 3) float
        Medial target positions (inner Voronoi poles) for each vertex.
    weights : (n,) float
        Suggested diagonal guidance weights per vertex in ``[0, 1]``.

    Raises
    ------
    TypeError
        If ``mesh`` is not a ``trimesh.Trimesh``.
    """
    if not isinstance(mesh, tm.Trimesh):
        raise TypeError("compute_voronoi_poles expects a trimesh.Trimesh")

    P = np.asarray(mesh.vertices, dtype=float)
    n = P.shape[0]
    if n == 0:
        return P.copy(), np.zeros((0,), dtype=float)

    # Vertex normals (unit)
    if use_vertex_normals and mesh.vertex_normals is not None:
        N = np.asarray(mesh.vertex_normals, dtype=float).copy()
    else:
        N = np.asarray(mesh.vertex_normals, dtype=float).copy()
    norms = np.linalg.norm(N, axis=1)
    norms = np.where(norms > 0, norms, 1.0)
    N = N / norms[:, None]

    vor = Voronoi(P)

    targets = P.copy()
    weights = np.zeros(n, dtype=float)

    eps = 1e-9
    Vverts = vor.vertices
    point_region = vor.point_region
    regions = vor.regions
    bounds_min = P.min(axis=0)
    bounds_max = P.max(axis=0)
    locus_valid = np.all(
        (Vverts >= bounds_min[None, :]) & (Vverts <= bounds_max[None, :]),
        axis=1,
    )
    valid_loci = np.flatnonzero(locus_valid)
    fallback_pole = Vverts[valid_loci[0]] if valid_loci.size else None

    for i in range(n):
        r_idx = int(point_region[i])
        if r_idx < 0 or r_idx >= len(regions):
            continue
        reg = regions[r_idx]
        if not reg:
            continue
        # Filter finite, in-bounds loci without a Python inner try/branch forest.
        finite = [v_idx for v_idx in reg if v_idx >= 0 and locus_valid[v_idx]]
        if not finite:
            if fallback_pole is not None:
                targets[i] = fallback_pole
            continue

        C = Vverts[np.asarray(finite, dtype=int)]
        projections = (C - P[i]) @ N[i]
        inner = projections < 0
        if np.any(inner):
            idx = int(np.argmin(np.where(inner, projections, np.inf)))
            pole = C[idx]
        elif fallback_pole is not None:
            # Starlab initializes the pole index to zero if no negative locus is
            # found, which corresponds to the first retained global locus.
            pole = fallback_pole
        else:
            continue

        targets[i] = pole
        d = max(eps, float(np.linalg.norm(pole - P[i])))
        weights[i] = 1.0 / d

    if np.any(weights > 0):
        wmax = float(np.max(weights))
        if wmax > 0:
            weights = weights / wmax

    logger.debug(
        "compute_voronoi_poles: n=%d weighted=%d",
        n,
        int(np.count_nonzero(weights > 0)),
    )
    return targets, weights
