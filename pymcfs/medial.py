from __future__ import annotations

import numpy as np
import trimesh as tm
from scipy.spatial import Voronoi


def compute_voronoi_poles(mesh: tm.Trimesh, *, use_vertex_normals: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Voronoi poles used by Starlab ``mcfskel``.

    This mirrors ``QhullVoronoiHelper``: retain finite Voronoi loci inside the
    input bounding box, then choose the locus in each vertex's Voronoi cell with
    the most negative projection along the surface normal.

    Returns
    -------
    targets : (n,3) float
        Medial target positions (inner Voronoi poles) for each vertex.
    weights : (n,) float
        Suggested diagonal guidance weights per vertex in ``[0, 1]``.
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

    return targets, weights
