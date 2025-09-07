from __future__ import annotations

import numpy as np
import trimesh as tm
from scipy.spatial import Voronoi


def compute_voronoi_poles(mesh: tm.Trimesh, *, use_vertex_normals: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Approximate Voronoi-pole medial targets for each vertex of a mesh.

    For each input vertex p with normal n, we compute the Voronoi region of p
    in 3D and select the farthest finite Voronoi vertex lying in the -n direction
    ("inner" pole). If none lie in -n, we select the farthest finite vertex.

    The returned target is the midpoint between p and the chosen pole, which
    approximates a medial position. A per-vertex weight is set inversely to
    the distance to the pole (clipped) to avoid over-weighting very close poles.

    Returns
    -------
    targets : (n,3) float
        Medial target positions for each vertex.
    weights : (n,) float
        Suggested diagonal guidance weights per vertex.
    """
    if not isinstance(mesh, tm.Trimesh):
        raise TypeError("compute_voronoi_poles expects a trimesh.Trimesh")

    P = np.asarray(mesh.vertices, dtype=float)
    n = P.shape[0]
    if n == 0:
        return P.copy(), np.zeros((0,), dtype=float)

    # Vertex normals (unit)
    if use_vertex_normals and mesh.vertex_normals is not None:
        N = np.asarray(mesh.vertex_normals, dtype=float)
        # Normalize just in case
        norms = np.linalg.norm(N, axis=1)
        norms = np.where(norms > 0, norms, 1.0)
        N = N / norms[:, None]
    else:
        # Fallback: approximate normals via PCA of neighborhood is overkill; use face normals average (trimesh default)
        N = mesh.vertex_normals
        norms = np.linalg.norm(N, axis=1)
        norms = np.where(norms > 0, norms, 1.0)
        N = N / norms[:, None]

    # Build 3D Voronoi diagram of point set
    vor = Voronoi(P)

    targets = P.copy()
    weights = np.zeros(n, dtype=float)

    eps = 1e-9
    Vverts = vor.vertices  # (m,3)
    point_region = vor.point_region
    regions = vor.regions

    for i in range(n):
        r_idx = point_region[i]
        if r_idx == -1 or r_idx >= len(regions):
            continue
        reg = regions[r_idx]
        # Filter out infinite regions
        if reg is None or len(reg) == 0:
            continue
        if any(v_idx == -1 for v_idx in reg):
            # Keep only finite vertices
            finite = [v_idx for v_idx in reg if v_idx != -1 and 0 <= v_idx < len(Vverts)]
            if len(finite) == 0:
                continue
            reg = finite
        else:
            reg = [v_idx for v_idx in reg if 0 <= v_idx < len(Vverts)]
            if len(reg) == 0:
                continue

        C = Vverts[np.asarray(reg, dtype=int)]  # candidate Voronoi vertices
        diffs = C - P[i]
        dists = np.linalg.norm(diffs, axis=1)
        # Inner direction (opposite normal)
        dots = diffs @ N[i]
        inner = np.where(dots < 0)[0]
        if inner.size > 0:
            idx = inner[np.argmax(dists[inner])]
        else:
            idx = int(np.argmax(dists))
        pole = C[idx]

        # Target: midpoint between point and its (inner) pole
        tgt = 0.5 * (P[i] + pole)
        targets[i] = tgt
        # Weight inversely proportional to distance to pole, capped
        d = max(eps, float(np.linalg.norm(pole - P[i])))
        weights[i] = 1.0 / d

    # Normalize weights to a reasonable scale
    if np.any(weights > 0):
        wmax = np.max(weights)
        if wmax > 0:
            weights = weights / wmax  # in [0,1]

    return targets, weights
