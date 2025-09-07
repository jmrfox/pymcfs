from __future__ import annotations

import logging
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _face_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _edge_lengths(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # edge lengths opposite to vertices 0,1,2
    a = np.linalg.norm(V[F[:, 1]] - V[F[:, 2]], axis=1)
    b = np.linalg.norm(V[F[:, 2]] - V[F[:, 0]], axis=1)
    c = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
    return a, b, c


def _cot_angles_from_edges(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Heron's formula for area
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 1e-32))
    # cot(alpha) opposite edge a, etc. Using 4A in denominator (since |u x v| = 2A)
    cot_alpha = (b * b + c * c - a * a) / (4.0 * area)
    cot_beta = (c * c + a * a - b * b) / (4.0 * area)
    cot_gamma = (a * a + b * b - c * c) / (4.0 * area)
    return cot_alpha, cot_beta, cot_gamma


def _angles_from_edges(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return angles at the triangle vertices given opposite edge lengths.

    For a triangle with vertex indices (i0, i1, i2), we define:
    - a opposite i0 (edge length |v1-v2|)
    - b opposite i1 (edge length |v2-v0|)
    - c opposite i2 (edge length |v0-v1|)
    Returns (alpha, beta, gamma) corresponding to angles at (i0,i1,i2).
    """
    # cos(theta) using law of cosines
    # Clip to avoid numerical issues
    def acos_clipped(x: np.ndarray) -> np.ndarray:
        return np.arccos(np.clip(x, -1.0, 1.0))

    alpha = acos_clipped((b * b + c * c - a * a) / (2.0 * b * c))
    beta = acos_clipped((c * c + a * a - b * b) / (2.0 * c * a))
    gamma = acos_clipped((a * a + b * b - c * c) / (2.0 * a * b))
    return alpha, beta, gamma


def cotangent_laplacian(V: np.ndarray, F: np.ndarray, *, verbose: bool = False) -> sp.csr_matrix:
    """Build symmetric cotangent Laplacian L for a triangle mesh.

    L(i,i) = -sum_{j!=i} L(i,j)
    L(i,j) = -(cot alpha + cot beta)/2 for edge (i,j).

    Parameters
    ----------
    V : (n,3) float array
    F : (m,3) int array (triangles)

    Returns
    -------
    L : (n,n) csr_matrix
    """
    n = V.shape[0]
    if verbose:
        logger.info("Building cotangent Laplacian for %d vertices, %d faces", n, F.shape[0])
    a, b, c = _edge_lengths(V, F)
    cot_a, cot_b, cot_c = _cot_angles_from_edges(a, b, c)

    I = []
    J = []
    W = []

    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

    for (ii, jj, w) in [
        (i1, i2, cot_a),  # opposite i0
        (i2, i0, cot_b),  # opposite i1
        (i0, i1, cot_c),  # opposite i2
    ]:
        I.extend(ii)
        J.extend(jj)
        W.extend(w)
        I.extend(jj)
        J.extend(ii)
        W.extend(w)

    C = sp.coo_matrix((np.array(W), (np.array(I), np.array(J))), shape=(n, n)).tocsr()

    L = -0.5 * C
    diag = -np.array(L.sum(axis=1)).ravel()
    L = L + sp.diags(diag, format="csr")
    if verbose:
        logger.info("Laplacian built: nnz=%d", L.nnz)
    return L


def mean_value_laplacian(V: np.ndarray, F: np.ndarray, *, verbose: bool = False) -> sp.csr_matrix:
    """Build symmetric mean value Laplacian L for a triangle mesh.

    Based on Floater (2003). For an interior edge (i,j) shared by two triangles,
    the weight is w_ij = sum_t tan(theta_t/2) / ||vi - vj||, where theta_t is the
    angle opposite the edge in triangle t. We accumulate contributions per triangle
    and symmetrize. Diagonal is set s.t. row sums are zero.

    Parameters
    ----------
    V : (n,3) float array
    F : (m,3) int array

    Returns
    -------
    L : (n,n) csr_matrix
    """
    n = V.shape[0]
    if verbose:
        logger.info("Building mean value Laplacian for %d vertices, %d faces", n, F.shape[0])

    # Opposite edge lengths a,b,c for vertices (i0,i1,i2)
    a, b, c = _edge_lengths(V, F)
    alpha, beta, gamma = _angles_from_edges(a, b, c)

    # For each triangle, add contributions to its three edges from opposite angles
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

    # Edge lengths for each edge
    len_12 = np.linalg.norm(V[i1] - V[i2], axis=1)  # opposite i0 -> use alpha
    len_20 = np.linalg.norm(V[i2] - V[i0], axis=1)  # opposite i1 -> use beta
    len_01 = np.linalg.norm(V[i0] - V[i1], axis=1)  # opposite i2 -> use gamma

    # Weights per triangle edge: tan(angle/2) / edge_length
    # Small epsilon to avoid division by zero for degenerate edges
    eps = 1e-16
    w_12 = np.tan(alpha / 2.0) / np.maximum(len_12, eps)
    w_20 = np.tan(beta / 2.0) / np.maximum(len_20, eps)
    w_01 = np.tan(gamma / 2.0) / np.maximum(len_01, eps)

    I: list[int] = []
    J: list[int] = []
    W: list[float] = []

    # Accumulate symmetric weights for each triangle edge
    def add_edge(ii: np.ndarray, jj: np.ndarray, ww: np.ndarray) -> None:
        I.extend(ii)
        J.extend(jj)
        W.extend(ww)
        I.extend(jj)
        J.extend(ii)
        W.extend(ww)

    add_edge(i1, i2, w_12)
    add_edge(i2, i0, w_20)
    add_edge(i0, i1, w_01)

    Wmat = sp.coo_matrix((np.array(W), (np.array(I), np.array(J))), shape=(n, n)).tocsr()

    # Symmetrize explicitly
    Wsym = 0.5 * (Wmat + Wmat.T)
    L = -Wsym
    diag = -np.array(L.sum(axis=1)).ravel()
    L = L + sp.diags(diag, format="csr")
    if verbose:
        logger.info("Mean value Laplacian built: nnz=%d", L.nnz)
    return L


def lumped_mass_matrix(V: np.ndarray, F: np.ndarray, *, verbose: bool = False) -> sp.csr_matrix:
    """Lumped (barycentric) mass matrix: M(i,i) = 1/3 sum of incident triangle areas."""
    n = V.shape[0]
    areas = _face_areas(V, F)
    Mdiag = np.zeros(n, dtype=float)
    for k in range(3):
        np.add.at(Mdiag, F[:, k], areas / 3.0)
    M = sp.diags(Mdiag, format="csr")
    if verbose:
        logger.info("Mass matrix built: positive entries=%d", int(np.count_nonzero(Mdiag > 0)))
    return M
