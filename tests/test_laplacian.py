import numpy as np
import scipy.sparse as sp
import trimesh as tm

from pymcfs.laplacian import (
    cotangent_laplacian,
    lumped_mass_matrix,
    mean_value_laplacian,
    mcfs_cotangent_laplacian,
)


def test_laplacian_basic_properties():
    # Create a simple sphere mesh
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)

    L = cotangent_laplacian(V, F)
    assert sp.isspmatrix_csr(L)

    # Row-sum should be ~0
    rowsum = np.array(L.sum(axis=1)).ravel()
    assert np.allclose(rowsum, 0.0, atol=1e-8)

    # Symmetry
    assert (L - L.T).nnz == 0

    # Mass matrix diagonal positive
    M = lumped_mass_matrix(V, F)
    mdiag = M.diagonal()
    assert np.all(mdiag > 0)


def test_mean_value_laplacian_properties():
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)

    L = mean_value_laplacian(V, F)
    # CSR and symmetric
    assert sp.isspmatrix_csr(L)
    assert (L - L.T).nnz == 0
    # Row-sum should be ~0
    rowsum = np.array(L.sum(axis=1)).ravel()
    assert np.allclose(rowsum, 0.0, atol=1e-8)


def test_mcfs_cotangent_laplacian_matches_reference_scale():
    # Regular tetrahedron: every opposite angle is 60 degrees, so Starlab's
    # unhalved interior-edge weight is 2*cot(60) = 2/sqrt(3).
    V = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    L = mcfs_cotangent_laplacian(V, F)
    expected_weight = 2.0 / np.sqrt(3.0)
    dense = L.toarray()
    assert np.allclose(dense[np.triu_indices(4, k=1)], expected_weight)
    assert np.allclose(np.diag(dense), -3.0 * expected_weight)
    assert np.allclose(np.asarray(L.sum(axis=1)).ravel(), 0.0)


def _mcfs_cotangent_laplacian_reference(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """Pre-Tier-A assembly (np.unique + add.at) kept for regression forever."""
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)
    n = V.shape[0]
    if F.size == 0:
        return sp.csr_matrix((n, n), dtype=float)

    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

    def cotangent_at(center: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        u = V[a] - V[center]
        v = V[b] - V[center]
        un = np.linalg.norm(u, axis=1)
        vn = np.linalg.norm(v, axis=1)
        denom = np.maximum(un * vn, 1e-30)
        cosine = np.einsum("ij,ij->i", u, v) / denom
        cosine = np.clip(cosine, -0.999, 0.999)
        return cosine / np.sqrt(np.maximum(1.0 - cosine * cosine, 1e-30))

    cot0 = cotangent_at(i0, i1, i2)
    cot1 = cotangent_at(i1, i2, i0)
    cot2 = cotangent_at(i2, i0, i1)
    edges = np.vstack(
        [
            np.column_stack([i1, i2]),
            np.column_stack([i2, i0]),
            np.column_stack([i0, i1]),
        ]
    )
    edges.sort(axis=1)
    contributions = np.concatenate([cot0, cot1, cot2])
    unique_edges, inverse = np.unique(edges, axis=0, return_inverse=True)
    weights = np.zeros(unique_edges.shape[0], dtype=float)
    np.add.at(weights, inverse, contributions)
    weights = np.maximum(weights, 0.0)
    a, b = unique_edges[:, 0], unique_edges[:, 1]
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    data = np.concatenate([weights, weights])
    offdiag = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    diagonal = -np.asarray(offdiag.sum(axis=1)).ravel()
    return offdiag + sp.diags(diagonal, format="csr")


def test_mcfs_cotangent_laplacian_matches_unique_reference():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    L_new = mcfs_cotangent_laplacian(V, F)
    L_ref = _mcfs_cotangent_laplacian_reference(V, F)
    assert np.allclose(L_new.toarray(), L_ref.toarray(), atol=1e-12, rtol=1e-12)
