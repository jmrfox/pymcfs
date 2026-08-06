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
