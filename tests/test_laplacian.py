import numpy as np
import scipy.sparse as sp
import trimesh as tm

from pymcfs.laplacian import cotangent_laplacian, mean_value_laplacian, lumped_mass_matrix


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
