import numpy as np
import scipy.sparse as sp
import trimesh as tm

from pymcfs.laplacian import cotangent_laplacian


def test_cotangent_secure_properties():
    # Use a mesh with a range of triangle shapes
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=3)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)

    L = cotangent_laplacian(V, F, secure=True)

    # CSR
    assert sp.isspmatrix_csr(L)

    # Symmetry
    assert (L - L.T).nnz == 0

    # Row-sum ~ 0
    rowsum = np.array(L.sum(axis=1)).ravel()
    assert np.allclose(rowsum, 0.0, atol=1e-8)

    # Off-diagonal entries should be <= 0
    L_coo = L.tocoo()
    mask_off = L_coo.row != L_coo.col
    off_vals = L_coo.data[mask_off]
    assert np.all(off_vals <= 1e-14)  # allow tiny numerical positive noise
