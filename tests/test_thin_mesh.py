import numpy as np
import trimesh as tm

from pymcfs.skeleton import thin_mesh


def _no_degenerate_faces(F: np.ndarray) -> bool:
    if F.size == 0:
        return True
    if F.ndim != 2 or F.shape[1] != 3:
        return False
    return np.all((F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0]))


def test_thin_mesh_basic():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    Vt, Ft = thin_mesh(mesh, mcf_iters=15, is_medially_centered=False, omega_P=0.0)
    assert isinstance(Vt, np.ndarray) and Vt.ndim == 2 and Vt.shape[1] == 3
    assert isinstance(Ft, np.ndarray) and Ft.ndim == 2 and Ft.shape[1] == 3
    assert _no_degenerate_faces(Ft)
    assert Vt.shape[0] <= mesh.vertices.shape[0]


def test_thin_mesh_pq_heap_mode_runs():
    # Legacy collapse_mode kwarg is ignored; ensure call still works
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    Vt, Ft = thin_mesh(
        mesh,
        mcf_iters=10,
        is_medially_centered=False,
        omega_P=0.0,
        collapse_mode="pq_heap",
        collapse_ratio=0.2,
    )
    assert Vt.ndim == 2 and Ft.ndim == 2
    assert _no_degenerate_faces(Ft)


def test_thin_mesh_medial_protect_voronoi_runs():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    Vt, Ft = thin_mesh(
        mesh,
        mcf_iters=10,
        guidance_type="voronoi",
        omega_P=0.2,
        medial_protect=True,
    )
    assert Vt.shape[0] > 0
    assert _no_degenerate_faces(Ft)
