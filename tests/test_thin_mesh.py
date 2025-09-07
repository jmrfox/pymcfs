import numpy as np
import trimesh as tm

from pymcfs.skeleton import thin_mesh


def _no_degenerate_faces(F: np.ndarray) -> bool:
    if F.size == 0:
        return True
    # Faces must be (n,3) with distinct indices per face
    if F.ndim != 2 or F.shape[1] != 3:
        return False
    return np.all((F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0]))


def test_thin_mesh_basic():
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    Vt, Ft = thin_mesh(
        mesh,
        mcf_dt=1e-2,
        mcf_iters=5,
        collapse_passes=1,
        collapse_mode="pq",
        collapse_ratio=0.2,
    )

    assert isinstance(Vt, np.ndarray) and Vt.ndim == 2 and Vt.shape[1] == 3
    assert isinstance(Ft, np.ndarray) and Ft.ndim == 2 and Ft.shape[1] == 3
    assert Ft.shape[0] >= 0
    assert _no_degenerate_faces(Ft)
    # Should not increase vertex count
    assert Vt.shape[0] <= mesh.vertices.shape[0]


def test_thin_mesh_pq_heap_mode_runs():
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    Vt, Ft = thin_mesh(
        mesh,
        mcf_dt=1e-2,
        mcf_iters=5,
        collapse_passes=1,
        collapse_mode="pq_heap",
        collapse_ratio=0.2,
    )

    assert Vt.ndim == 2 and Vt.shape[1] == 3
    assert Ft.ndim == 2 and Ft.shape[1] == 3
    assert _no_degenerate_faces(Ft)


def test_thin_mesh_medial_protect_voronoi_runs():
    # Voronoi guidance + medial protect should still run and produce a valid mesh
    mesh = tm.primitives.Cylinder(radius=0.6, height=2.0, sections=48)
    Vt, Ft = thin_mesh(
        mesh,
        mcf_dt=1e-2,
        mcf_iters=8,
        guidance_type="voronoi",
        guidance_weight=1.0,
        collapse_passes=1,
        medial_protect=True,
        medial_protect_threshold=0.5,
        collapse_mode="pq",
        collapse_ratio=0.2,
    )

    assert Vt.ndim == 2 and Vt.shape[1] == 3
    assert Ft.ndim == 2 and Ft.shape[1] == 3
    assert _no_degenerate_faces(Ft)
