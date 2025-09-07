import numpy as np
import trimesh as tm

from pymcfs.mcf import mean_curvature_flow


def test_mcf_shrinks_sphere_volume():
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    vol_before = mesh.volume
    res = mean_curvature_flow(mesh, dt=2e-2, iterations=10)
    V2 = res.vertices
    mesh2 = tm.Trimesh(vertices=V2, faces=mesh.faces, process=False)
    vol_after = mesh2.volume
    assert vol_after < vol_before


def test_mcf_shape_dimensions():
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=1)
    res = mean_curvature_flow(mesh, dt=1e-2, iterations=1)
    assert res.vertices.shape == mesh.vertices.shape
