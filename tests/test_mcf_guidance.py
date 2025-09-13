import numpy as np
import trimesh as tm

from pymcfs.mcf import mean_curvature_flow


def test_guidance_original_reduces_displacement():
    # Sphere is closed/manifold; good for MCF
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    V0 = np.asarray(mesh.vertices)

    # Unguided MCF
    res_unguided = mean_curvature_flow(
        mesh,
        dt=2e-2,
        iterations=10,
        laplacian_type="cotangent",
        laplacian_secure=True,
    )
    V_ung = res_unguided.vertices

    # Guided toward original positions
    res_guided = mean_curvature_flow(
        mesh,
        dt=2e-2,
        iterations=10,
        laplacian_type="cotangent",
        laplacian_secure=True,
        guidance_type="original",
        guidance_weight=0.5,
    )
    V_guid = res_guided.vertices

    disp_ung = np.linalg.norm(V_ung - V0, axis=1).mean()
    disp_guid = np.linalg.norm(V_guid - V0, axis=1).mean()

    # With pull-to-original guidance, displacement should be reduced
    assert disp_guid < disp_ung
