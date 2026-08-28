import numpy as np
import trimesh as tm
from pathlib import Path

from pymcfs.mesh import MeshManager
from pymcfs.validate import validate_mcfs_mesh

DATA = Path(__file__).resolve().parents[1] / "data" / "mesh"


def test_mesh_manager_analyze_sphere():
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    mm = MeshManager(mesh)
    analysis = mm.analyze_mesh()

    assert analysis["vertex_count"] == len(mesh.vertices)
    assert analysis["face_count"] == len(mesh.faces)
    assert analysis["is_watertight"] is True
    # For a sphere, euler = 2, genus = 0
    assert analysis.get("euler_characteristic") in (2,)
    assert analysis.get("genus") in (0,)


def test_center_and_scale():
    mesh = tm.primitives.Sphere(radius=1.0, subdivisions=2)
    mm = MeshManager(mesh)

    # Center by centroid
    mm.center_mesh("centroid")
    centered = mm.to_trimesh()
    assert np.allclose(centered.centroid, np.zeros(3), atol=1e-6)

    # Uniform scale x2
    before_extent = centered.bounding_box.extents.copy()
    mm.scale_mesh(2.0)
    after_extent = mm.to_trimesh().bounding_box.extents
    assert np.allclose(after_extent, before_extent * 2.0, rtol=1e-6, atol=1e-8)


def test_load_mesh_process_false_preserves_watertight():
    """Trimesh process=True can break closed TS OBJs; MeshManager must not."""
    path = DATA / "TS3.obj"
    if not path.is_file():
        import pytest

        pytest.skip("TS3.obj not present")

    processed = tm.load(str(path), force="mesh", process=True)
    assert processed.is_watertight is False

    mm = MeshManager(verbose=False)
    mesh = mm.load_mesh(str(path), validate_mcfs=True)
    assert mesh.is_watertight is True
    validate_mcfs_mesh(mesh)
    assert len(mesh.vertices) == len(tm.load(str(path), force="mesh", process=False).vertices)
