import numpy as np
import trimesh as tm
from pathlib import Path

from pymcfs.mesh import MeshManager
from pymcfs.validate import load_and_repair, validate_mcfs_mesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "toric_spines" / "data" / "mesh"
# Legacy path used by older checkouts / optional local copies.
LEGACY_DATA = ROOT / "data" / "mesh"


def _ts_path(name: str) -> Path:
    for base in (DATA, LEGACY_DATA):
        path = base / name
        if path.is_file():
            return path
    return DATA / name


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
    path = _ts_path("TS3.obj")
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


def test_repair_mesh_preserves_watertight_ts():
    """repair_mesh must not weld coincident verts and destroy manifold topology."""
    import pytest

    path = _ts_path("TS21.obj")
    if not path.is_file():
        pytest.skip("TS21.obj not present")

    processed = tm.load(str(path), force="mesh", process=True)
    assert processed.is_watertight is False

    mm = MeshManager(verbose=False)
    mesh = mm.load_mesh(str(path), validate_mcfs=True)
    n0 = len(mesh.vertices)
    repaired = mm.repair_mesh(verbose=False)
    assert repaired.is_watertight is True
    assert len(repaired.vertices) == n0
    validate_mcfs_mesh(repaired)


def test_load_and_repair_skips_ready_mesh():
    """Already-valid meshes must not be mutated by load_and_repair."""
    import pytest

    path = _ts_path("TS3.obj")
    if not path.is_file():
        pytest.skip("TS3.obj not present")

    raw = tm.load(str(path), force="mesh", process=False)
    n0, f0 = len(raw.vertices), len(raw.faces)
    out = load_and_repair(str(path))
    assert out.is_watertight is True
    assert len(out.vertices) == n0
    assert len(out.faces) == f0
    validate_mcfs_mesh(out)
