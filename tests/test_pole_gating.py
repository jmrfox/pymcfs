"""CGAL-style exterior Voronoi pole gating."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh as tm

from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.medial import points_inside_mesh
from pymcfs.quality import analyze_skeleton
from pymcfs.skeleton import resolve_mcfs_profile, skeletonize

ROOT = Path(__file__).resolve().parents[1]
TS1 = ROOT / "data" / "mesh" / "TS1.obj"


def test_resolve_mcfs_profile_defaults():
    wh, wm, gate = resolve_mcfs_profile(None, w_H=0.5, w_M=5.0, gate_exterior_poles=None)
    assert (wh, wm, gate) == (0.5, 5.0, True)

    wh, wm, gate = resolve_mcfs_profile(
        "starlab", w_H=0.5, w_M=5.0, gate_exterior_poles=None
    )
    assert (wh, wm, gate) == (0.1, 0.2, False)

    wh, wm, gate = resolve_mcfs_profile(
        "starlab", w_H=0.3, w_M=1.0, gate_exterior_poles=True
    )
    assert (wh, wm, gate) == (0.3, 1.0, True)


def test_gate_exterior_poles_zeros_wM_for_outside_poles():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.5,
        w_M=5.0,
        gate_exterior_poles=True,
        verbose=False,
    )
    mcs.poles = mcs.V.copy()
    mcs.poles[0] = mcs.V[0] + np.array([100.0, 0.0, 0.0])
    mcs.pole_valid = mcs._compute_pole_valid(mcs.poles)
    assert not bool(mcs.pole_valid[0])
    assert bool(mcs.pole_valid[1:].all()) or int((~mcs.pole_valid).sum()) >= 1

    _, _, wM = mcs._update_constraint_weights()
    assert float(wM[0]) == 0.0
    # At least one interior pole keeps full medial weight.
    assert float(np.max(wM)) == pytest.approx(5.0)


def test_ungated_keeps_wM_for_outside_poles():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.1,
        w_M=0.2,
        gate_exterior_poles=False,
        verbose=False,
    )
    mcs.poles = mcs.V.copy()
    mcs.poles[0] = mcs.V[0] + np.array([100.0, 0.0, 0.0])
    mcs.pole_valid = mcs._compute_pole_valid(mcs.poles)
    _, _, wM = mcs._update_constraint_weights()
    assert float(wM[0]) == pytest.approx(0.2)


def _record_gating(monkeypatch, sizes: list[int]) -> None:
    """Record the batch size of every pole containment test the driver runs."""
    from pymcfs import mcfs as mcfs_mod

    original = mcfs_mod.points_inside_mesh

    def recording(mesh, points, *, fast=False):
        sizes.append(int(np.asarray(points).shape[0]))
        return original(mesh, points, fast=fast)

    monkeypatch.setattr(mcfs_mod, "points_inside_mesh", recording)


def test_pole_valid_contains_cached_across_geometry_steps(monkeypatch):
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    sizes: list[int] = []
    _record_gating(monkeypatch, sizes)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.5,
        w_M=5.0,
        gate_exterior_poles=True,
        use_cholmod=False,
        verbose=False,
    )
    assert len(sizes) >= 1
    n_after_init = len(sizes)
    mcs.contract_geometry()
    mcs.contract_geometry()
    assert len(sizes) == n_after_init


def test_collapse_carries_pole_valid_without_contains(monkeypatch):
    """Collapse keeps one of two existing poles, so validity carries by index."""
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    sizes: list[int] = []
    _record_gating(monkeypatch, sizes)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.5,
        w_M=5.0,
        gate_exterior_poles=True,
        use_cholmod=False,
        verbose=False,
    )
    diag = float(np.linalg.norm(mcs.V.max(axis=0) - mcs.V.min(axis=0)))
    mcs._min_edge = 0.15 * diag
    sizes.clear()

    assert mcs.collapse_edges() > 0
    assert sizes == []
    assert mcs.pole_valid.shape[0] == mcs.V.shape[0]
    assert np.array_equal(mcs.pole_valid, points_inside_mesh(mesh, mcs.poles))


def test_remesh_only_tests_newly_interpolated_poles(monkeypatch):
    """Every post-init containment batch covers new split poles only."""
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    sizes: list[int] = []
    _record_gating(monkeypatch, sizes)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.1,
        w_M=5.0,
        gate_exterior_poles=True,
        use_cholmod=False,
        verbose=False,
    )
    sizes.clear()

    for _ in range(4):
        mcs.contract()

    # No batch may cover the whole mesh: that would be a full re-test.
    assert all(size < mcs.V.shape[0] for size in sizes)
    assert np.array_equal(mcs.pole_valid, points_inside_mesh(mesh, mcs.poles))


def test_gating_defaults_to_exact_float64_backend():
    """Gating must not silently switch to Embree just because it is installed.

    Embree traces in single precision, which flips most pole decisions on
    meshes far from the origin, so the exact traverser stays the default.
    """
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh, w_H=0.5, w_M=5.0, use_cholmod=False, verbose=False
    )
    assert mcs.fast_gating is False

    # Same containment whether the mesh sits at the origin or far from it.
    offset = np.array([6000.0, 1200.0, 3400.0])
    far = tm.Trimesh(
        vertices=np.asarray(mesh.vertices) + offset,
        faces=np.asarray(mesh.faces),
        process=False,
    )
    probes = np.asarray(mesh.vertices) * 0.5
    assert np.array_equal(
        points_inside_mesh(mesh, probes),
        points_inside_mesh(far, probes + offset),
    )


def test_pole_valid_recomputed_after_marking_dirty(monkeypatch):
    """``_mark_poles_dirty`` remains an escape hatch to a full re-test."""
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    sizes: list[int] = []
    _record_gating(monkeypatch, sizes)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.5,
        w_M=5.0,
        gate_exterior_poles=True,
        use_cholmod=False,
        verbose=False,
    )
    sizes.clear()
    mcs._mark_poles_dirty()
    mcs._sync_pole_valid()
    assert sizes == [mcs.V.shape[0]]


@pytest.mark.e2e
def test_ts1_robust_no_screenshot_exterior_spikes():
    """TS1 with robust defaults should not grow the known exterior spike region."""
    if not TS1.is_file():
        pytest.skip("data/mesh/TS1.obj not present")
    mesh = tm.load(str(TS1), force="mesh", process=False)
    if not isinstance(mesh, tm.Trimesh):
        mesh = mesh.dump(concatenate=True)
    mesh = tm.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    if not mesh.is_watertight:
        pytest.skip("TS1 mesh is not watertight after load")

    skel = skeletonize(
        mesh,
        max_iterations=500,
        timeout_seconds=180.0,
        refine=False,
        verbose=False,
    )
    nodes = np.asarray(skel.nodes, dtype=float)
    region = (nodes[:, 0] > 7400.0) & (nodes[:, 2] > 3300.0)
    if region.any():
        inside = np.asarray(mesh.contains(nodes[region]), dtype=bool)
        n_out = int((~inside).sum())
    else:
        n_out = 0
    assert n_out == 0, (
        f"expected 0 exterior nodes in screenshot region x>7400,z>3300; got {n_out}"
    )

    report = analyze_skeleton(mesh, skel)
    assert report.nodes_inside_frac is not None
    assert report.nodes_inside_frac >= 0.95
