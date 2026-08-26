"""CGAL-style exterior Voronoi pole gating."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh as tm

from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
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


def test_pole_valid_contains_cached_across_geometry_steps(monkeypatch):
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    calls = {"n": 0}
    orig = mesh.contains

    def counting_contains(points):
        calls["n"] += 1
        return orig(points)

    monkeypatch.setattr(mesh, "contains", counting_contains)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.5,
        w_M=5.0,
        gate_exterior_poles=True,
        use_cholmod=False,
        verbose=False,
    )
    n_after_init = calls["n"]
    assert n_after_init >= 1
    mcs.contract_geometry()
    mcs.contract_geometry()
    assert calls["n"] == n_after_init


def test_pole_valid_recomputed_after_remesh_marks_dirty(monkeypatch):
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    calls = {"n": 0}
    orig = mesh.contains

    def counting_contains(points):
        calls["n"] += 1
        return orig(points)

    monkeypatch.setattr(mesh, "contains", counting_contains)
    mcs = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.5,
        w_M=5.0,
        gate_exterior_poles=True,
        use_cholmod=False,
        verbose=False,
    )
    n0 = calls["n"]
    mcs._mark_poles_dirty()
    mcs._sync_pole_valid()
    assert calls["n"] == n0 + 1


@pytest.mark.slow
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
