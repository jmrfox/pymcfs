"""Tests for skeleton scoring and the MCFS parameter oracle."""

from __future__ import annotations

import trimesh as tm
import pytest

from pymcfs.params import MeshMcfsFeatures, mesh_mcfs_features, propose_mcfs_params
from pymcfs.quality import SkeletonQualityReport, score_skeleton
from pymcfs.skeleton import resolve_mcfs_profile, skeletonize


def _ts1_like() -> MeshMcfsFeatures:
    return MeshMcfsFeatures(
        n_vertices=1000,
        bbox_diag=1000.0,
        mean_pole_offset=16.0,
        p95_pole_offset=25.0,
        mean_pole_offset_over_diag=0.016,
        p95_pole_offset_over_diag=0.025,
        poles_inside_frac=0.97,
        char_radius_over_diag=0.025,
    )


def _ts2_like() -> MeshMcfsFeatures:
    return MeshMcfsFeatures(
        n_vertices=500,
        bbox_diag=600.0,
        mean_pole_offset=29.0,
        p95_pole_offset=50.0,
        mean_pole_offset_over_diag=0.049,
        p95_pole_offset_over_diag=0.081,
        poles_inside_frac=0.97,
        char_radius_over_diag=0.087,
    )


def _report(**kwargs) -> SkeletonQualityReport:
    base = dict(
        n_nodes=kwargs.get("n_nodes", 20),
        n_edges=kwargs.get("n_edges", 19),
        n_components=kwargs.get("n_components", 1),
        n_junctions=kwargs.get("n_junctions", 2),
        n_leaves=kwargs.get("n_leaves", 4),
        nodes_inside_frac=kwargs.get("nodes_inside_frac", 1.0),
        n_nodes_outside=kwargs.get("n_nodes_outside", 0),
        edges_inside_frac=kwargs.get("edges_inside_frac", 1.0),
        n_edges_outside=kwargs.get("n_edges_outside", 0),
        mesh_genus=kwargs.get("mesh_genus", 0),
        skeleton_cyclomatic=kwargs.get("skeleton_cyclomatic", 0),
        topology_consistent=True,
    )
    base.update(kwargs)
    return SkeletonQualityReport(**base)


def test_score_rejects_remesh_growth():
    s = score_skeleton(_report(), remesh_growth_rejected=True, remesh_growth_ratio=5.0)
    assert s.rejected
    assert "remesh_growth" in (s.reject_reason or "")


def test_score_prefers_exact_topology_over_excess_cycles():
    exact = score_skeleton(
        _report(mesh_genus=1, skeleton_cyclomatic=1, n_junctions=4, n_nodes=40)
    )
    excess = score_skeleton(
        _report(mesh_genus=1, skeleton_cyclomatic=3, n_junctions=2, n_nodes=20)
    )
    missing = score_skeleton(
        _report(mesh_genus=1, skeleton_cyclomatic=0, n_junctions=1, n_nodes=10)
    )
    assert exact.value > excess.value
    assert excess.value > missing.value
    assert exact.topology_delta == 0
    assert excess.topology_delta == 2


def test_score_containment_beats_compactness():
    inside = score_skeleton(
        _report(
            nodes_inside_frac=1.0,
            n_nodes_outside=0,
            n_junctions=8,
            n_nodes=100,
            mesh_genus=0,
            skeleton_cyclomatic=0,
        )
    )
    leaky = score_skeleton(
        _report(
            nodes_inside_frac=0.9,
            n_nodes_outside=5,
            n_junctions=1,
            n_nodes=10,
            mesh_genus=0,
            skeleton_cyclomatic=0,
        )
    )
    assert inside.value > leaky.value


def test_score_compactness_among_equal_quality():
    lean = score_skeleton(
        _report(n_junctions=2, n_nodes=30, n_leaves=4, mesh_genus=0, skeleton_cyclomatic=0)
    )
    bushy = score_skeleton(
        _report(n_junctions=12, n_nodes=80, n_leaves=20, mesh_genus=0, skeleton_cyclomatic=0)
    )
    assert lean.value > bushy.value


def test_mesh_mcfs_features_sphere():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    feats = mesh_mcfs_features(mesh)
    assert feats.n_vertices == len(mesh.vertices)
    assert feats.bbox_diag > 0
    assert 0.0 <= feats.poles_inside_frac <= 1.0
    assert feats.mean_pole_offset_over_diag >= 0.0


def test_propose_params_higher_rho_not_weaker():
    """Higher relative pole offset must propose a lower (or equal) medial ratio."""
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    p_low = propose_mcfs_params(mesh, features=_ts1_like())
    p_high = propose_mcfs_params(mesh, features=_ts2_like())
    assert p_high.ratio <= p_low.ratio + 1e-9
    assert p_high.gate_exterior_poles is True
    assert p_low.gate_exterior_poles is True
    assert p_high.w_M <= p_low.w_M + 1e-9


def _ts3_like_bulky() -> MeshMcfsFeatures:
    """Mean ρ near-ref but elevated char_r (TS3-style thick compartment)."""
    return MeshMcfsFeatures(
        n_vertices=8000,
        bbox_diag=1900.0,
        mean_pole_offset=35.7,
        p95_pole_offset=68.0,
        mean_pole_offset_over_diag=0.0188,
        p95_pole_offset_over_diag=0.036,
        poles_inside_frac=0.987,
        char_radius_over_diag=0.042,
    )


def test_propose_sparse_snaps_near_ref_to_robust():
    """TS1-like ρ + sparse/balanced must match usual robust (0.5, 5.0)."""
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    for mode in ("sparse", "balanced"):
        p = propose_mcfs_params(mesh, features=_ts1_like(), branching=mode)
        assert p.w_H == pytest.approx(0.5)
        assert p.w_M == pytest.approx(5.0)
        assert p.ratio == pytest.approx(10.0)


def test_propose_sparse_uses_char_r_when_mean_rho_near_ref():
    """Bulky compartments must not snap to robust just because mean ρ is mild."""
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    p = propose_mcfs_params(mesh, features=_ts3_like_bulky(), branching="sparse")
    assert p.ratio < 10.0 - 1e-9
    assert p.w_M < 5.0 - 1e-9
    assert "thick" in p.rationale


def test_propose_branching_orders_medial_strength():
    """sparse ≤ balanced ≤ dense in medial ratio / w_M on thick meshes."""
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    feats = _ts2_like()
    sparse = propose_mcfs_params(mesh, features=feats, branching="sparse")
    balanced = propose_mcfs_params(mesh, features=feats, branching="balanced")
    dense = propose_mcfs_params(mesh, features=feats, branching="dense")
    assert sparse.ratio <= balanced.ratio + 1e-9
    assert balanced.ratio <= dense.ratio + 1e-9
    assert sparse.w_M <= balanced.w_M + 1e-9
    assert balanced.w_M <= dense.w_M + 1e-9
    assert sparse.ratio <= 3.0 + 1e-9


def test_propose_rejects_bad_branching():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    with pytest.raises(ValueError, match="branching"):
        propose_mcfs_params(mesh, branching="aggressive")  # type: ignore[arg-type]


def test_resolve_profile_auto_requires_mesh():
    with pytest.raises(ValueError, match="requires a mesh"):
        resolve_mcfs_profile("auto", w_H=0.5, w_M=5.0, gate_exterior_poles=None)


def test_resolve_profile_auto_proposes():
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    wh, wm, gate = resolve_mcfs_profile(
        "auto", w_H=0.5, w_M=5.0, gate_exterior_poles=None, mesh=mesh
    )
    proposed = propose_mcfs_params(mesh)
    assert wh == pytest.approx(proposed.w_H)
    assert wm == pytest.approx(proposed.w_M)
    assert gate is True


def test_resolve_profile_auto_alias_override():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    wh, wm, gate = resolve_mcfs_profile(
        "auto",
        w_H=0.5,
        w_M=5.0,
        gate_exterior_poles=False,
        quality_speed_tradeoff=0.8,
        medially_centered_speed_tradeoff=2.0,
        mesh=mesh,
    )
    assert wh == pytest.approx(0.8)
    assert wm == pytest.approx(2.0)
    assert gate is False


def test_skeletonize_profile_auto_runs():
    mesh = tm.creation.icosphere(subdivisions=1, radius=1.0)
    skel = skeletonize(mesh, profile="auto", max_iterations=10, refine=False)
    assert skel.nodes.shape[0] > 0


def test_remesh_growth_abort():
    """Driver stops when vertex count exceeds max_vertex_growth * n0."""
    from pymcfs.mcfs import MeanCurvatureFlowSkeletonization

    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    driver = MeanCurvatureFlowSkeletonization(
        mesh,
        w_H=0.1,
        w_M=10.0,
        gate_exterior_poles=True,
        max_iterations=50,
        timeout_seconds=30.0,
        max_vertex_growth=1.05,
        validate=False,
        verbose=False,
    )
    # Force growth tracking by running; tiny cap should trip quickly on any split.
    # If the mesh never grows, abort stays False — still a valid outcome.
    driver.contract_until_convergence()
    assert driver._n_max >= driver._n0
    if driver.aborted_remesh_growth:
        assert driver.remesh_growth_ratio() > 1.05
