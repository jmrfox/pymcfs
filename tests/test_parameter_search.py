"""Tests for MCFS parameter search."""
from __future__ import annotations

from unittest.mock import patch

import networkx as nx
import numpy as np
import trimesh as tm

from pymcfs.mesh import example_mesh
from pymcfs.quality import SkeletonQualityReport
from pymcfs.search import (
    McfsSearchResult,
    _weight_candidates,
    score_skeleton_candidate,
    search_mcfs_params,
)
from pymcfs.skeleton import Skeleton, skeletonize


def _base_report(**kwargs) -> SkeletonQualityReport:
    defaults = dict(
        n_nodes=10,
        n_edges=9,
        n_components=1,
        n_junctions=1,
        n_leaves=2,
        nodes_inside_frac=1.0,
        n_nodes_outside=0,
        edges_inside_frac=1.0,
        n_edges_outside=0,
        mesh_genus=0,
        skeleton_cyclomatic=0,
        topology_consistent=True,
    )
    defaults.update(kwargs)
    return SkeletonQualityReport(**defaults)


def test_weight_candidates_clamped_and_unique():
    cands = _weight_candidates(0.5, 5.0, max_contracts=4)
    assert len(cands) >= 2
    assert len(cands) <= 4
    assert cands[0] == (0.5, 5.0)
    for wh, wm in cands:
        assert 0.25 <= wh <= 2.0
        assert 1.5 <= wm / wh <= 20.0 + 1e-9
    keys = {(round(a, 6), round(b, 6)) for a, b in cands}
    assert len(keys) == len(cands)


def test_deep_tip_penalty_lowers_score():
    """A tip deep inside the volume should score worse than a near-surface tip."""
    mesh = example_mesh("cylinder", radius=0.5, height=2.0, sections=24)
    # Deep tip near axis center vs tip near end-cap.
    G_deep = nx.Graph()
    G_deep.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G_deep.add_node(1, pos=np.array([0.0, 0.0, 0.1]))
    G_deep.add_edge(0, 1)
    skel_deep = Skeleton.from_graph(G_deep)

    G_near = nx.Graph()
    G_near.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G_near.add_node(1, pos=np.array([0.0, 0.0, 0.95]))
    G_near.add_edge(0, 1)
    skel_near = Skeleton.from_graph(G_near)

    report = _base_report(n_nodes=2, n_edges=1, n_junctions=0, n_leaves=2)
    deep = score_skeleton_candidate(report, mesh, skel_deep, tip_clearance_frac=0.01)
    near = score_skeleton_candidate(report, mesh, skel_near, tip_clearance_frac=0.01)
    assert near.value > deep.value


def test_thick_hub_leaf_excess_penalty():
    """Extra leaf arms at a thick hub should lower the search score."""
    # Large ball so hub at origin is thick; leaf tips near surface.
    mesh = tm.creation.icosphere(subdivisions=2, radius=1.0)
    hub = np.array([0.0, 0.0, 0.0])
    tips = [
        np.array([0.9, 0.0, 0.0]),
        np.array([-0.9, 0.0, 0.0]),
        np.array([0.0, 0.9, 0.0]),
        np.array([0.0, -0.9, 0.0]),
    ]
    G = nx.Graph()
    G.add_node(0, pos=hub)
    for i, t in enumerate(tips, start=1):
        G.add_node(i, pos=t)
        G.add_edge(0, i)
    skel = Skeleton.from_graph(G)
    report = _base_report(n_nodes=5, n_edges=4, n_junctions=1, n_leaves=4)

    keep2 = score_skeleton_candidate(
        report,
        mesh,
        skel,
        keep_hub_branches=2,
        hub_degree_min=4,
        hub_radius_frac=0.01,
    )
    keep4 = score_skeleton_candidate(
        report,
        mesh,
        skel,
        keep_hub_branches=4,
        hub_degree_min=4,
        hub_radius_frac=0.01,
    )
    assert keep4.value > keep2.value


def test_skeletonize_parameter_search_cylinder():
    mesh = example_mesh("cylinder", radius=0.5, height=2.0, sections=16)
    skel = skeletonize(
        mesh,
        profile="robust",
        parameter_search=True,
        max_search_contracts=2,
        max_iterations=80,
        timeout_seconds=60.0,
        validate=False,
    )
    assert isinstance(skel, Skeleton)
    assert skel.nodes.shape[0] > 0


def test_skeletonize_default_no_search():
    mesh = example_mesh("cylinder", radius=0.5, height=1.5, sections=12)
    skel = skeletonize(
        mesh,
        profile="robust",
        parameter_search=False,
        max_iterations=60,
        timeout_seconds=45.0,
        validate=False,
    )
    assert skel.nodes.shape[0] > 0


def test_search_mcfs_params_selects_best():
    mesh = example_mesh("cylinder", radius=0.4, height=1.5, sections=12)
    result = search_mcfs_params(
        mesh,
        profile="robust",
        max_search_contracts=3,
        max_iterations=60,
        timeout_seconds=45.0,
        validate=False,
        return_trials=True,
        prune_thick_hubs=True,
        extend_tips=False,
    )
    assert isinstance(result, McfsSearchResult)
    assert result.n_contracts >= 2
    assert result.skeleton.nodes.shape[0] > 0
    non_rej = [t for t in result.trials if not t["rejected"] and t["skeleton"] is not None]
    assert non_rej
    assert result.score.value >= max(t["score"].value for t in non_rej) - 1e-9


def test_search_skips_rejected_when_valid_exists():
    """A remesh-aborted contraction must not win if a valid convert exists."""
    mesh = example_mesh("cylinder", radius=0.5, height=1.0, sections=12)

    G = nx.Graph()
    G.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, pos=np.array([0.1, 0.0, 0.0]))
    G.add_edge(0, 1)
    good_skel = Skeleton.from_graph(G)

    call_count = {"n": 0}

    class _MixedDriver:
        def __init__(self, *args, **kwargs):
            call_count["n"] += 1
            self._abort = call_count["n"] == 1
            if self._abort:
                self.aborted_remesh_growth = True
                self.V = np.zeros((0, 3))
                self.F = np.zeros((0, 3), dtype=int)
            else:
                self.aborted_remesh_growth = False
                self.V = np.eye(3)
                self.F = np.array([[0, 1, 2]], dtype=int)
            self.area_overshoot_seen = False

        def contract_until_convergence(self):
            return 0

        def remesh_growth_ratio(self):
            return 10.0 if self._abort else 1.0

        def convert_to_skeleton(self, **kwargs):
            if self._abort:
                raise AssertionError("convert should not run after abort")
            return good_skel

    with patch("pymcfs.search.MeanCurvatureFlowSkeletonization", _MixedDriver):
        with patch("pymcfs.search.analyze_skeleton") as mock_analyze:
            mock_analyze.return_value = _base_report(
                n_nodes=2, n_edges=1, n_junctions=0, n_leaves=2
            )
            result = search_mcfs_params(
                mesh,
                profile="robust",
                max_search_contracts=2,
                validate=False,
                return_trials=True,
                prune_thick_hubs=False,
                extend_tips=False,
            )

    assert result.skeleton is good_skel
    assert not result.score.rejected
    aborted = [t for t in result.trials if t["rejected"]]
    valid = [t for t in result.trials if not t["rejected"]]
    assert aborted
    assert valid
    assert result.score.value >= max(t["score"].value for t in valid) - 1e-9
