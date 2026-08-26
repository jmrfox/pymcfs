"""Starlab parity fixtures, metrics, and stage gates."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from pymcfs.cg_io import read_cg, write_cg
from pymcfs.medial import compute_voronoi_poles
from pymcfs.mcfs import MeanCurvatureFlowSkeletonization
from pymcfs.parity import (
    FIXTURES_ROOT,
    compare_curves,
    compare_point_clouds,
    compare_poles,
    find_input_mesh,
    fixture_dir,
    load_mesh,
    load_meso_vertices,
    read_starlab_poles_off,
)

SINDO = FIXTURES_ROOT / "sindorelax"
CYL = FIXTURES_ROOT / "cylinder"


def _meso_n1_path(side: Path) -> Path | None:
    for name in ("meso_N0001.off", "meso_N0001.obj", "meso_N0001.npz"):
        p = side / name
        if p.is_file():
            return p
    return None


@pytest.fixture(scope="module")
def sindorelax_mesh():
    if not (SINDO / "input.off").is_file():
        pytest.skip("sindorelax fixture input.off missing")
    return load_mesh(SINDO / "input.off")


def test_read_starlab_poles_off_shape(sindorelax_mesh):
    poles_path = SINDO / "starlab" / "poles.off"
    assert poles_path.is_file()
    poles = read_starlab_poles_off(poles_path)
    assert poles.shape == (len(sindorelax_mesh.vertices), 3)
    assert np.isfinite(poles).all()


def test_cg_roundtrip(tmp_path: Path):
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float)
    edges = np.array([[0, 1], [1, 2]], dtype=int)
    path = tmp_path / "s.cg"
    write_cg(path, nodes, edges)
    n2, e2 = read_cg(path)
    assert np.allclose(nodes, n2)
    assert np.array_equal(edges, e2)


def test_compare_poles_detects_shift():
    a = np.zeros((10, 3), dtype=float)
    b = a + np.array([0.1, 0.0, 0.0])
    r = compare_poles(a, b, surface_points=a, rel_threshold=1e-3)
    # bbox of zeros degenerates to diag=1.0 fallback
    assert r.mean == pytest.approx(0.1)
    assert r.frac_above == 1.0


def test_compare_clouds_and_curves_metrics():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    b = a + np.array([0.01, 0.0, 0.0])
    cloud = compare_point_clouds(a, b)
    assert cloud.chamfer == pytest.approx(0.01, rel=1e-6)

    Ga = nx.Graph()
    Ga.add_node(0, pos=np.array([0.0, 0.0, 0.0]))
    Ga.add_node(1, pos=np.array([1.0, 0.0, 0.0]))
    Ga.add_edge(0, 1, weight=1.0)
    Gb = nx.Graph()
    Gb.add_node(0, pos=np.array([0.0, 0.05, 0.0]))
    Gb.add_node(1, pos=np.array([1.0, 0.05, 0.0]))
    Gb.add_edge(0, 1, weight=1.0)
    curve = compare_curves(Ga, Gb, spacing_frac=0.5)
    assert curve.n_leaves_a == 2
    assert curve.n_leaves_b == 2
    assert curve.cyclomatic_a == 0
    assert curve.one_sided_a_to_b > 0


def test_stage1_sindorelax_poles_within_tolerance(sindorelax_mesh):
    """Hard gate: pymcfs Voronoi poles vs Starlab sindorelax_poles.off."""
    star = read_starlab_poles_off(SINDO / "starlab" / "poles.off")
    py, _ = compute_voronoi_poles(sindorelax_mesh)
    # Persist for the compare script / fixtures layout
    out = SINDO / "pymcfs"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "poles.npy", py)

    r = compare_poles(py, star, surface_points=sindorelax_mesh.vertices)
    # Plan success criterion: mean ≪ 0.01 × bbox_diag
    assert r.mean < 0.01 * r.bbox_diag, (
        f"pole mean error {r.mean:.6g} exceeds 0.01*diag={0.01 * r.bbox_diag:.6g} "
        f"(max={r.max:.6g}, frac_above={r.frac_above:.4f})"
    )
    assert r.max < 0.05 * r.bbox_diag, (
        f"pole max error {r.max:.6g} exceeds 0.05*diag={0.05 * r.bbox_diag:.6g}"
    )


def test_stage2_meso_n1_within_tolerance():
    """Hard gate when Starlab meso_N0001 is present; otherwise skip."""
    # Prefer sindorelax; fall back to cylinder if that has both sides.
    cases = []
    for name in ("sindorelax", "cylinder", "indorelax"):
        d = fixture_dir(name)
        star_m = _meso_n1_path(d / "starlab")
        # pymcfs side may be generated below
        if star_m is not None:
            cases.append((name, d, star_m))
    if not cases:
        pytest.skip(
            "no starlab/meso_N0001.* fixtures yet "
            "(Starlab demo MCF crashes without CHOLMOD; add dumps when available)"
        )

    name, case_dir, star_path = cases[0]
    mesh = load_mesh(find_input_mesh(case_dir))
    py_dir = case_dir / "pymcfs"
    py_path = _meso_n1_path(py_dir)
    if py_path is None:
        # Generate N=1 meso on the fly for the gate
        driver = MeanCurvatureFlowSkeletonization(
            mesh,
            w_H=0.1,
            w_M=0.2,
            gate_exterior_poles=False,  # Starlab parity profile
            max_iterations=1,
            timeout_seconds=120.0,
            validate=True,
            verbose=False,
        )
        driver.contract()
        py_dir.mkdir(parents=True, exist_ok=True)
        off = py_dir / "meso_N0001.off"
        driver.meso_skeleton_mesh().export(str(off))
        np.savez_compressed(
            py_dir / "meso_N0001.npz",
            V=driver.V.copy(),
            F=driver.F.copy(),
        )
        py_path = off

    Va = load_meso_vertices(py_path)
    Vb = load_meso_vertices(star_path)
    r = compare_point_clouds(Va, Vb, bbox_ref=np.asarray(mesh.vertices, float))
    # Remesh makes exact match impossible; keep a loose but meaningful bound.
    assert r.chamfer < 0.05 * r.bbox_diag, (
        f"{name} meso_N0001 chamfer {r.chamfer:.6g} exceeds "
        f"0.05*diag={0.05 * r.bbox_diag:.6g}"
    )
