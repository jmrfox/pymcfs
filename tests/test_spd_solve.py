"""Tests for SPD AtA solvers used by MCFS contraction."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pymcfs.spd_solve import cholmod_available, resolve_use_cholmod, solve_spd_ata


def _random_spd(n: int = 20, seed: int = 0) -> tuple[sp.csc_matrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.2, format="csr", random_state=rng)
    AtA = (A.T @ A + sp.eye(n) * 1e-2).tocsc()
    X_true = rng.normal(size=(n, 3))
    rhs = AtA @ X_true
    return AtA, rhs


def test_spd_solve_fallback_superlu():
    AtA, rhs = _random_spd()
    X, backend = solve_spd_ata(AtA, rhs, use_cholmod=False)
    assert backend in ("superlu", "spsolve")
    resid = AtA @ X - rhs
    assert float(np.linalg.norm(resid)) < 1e-8


def test_resolve_use_cholmod_false():
    assert resolve_use_cholmod(False) is False


def test_spd_solve_cholmod():
    pytest.importorskip("sksparse")
    assert cholmod_available()
    AtA, rhs = _random_spd()
    X, backend = solve_spd_ata(AtA, rhs, use_cholmod=True)
    assert backend == "cholmod"
    resid = AtA @ X - rhs
    assert float(np.linalg.norm(resid)) < 1e-8
