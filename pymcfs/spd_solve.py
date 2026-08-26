"""SPD normal-equation solvers for MCFS geometry contraction."""
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from sksparse.cholmod import cholesky as _cholmod_cholesky
except ImportError:  # optional SuiteSparse CHOLMOD via scikit-sparse
    _cholmod_cholesky = None

SpdBackend = Literal["cholmod", "superlu", "spsolve"]


def cholmod_available() -> bool:
    """Return True if scikit-sparse CHOLMOD is importable."""
    return _cholmod_cholesky is not None


def resolve_use_cholmod(use_cholmod: bool | None) -> bool:
    """Resolve driver ``use_cholmod``: None means use CHOLMOD when available."""
    if use_cholmod is False:
        return False
    if use_cholmod is True:
        if not cholmod_available():
            raise ImportError(
                "use_cholmod=True requires scikit-sparse (pip/conda install "
                "pymcfs[cholmod] or scikit-sparse with SuiteSparse)"
            )
        return True
    return cholmod_available()


def solve_spd_ata(
    AtA: sp.spmatrix,
    At_rhs: np.ndarray,
    *,
    use_cholmod: bool = False,
) -> tuple[np.ndarray, SpdBackend]:
    """Solve ``AtA X = At_rhs`` for a 3-column RHS.

    Prefers CHOLMOD when ``use_cholmod`` is True, then SciPy SuperLU
    (``factorized``), then ``spsolve`` as a last resort.
    """
    AtA = AtA.tocsc()
    At_rhs = np.asarray(At_rhs, dtype=float)
    if At_rhs.ndim != 2 or At_rhs.shape[1] != 3:
        raise ValueError(f"At_rhs must have shape (n, 3); got {At_rhs.shape}")
    n = AtA.shape[0]
    X = np.empty((n, 3), dtype=float)

    if use_cholmod and _cholmod_cholesky is not None:
        try:
            factor = _cholmod_cholesky(AtA)
            for c in range(3):
                X[:, c] = np.asarray(factor(np.asarray(At_rhs[:, c]).ravel())).ravel()
            return X, "cholmod"
        except Exception:
            pass

    try:
        solver = spla.factorized(AtA)
        for c in range(3):
            X[:, c] = solver(np.asarray(At_rhs[:, c]).ravel())
        return X, "superlu"
    except Exception:
        for c in range(3):
            X[:, c] = spla.spsolve(AtA, np.asarray(At_rhs[:, c]).ravel())
        return X, "spsolve"
