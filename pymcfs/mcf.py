from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import logging
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh as tm

from .laplacian import cotangent_laplacian, mean_value_laplacian, lumped_mass_matrix

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@dataclass
class MCFResult:
    vertices: np.ndarray  # (n,3) final vertex positions
    history: Optional[Sequence[np.ndarray]] = None  # optional list of intermediate vertices


def mean_curvature_flow(
    mesh: tm.Trimesh,
    *,
    dt: float = 1e-2,
    iterations: int = 20,
    laplacian_type: str = "cotangent",
    laplacian_secure: bool = False,
    guidance_type: str | None = None,
    guidance_weight: float = 0.0,
    guidance_targets: np.ndarray | None = None,
    guidance_diag: np.ndarray | float | None = None,
    record_history: bool = False,
    verbose: bool = False,
    log: Optional[logging.Logger] = None,
) -> MCFResult:
    """Implicit mean curvature flow on a closed triangle mesh.

    We evolve vertex positions V by dV/dt = -L V using implicit Euler. Multiplying
    by the lumped mass matrix M, one step solves the linear system:

        (M + dt·L) V^{k+1} = M V^k

    With optional soft positional guidance (targets T with diagonal weights W):

        (M + dt·L + W) V^{k+1} = M V^k + W T

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input (ideally closed, manifold) triangle mesh with faces of shape (m,3) and
        vertices of shape (n,3).
    dt : float, default 1e-2
        Time step. Larger values shrink faster but may over-smooth features; typical 1e-3..5e-2.
    iterations : int, default 20
        Number of implicit steps to perform.
    laplacian_type : {"cotangent", "mean_value"}, default "cotangent"
        Discrete Laplacian variant to use.
    laplacian_secure : bool, default False
        When using the cotangent Laplacian, if True clamp negative cotangent
        weights to zero for extra robustness (mirrors the reference C++ behavior).
    guidance_type : {None, "centroid", "original"}, optional
        Convenience option to set targets T uniformly to the centroid when "centroid".
        When "original", targets T are set to the input mesh vertices (pull-to-original).
        Ignored if ``guidance_targets`` is provided.
    guidance_weight : float, default 0.0
        Scalar guidance strength. Used only if ``guidance_targets`` is provided with
        ``guidance_diag=None``, or when using ``guidance_type`` without an explicit
        diagonal. Set to 0 to disable guidance.
    guidance_targets : (n,3) array-like, optional
        Per-vertex target positions T. Takes precedence over ``guidance_type``.
    guidance_diag : float or (n,) array-like, optional
        Diagonal of W. If a scalar, W = w·I. If a vector of length n, W = diag(d).
        When not provided, a scalar W is built from ``guidance_weight`` when applicable.
    record_history : bool, default False
        If True, return a list of vertices after each step in ``MCFResult.history``.
    verbose : bool, default False
        If True, log basic progress information.
    log : logging.Logger, optional
        Custom logger. If None, use module logger.

    Returns
    -------
    MCFResult
        Dataclass with
        - ``vertices``: (n,3) array of final vertex positions,
        - ``history``: optional list of intermediate (n,3) arrays when ``record_history`` is True.
    """
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError("mesh.vertices must have shape (n,3)")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("mesh.faces must have shape (m,3)")

    V = mesh.vertices.view(np.ndarray).astype(np.float64, copy=True)
    F = mesh.faces.view(np.ndarray).astype(np.int64, copy=False)
    V0 = V.copy()  # keep a copy of the input vertices for guidance_type="original"

    _log = log or logger
    if verbose:
        _log.info("MCF: starting with %d vertices, %d faces; dt=%.3g, iters=%d", V.shape[0], F.shape[0], dt, iterations)

    # Precompute operators
    if laplacian_type == "cotangent":
        L = cotangent_laplacian(V, F, verbose=verbose, secure=laplacian_secure)
    elif laplacian_type == "mean_value":
        L = mean_value_laplacian(V, F, verbose=verbose)
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")
    M = lumped_mass_matrix(V, F, verbose=verbose)   # (n,n)

    # Optional guidance matrix (diagonal weights) for soft positional targets
    W = None
    T = None
    # Priority: explicit guidance_targets override guidance_type
    if guidance_targets is not None:
        if guidance_targets.shape != V.shape:
            raise ValueError("guidance_targets must match V shape (n,3)")
        T = guidance_targets.astype(float, copy=False)
        if guidance_diag is None:
            # fallback to scalar guidance_weight
            w = float(guidance_weight) if guidance_weight > 0 else 0.0
            if w > 0:
                W = sp.identity(V.shape[0], dtype=float, format="csr") * w
        else:
            if np.isscalar(guidance_diag):
                W = sp.identity(V.shape[0], dtype=float, format="csr") * float(guidance_diag)  # type: ignore[arg-type]
            else:
                d = np.asarray(guidance_diag, dtype=float).ravel()
                if d.shape[0] != V.shape[0]:
                    raise ValueError("guidance_diag vector must have length n (num vertices)")
                W = sp.diags(d, format="csr")
    elif guidance_type is not None and guidance_weight > 0.0:
        if guidance_type == "centroid":
            centroid = V.mean(axis=0)
            T = np.tile(centroid.reshape(1, 3), (V.shape[0], 1))
        elif guidance_type == "original":
            T = V0
        else:
            raise ValueError(f"Unknown guidance_type: {guidance_type}")
        W = sp.identity(V.shape[0], dtype=float, format="csr") * float(guidance_weight)

    # Left-hand side is constant if topology doesn't change
    # Implicit Euler for dV/dt = -L V (plus optional soft guidance):
    # (M + dt*L + W) V^{k+1} = M V^k + W T
    A = (M + dt * L)
    if W is not None:
        A = (A + W)
    A = A.tocsr()

    # Factor once for efficiency
    solver = spla.factorized(A.tocsc())
    if verbose:
        _log.info("MCF: factorization complete (A nnz=%d)", A.nnz)

    hist: list[np.ndarray] | None = [] if record_history else None

    for k in range(iterations):
        RHSx = M @ V  # (n,3)
        if W is not None and T is not None:
            RHSx = RHSx + W @ T
        V[:, 0] = solver(RHSx[:, 0])
        V[:, 1] = solver(RHSx[:, 1])
        V[:, 2] = solver(RHSx[:, 2])
        if hist is not None:
            hist.append(V.copy())
        if verbose and ((k + 1) % max(1, iterations // 5) == 0 or k == iterations - 1):
            _log.info("MCF: completed step %d/%d", k + 1, iterations)

    return MCFResult(vertices=V, history=hist)
