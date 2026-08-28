"""Mesh-conditioned MCFS parameter proposals (parameter oracle)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import trimesh as tm

from .medial import compute_voronoi_poles, points_inside_mesh

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Anchor band from TS1-like meshes where robust (0.5, 5.0) works well.
_RHO_REF = 0.016
# Characteristic radius / diag for the same TS1-like band. Thick compartments
# (e.g. TS3 soma) can keep mean ρ near-ref while char_r is elevated — using
# only mean ρ then snaps to robust and fills the volume with spurious branches.
_CHAR_REF = 0.025
_ROBUST_W_H = 0.5
_ROBUST_W_M = 5.0
_ROBUST_RATIO = 10.0
# Below this relative scale, treat as TS1-like and prefer exact robust weights
# under sparse/balanced (avoids tiny oracle drift → extra junctions).
_NEAR_REF_SCALE = 1.2

BranchingPreference = Literal["sparse", "balanced", "dense"]

# Multipliers on the medial ratio after the geometry base proposal.
# Sparse errs toward fewer junctions (neuroscience default).
_BRANCHING_RATIO_SCALE: dict[BranchingPreference, float] = {
    "sparse": 0.70,
    "balanced": 1.00,
    "dense": 1.30,
}


@dataclass(frozen=True)
class MeshMcfsFeatures:
    """Preflight geometry features used by :func:`propose_mcfs_params`.

    Attributes
    ----------
    n_vertices :
        Number of mesh vertices.
    bbox_diag :
        Axis-aligned bounding-box diagonal length.
    mean_pole_offset :
        Mean Euclidean distance from each vertex to its Voronoi pole.
    p95_pole_offset :
        95th percentile of those offsets.
    mean_pole_offset_over_diag :
        ``mean_pole_offset / bbox_diag`` (dominant oracle signal ``ρ``).
    p95_pole_offset_over_diag :
        ``p95_pole_offset / bbox_diag``.
    poles_inside_frac :
        Fraction of poles classified inside the mesh volume.
    char_radius_over_diag :
        Characteristic radius over diagonal (volume/area heuristic when
        available, else pole-offset based).
    """

    n_vertices: int
    bbox_diag: float
    mean_pole_offset: float
    p95_pole_offset: float
    mean_pole_offset_over_diag: float
    p95_pole_offset_over_diag: float
    poles_inside_frac: float
    char_radius_over_diag: float

    def summary(self) -> str:
        """One-line feature summary for logs."""
        return (
            f"n={self.n_vertices} diag={self.bbox_diag:.4g} "
            f"pole/diag={100.0 * self.mean_pole_offset_over_diag:.2f}% "
            f"(p95={100.0 * self.p95_pole_offset_over_diag:.2f}%) "
            f"poles_inside={100.0 * self.poles_inside_frac:.1f}% "
            f"char_r/diag={100.0 * self.char_radius_over_diag:.2f}%"
        )


@dataclass(frozen=True)
class McfsParams:
    """Suggested contraction weights for a mesh.

    Attributes
    ----------
    w_H :
        Quality/speed (attraction) weight.
    w_M :
        Medial-centering weight.
    gate_exterior_poles :
        Whether to gate exterior Voronoi poles (always True from the oracle).
    ratio :
        ``w_M / w_H`` after proposal.
    branching :
        Branching preference used to produce this proposal.
    features :
        Features used for the proposal, if available.
    rationale :
        Short human-readable explanation of the choice.
    """

    w_H: float
    w_M: float
    gate_exterior_poles: bool = True
    ratio: float = 10.0
    branching: BranchingPreference = "sparse"
    features: MeshMcfsFeatures | None = None
    rationale: str = ""

    def summary(self) -> str:
        """One-line parameter summary for logs."""
        bits = [
            f"w_H={self.w_H:.4g}",
            f"w_M={self.w_M:.4g}",
            f"r={self.ratio:.3g}",
            f"branching={self.branching}",
            f"gate={self.gate_exterior_poles}",
        ]
        if self.rationale:
            bits.append(self.rationale)
        return " ".join(bits)


def mesh_mcfs_features(
    mesh: tm.Trimesh,
    *,
    fast_gating: bool = False,
) -> MeshMcfsFeatures:
    """Extract scale-free features that correlate with good ``(w_H, w_M)``.

    The dominant signal is mean Voronoi-pole offset over bbox diagonal: thick /
    compact fragments (e.g. TS2) have larger relative medial targets and need
    a milder medial ratio than thin elongated meshes (e.g. TS1).

    Parameters
    ----------
    mesh :
        Closed triangle mesh (``trimesh.Trimesh``).
    fast_gating :
        Forwarded to :func:`pymcfs.medial.points_inside_mesh` when measuring
        ``poles_inside_frac`` (Embree / mesh ray backend when True).

    Returns
    -------
    MeshMcfsFeatures

    Raises
    ------
    TypeError
        If ``mesh`` is not a ``trimesh.Trimesh``.
    ValueError
        If the mesh has no vertices.
    """
    if not isinstance(mesh, tm.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    V = np.asarray(mesh.vertices, dtype=float)
    n = int(V.shape[0])
    if n == 0:
        raise ValueError("mesh has no vertices")
    bb = V.max(axis=0) - V.min(axis=0)
    diag = float(np.linalg.norm(bb))
    diag = max(diag, 1e-12)

    poles, _weights = compute_voronoi_poles(mesh)
    offsets = np.linalg.norm(poles - V, axis=1)
    mean_off = float(np.mean(offsets)) if offsets.size else 0.0
    p95_off = float(np.percentile(offsets, 95)) if offsets.size else 0.0
    inside = points_inside_mesh(mesh, poles, fast=bool(fast_gating))
    inside_frac = float(np.mean(inside)) if inside.size else 1.0

    char_r = mean_off
    try:
        if bool(mesh.is_watertight) and float(mesh.area) > 0:
            vol = float(mesh.volume)
            if np.isfinite(vol) and vol > 0:
                char_r = 3.0 * vol / float(mesh.area)
    except Exception:
        pass
    if inside.any():
        char_r = max(char_r, float(np.mean(offsets[inside])))

    return MeshMcfsFeatures(
        n_vertices=n,
        bbox_diag=diag,
        mean_pole_offset=mean_off,
        p95_pole_offset=p95_off,
        mean_pole_offset_over_diag=mean_off / diag,
        p95_pole_offset_over_diag=p95_off / diag,
        poles_inside_frac=inside_frac,
        char_radius_over_diag=float(char_r) / diag,
    )


def propose_mcfs_params(
    mesh: tm.Trimesh,
    *,
    features: MeshMcfsFeatures | None = None,
    fast_gating: bool = False,
    branching: BranchingPreference = "sparse",
) -> McfsParams:
    """Propose ``(w_H, w_M)`` from mesh features.

    Defaults to ``branching=\"sparse\"``: prefer fewer junctions / less spurious
    volume branching (priority for neuroscience centerlines). Near the TS1
    relative-pole *and* characteristic-radius band this returns exact robust
    ``(0.5, 5.0)``. Thick meshes (high ``ρ`` or high ``char_r``) get a low
    medial ratio for remesh stability / fewer volume junctions, with
    ``sparse`` pushing that ratio further down than ``balanced`` / ``dense``.

    Parameters
    ----------
    mesh :
        Input mesh used when ``features`` is omitted.
    features :
        Precomputed :class:`MeshMcfsFeatures`. If None, computed via
        :func:`mesh_mcfs_features`.
    fast_gating :
        Passed to :func:`mesh_mcfs_features` when features are computed here.
    branching :
        ``\"sparse\"`` (default) — fewest junctions; snap to robust on TS1-like
        meshes; lowest medial ratio on thick meshes.
        ``\"balanced\"`` — geometry base proposal (robust near ref; mild ratio
        cut when thick).
        ``\"dense\"`` — stronger medial pull / higher ratio (more branching;
        still capped for remesh safety).

    Returns
    -------
    McfsParams
        Proposed weights and metadata (always ``gate_exterior_poles=True``).

    Raises
    ------
    ValueError
        If ``branching`` is not one of ``sparse`` / ``balanced`` / ``dense``.

    Notes
    -----
    Always proposes ``gate_exterior_poles=True`` for production meshes.
    The thickness scale is ``max(ρ / ρ_ref, char_r / char_ref)`` so a mesh with
    thin processes (low mean ``ρ``) but a bulky compartment (high ``char_r``)
    is not mis-classified as near-ref.
    """
    if branching not in _BRANCHING_RATIO_SCALE:
        raise ValueError(
            f"branching must be one of {sorted(_BRANCHING_RATIO_SCALE)}; got {branching!r}"
        )

    feats = features if features is not None else mesh_mcfs_features(
        mesh, fast_gating=fast_gating
    )
    rho = float(feats.mean_pole_offset_over_diag)
    char = float(feats.char_radius_over_diag)
    rho_scale = float(np.clip(rho / _RHO_REF, 0.5, 4.0))
    char_scale = float(np.clip(char / _CHAR_REF, 0.5, 4.0))
    # Dominant thickness signal: mean pole offset or characteristic radius.
    scale = float(max(rho_scale, char_scale))
    bscale = float(_BRANCHING_RATIO_SCALE[branching])

    # Near-reference: exact robust for sparse/balanced so TS1 matches "usual"
    # values. Dense may raise medial strength slightly.
    if scale <= _NEAR_REF_SCALE:
        w_H = _ROBUST_W_H
        if branching == "dense":
            ratio = float(np.clip(_ROBUST_RATIO * bscale, 2.0, 20.0))
        else:
            ratio = _ROBUST_RATIO
        w_M = float(w_H * ratio)
        rationale = (
            f"rho={100.0 * rho:.2f}%diag char={100.0 * char:.2f}%diag "
            f"scale={scale:.2f} near-ref; branching={branching}"
        )
    else:
        # Thick / high-ρ or bulky char_r: sweep winners used low ratio (~2).
        # Sparse goes to the floor; dense allows a bit more medial pull but
        # stays remesh-safe.
        w_H = float(np.clip(_ROBUST_W_H / (scale**0.35), 0.25, 1.0))
        base_ratio = float(np.clip(_ROBUST_RATIO / (scale**1.5), 2.0, 10.0))
        ratio = float(np.clip(base_ratio * bscale, 2.0, 12.0))
        if branching == "sparse":
            # Prefer the sweep-best low-ratio band for thick fragments.
            ratio = float(min(ratio, 3.0))
        w_M = float(w_H * ratio)
        # Never exceed robust absolute medial under sparse (extra conservatism).
        if branching == "sparse":
            w_M = float(min(w_M, _ROBUST_W_M))
            ratio = float(w_M / max(w_H, 1e-12))
        rationale = (
            f"rho={100.0 * rho:.2f}%diag char={100.0 * char:.2f}%diag "
            f"scale={scale:.2f} thick; branching={branching} "
            f"(low ratio for stability/sparsity)"
        )

    logger.debug(
        "propose_mcfs_params: %s branching=%s -> w_H=%.3g w_M=%.3g",
        feats.summary(),
        branching,
        w_H,
        w_M,
    )
    return McfsParams(
        w_H=w_H,
        w_M=w_M,
        gate_exterior_poles=True,
        ratio=ratio,
        branching=branching,
        features=feats,
        rationale=rationale,
    )
