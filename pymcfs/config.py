"""Frozen settings for MCFS contraction and the post-conversion refine phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .params import BranchingPreference

McfsProfile = Literal["robust", "starlab", "auto"]
ResampleMode = Literal["uniform", "compress"]


@dataclass(frozen=True)
class ContractionSettings:
    """Controls for meso-skeleton contraction (mean-curvature flow).

    The surface shrinks toward the medial axis while staying a valid mesh.
    Attraction weight pulls vertices toward their current positions (stability).
    Medial weight pulls toward Voronoi poles (approximate medial-axis targets
    inside the volume) so the contracted surface stays centered.
    """

    attraction_weight: float = 0.5
    medial_weight: float = 5.0
    profile: McfsProfile | None = None
    branching: BranchingPreference = "sparse"
    gate_exterior_poles: bool | None = None
    fast_gating: bool = False
    use_cholmod: bool | None = None
    max_iterations: int = 500
    timeout_seconds: float | None = 120.0
    min_edge_length: float | None = None
    max_triangle_angle: float = 110.0
    area_variation_factor: float = 1e-4
    max_vertex_growth: float | None = 4.0
    pinned_attraction_floor: float = 1e-7


@dataclass(frozen=True)
class RefineSettings:
    """Post-contraction quality phase: prune, tip extension, and curve resampling.

    Runs after the meso-skeleton is converted to a curve graph.
    ``resample`` / ``resample_spacing`` / ``resample_spacing_frac`` control
    curve-node density only — distinct from contraction parameter search.
    """

    prune_exterior: bool = True
    prune_short_leaves: bool = True
    short_leaf_scale: float = 1.0
    prune_thick_hubs: bool = True
    keep_hub_branches: int = 2
    hub_degree_min: int = 4
    hub_radius_frac: float = 0.015
    extend_tips: bool = False
    tip_extend_scale: float = 1.0
    tip_clearance_frac: float = 0.01
    tip_cone_deg: float = 40.0
    keep_largest_component: bool = False
    resample: bool | ResampleMode = False
    resample_spacing: float | None = None
    resample_spacing_frac: float | None = None


@dataclass(frozen=True)
class SkeletonizeSettings:
    """Top-level settings for :func:`pymcfs.skeleton.skeletonize` (contract + refine)."""

    contraction: ContractionSettings = field(default_factory=ContractionSettings)
    refine: RefineSettings = field(default_factory=RefineSettings)
    validate: bool = True
    parameter_search: bool = False
    max_search_contracts: int = 4


__all__ = [
    "ContractionSettings",
    "RefineSettings",
    "SkeletonizeSettings",
    "McfsProfile",
    "ResampleMode",
]
