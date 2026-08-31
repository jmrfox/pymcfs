"""Shared pytest fixtures for pymcfs tests.

Prefer ``Skeleton.from_graph(G)`` over hand-rolled graph→Skeleton helpers.
Prefer ``example_mesh("cylinder", ...)`` over local closed-cylinder builders
when the default axially-sampled cylinder is adequate.
"""

from __future__ import annotations

import pytest
import trimesh as tm

from pymcfs.mesh import example_mesh


@pytest.fixture
def unit_sphere() -> tm.Trimesh:
    """Unit icosphere suitable for quick contraction / quality checks."""
    return tm.creation.icosphere(subdivisions=2, radius=1.0)


@pytest.fixture
def cylinder_mesh() -> tm.Trimesh:
    """Closed, axially sampled cylinder via :func:`pymcfs.mesh.example_mesh`."""
    return example_mesh("cylinder")
