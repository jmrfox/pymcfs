#!/usr/bin/env python3
"""
Generate simple closed primitive meshes for MCFS development and save to data/mesh/.

Usage:
  uv run python scripts/make_test_meshes.py
  uv run python scripts/make_test_meshes.py --outdir data/mesh
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh as tm


def closed_cylinder(
    radius: float = 0.5,
    height: float = 2.0,
    *,
    radial_sections: int = 32,
    height_sections: int = 16,
) -> tm.Trimesh:
    """Watertight cylinder with multiple rings along the axis.

    ``trimesh.creation.cylinder`` only samples the top/bottom rims, which is too
    coarse for MCFS. This builder adds ``height_sections`` rings so the medial
    axis is well sampled along ``z``.
    """
    rs = int(radial_sections)
    hs = int(height_sections)
    if rs < 3:
        raise ValueError("radial_sections must be >= 3")
    if hs < 2:
        raise ValueError("height_sections must be >= 2")

    angles = np.linspace(0.0, 2.0 * np.pi, rs, endpoint=False)
    zs = np.linspace(-0.5 * height, 0.5 * height, hs)
    ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    verts: list[list[float]] = []
    for z in zs:
        for x, y in ring:
            verts.append([float(x), float(y), float(z)])
    top_c = len(verts)
    verts.append([0.0, 0.0, 0.5 * height])
    bot_c = len(verts)
    verts.append([0.0, 0.0, -0.5 * height])
    V = np.asarray(verts, dtype=float)

    faces: list[list[int]] = []
    for i in range(hs - 1):
        for j in range(rs):
            j2 = (j + 1) % rs
            a = i * rs + j
            b = i * rs + j2
            c = (i + 1) * rs + j2
            d = (i + 1) * rs + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    base_top = (hs - 1) * rs
    for j in range(rs):
        j2 = (j + 1) % rs
        faces.append([base_top + j, base_top + j2, top_c])
    for j in range(rs):
        j2 = (j + 1) % rs
        faces.append([j2, j, bot_c])

    mesh = tm.Trimesh(vertices=V, faces=np.asarray(faces, dtype=int), process=True)
    if not mesh.is_watertight:
        mesh.fix_normals()
    return mesh


def build_primitives() -> dict[str, tm.Trimesh]:
    """Return named watertight triangle meshes suitable for MCFS testing."""
    return {
        "sphere": tm.creation.icosphere(subdivisions=3, radius=1.0),
        "cylinder": closed_cylinder(radius=0.5, height=2.0, radial_sections=32, height_sections=16),
        "torus": tm.creation.torus(
            major_radius=1.0,
            minor_radius=0.3,
            major_sections=48,
            minor_sections=16,
        ),
        "cube": tm.creation.box(extents=[1.0, 1.0, 1.0]),
        "capsule": tm.creation.capsule(radius=0.4, height=1.5, count=[32, 16]),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Write primitive test meshes under data/mesh/")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=root / "data" / "mesh",
        help="Output directory (default: data/mesh)",
    )
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    meshes = build_primitives()
    for name, mesh in meshes.items():
        if mesh.faces.shape[1] != 3:
            mesh = mesh.triangulate()
        path = outdir / f"{name}.obj"
        mesh.export(path, file_type="obj")
        print(
            f"{path.name:12s}  verts={len(mesh.vertices):5d}  "
            f"faces={len(mesh.faces):5d}  watertight={mesh.is_watertight}  "
            f"-> {path}"
        )


if __name__ == "__main__":
    main()
