"""Stage-wise Starlab parity metrics and fixture loaders.

Compares pymcfs dumps against Starlab reference dumps under
``fixtures/parity/<mesh>/{starlab,pymcfs}/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import trimesh as tm
from scipy.spatial import cKDTree

from .cg_io import graph_from_cg
from .refine import refine_skeleton_graph

FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "parity"


@dataclass(frozen=True)
class PoleCompareResult:
    n: int
    bbox_diag: float
    mean: float
    max: float
    frac_above: float
    threshold: float


@dataclass(frozen=True)
class CloudCompareResult:
    n_a: int
    n_b: int
    bbox_diag: float
    mean_a_to_b: float
    mean_b_to_a: float
    chamfer: float
    hausdorff_approx: float


@dataclass(frozen=True)
class CurveCompareResult:
    bbox_diag: float
    one_sided_a_to_b: float
    one_sided_b_to_a: float
    n_leaves_a: int
    n_leaves_b: int
    n_junctions_a: int
    n_junctions_b: int
    cyclomatic_a: int
    cyclomatic_b: int
    n_components_a: int
    n_components_b: int


def bbox_diagonal(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return 1.0
    extents = points.max(axis=0) - points.min(axis=0)
    d = float(np.linalg.norm(extents))
    return d if d > 0 else 1.0


def load_mesh(path: str | Path) -> tm.Trimesh:
    mesh = tm.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, tm.Trimesh):
        raise TypeError(f"expected triangle mesh at {path}")
    return mesh


def read_starlab_poles_off(path: str | Path) -> np.ndarray:
    """Read Starlab medial ``nOFF`` poles file (``x y z angle radius`` per vertex).

    Returns
    -------
    poles : (n, 3) float
    """
    path = Path(path)
    poles: list[list[float]] = []
    n_vertices: int | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            token0 = parts[0].upper()
            if token0 in {"NOFF", "OFF", "COFF"}:
                continue
            # Property-count line used by some medial dumps ("5" before nv nf ne).
            if n_vertices is None and len(parts) == 1 and _is_int_token(parts[0]):
                continue
            if n_vertices is None and len(parts) >= 3 and all(
                _is_int_token(p) for p in parts[:3]
            ):
                n_vertices = int(parts[0])
                continue
            if n_vertices is None:
                # No counts header — treat as data (caller validates length).
                n_vertices = 10**9
            if len(poles) >= n_vertices:
                break
            if len(parts) < 3:
                continue
            poles.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if n_vertices is not None and len(poles) >= n_vertices:
                break
    if not poles:
        raise ValueError(f"no pole vertices found in {path}")
    return np.asarray(poles, dtype=float)


def _is_int_token(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def compare_poles(
    poles_a: np.ndarray,
    poles_b: np.ndarray,
    *,
    surface_points: np.ndarray | None = None,
    rel_threshold: float = 1e-3,
) -> PoleCompareResult:
    """Per-vertex pole distance (requires equal length / correspondence)."""
    a = np.asarray(poles_a, dtype=float)
    b = np.asarray(poles_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"pole shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("poles must have shape (n, 3)")
    ref = surface_points if surface_points is not None else a
    diag = bbox_diagonal(np.asarray(ref, dtype=float))
    d = np.linalg.norm(a - b, axis=1)
    thr = float(rel_threshold) * diag
    return PoleCompareResult(
        n=int(a.shape[0]),
        bbox_diag=diag,
        mean=float(d.mean()) if d.size else 0.0,
        max=float(d.max()) if d.size else 0.0,
        frac_above=float(np.mean(d > thr)) if d.size else 0.0,
        threshold=thr,
    )


def _nn_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.size == 0:
        return np.zeros(0, dtype=float)
    if dst.size == 0:
        return np.full(src.shape[0], np.inf, dtype=float)
    tree = cKDTree(dst)
    dist, _ = tree.query(src, k=1)
    return np.asarray(dist, dtype=float)


def compare_point_clouds(
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    bbox_ref: np.ndarray | None = None,
) -> CloudCompareResult:
    """Bidirectional nearest-neighbor / Chamfer distances between clouds."""
    a = np.asarray(points_a, dtype=float)
    b = np.asarray(points_b, dtype=float)
    ref = bbox_ref if bbox_ref is not None else np.vstack([a, b]) if a.size and b.size else a
    diag = bbox_diagonal(np.asarray(ref, dtype=float))
    d_ab = _nn_distances(a, b)
    d_ba = _nn_distances(b, a)
    mean_ab = float(d_ab.mean()) if d_ab.size else 0.0
    mean_ba = float(d_ba.mean()) if d_ba.size else 0.0
    haus = 0.0
    if d_ab.size or d_ba.size:
        haus = float(max(d_ab.max() if d_ab.size else 0.0, d_ba.max() if d_ba.size else 0.0))
    return CloudCompareResult(
        n_a=int(a.shape[0]),
        n_b=int(b.shape[0]),
        bbox_diag=diag,
        mean_a_to_b=mean_ab,
        mean_b_to_a=mean_ba,
        chamfer=0.5 * (mean_ab + mean_ba),
        hausdorff_approx=haus,
    )


def _degree_counts(G: nx.Graph) -> tuple[int, int, int]:
    if G.number_of_nodes() == 0:
        return 0, 0, 0
    deg = dict(G.degree())
    leaves = sum(1 for d in deg.values() if d == 1)
    junctions = sum(1 for d in deg.values() if d >= 3)
    return leaves, junctions, nx.number_connected_components(G)


def _cyclomatic(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    return int(G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G))


def densify_skeleton_points(
    G: nx.Graph,
    *,
    spacing_frac: float = 0.002,
) -> np.ndarray:
    """Resample curve chains then return all node positions as a point cloud."""
    if G.number_of_nodes() == 0:
        return np.zeros((0, 3), dtype=float)
    H = refine_skeleton_graph(G, mode="uniform", spacing_frac=float(spacing_frac))
    return np.array([np.asarray(H.nodes[n]["pos"], dtype=float) for n in H.nodes], dtype=float)


def compare_curves(
    G_a: nx.Graph,
    G_b: nx.Graph,
    *,
    spacing_frac: float = 0.002,
    bbox_ref: np.ndarray | None = None,
) -> CurveCompareResult:
    """Starlab-style one-sided NN / bbox-diag after densifying both curves."""
    pts_a = densify_skeleton_points(G_a, spacing_frac=spacing_frac)
    pts_b = densify_skeleton_points(G_b, spacing_frac=spacing_frac)
    if bbox_ref is None:
        stacks = [p for p in (pts_a, pts_b) if p.size]
        bbox_ref = np.vstack(stacks) if stacks else np.zeros((1, 3))
    cloud = compare_point_clouds(pts_a, pts_b, bbox_ref=bbox_ref)
    la, ja, ca = _degree_counts(G_a)
    lb, jb, cb = _degree_counts(G_b)
    return CurveCompareResult(
        bbox_diag=cloud.bbox_diag,
        one_sided_a_to_b=cloud.mean_a_to_b / cloud.bbox_diag,
        one_sided_b_to_a=cloud.mean_b_to_a / cloud.bbox_diag,
        n_leaves_a=la,
        n_leaves_b=lb,
        n_junctions_a=ja,
        n_junctions_b=jb,
        cyclomatic_a=_cyclomatic(G_a),
        cyclomatic_b=_cyclomatic(G_b),
        n_components_a=ca,
        n_components_b=cb,
    )


def fixture_dir(name: str, *, root: Path | None = None) -> Path:
    return (root or FIXTURES_ROOT) / name


def find_input_mesh(case_dir: Path) -> Path:
    for name in ("input.off", "input.obj", "input.ply"):
        p = case_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"no input mesh in {case_dir}")


def load_curve_graph(path: Path) -> nx.Graph:
    path = Path(path)
    if path.suffix.lower() == ".cg":
        return graph_from_cg(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        nodes = np.asarray(data["nodes"], dtype=float)
        edges = np.asarray(data["edges"], dtype=int)
        G = nx.Graph()
        for i, p in enumerate(nodes):
            G.add_node(i, pos=p)
        for u, v in edges:
            pu, pv = nodes[int(u)], nodes[int(v)]
            G.add_edge(int(u), int(v), weight=float(np.linalg.norm(pu - pv)))
        return G
    raise ValueError(f"unsupported curve format: {path}")


def list_parity_cases(root: Path | None = None) -> list[str]:
    root = root or FIXTURES_ROOT
    if not root.is_dir():
        return []
    out: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("_"):
            try:
                find_input_mesh(child)
            except FileNotFoundError:
                continue
            out.append(child.name)
    return out


def iter_meso_snapshots(side_dir: Path) -> Iterable[tuple[int, Path]]:
    """Yield ``(N, path)`` for ``meso_NXXXX.off`` / ``.obj`` / ``.npz`` files."""
    side_dir = Path(side_dir)
    if not side_dir.is_dir():
        return
    seen_n: set[int] = set()
    for path in sorted(side_dir.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem
        # Accept meso_N0001 / meso_n0001 (Windows/Starlab naming).
        if not stem.lower().startswith("meso_n"):
            continue
        suffix = stem[6:]  # after 'meso_N' / 'meso_n'
        if not suffix.isdigit():
            continue
        n = int(suffix)
        # Prefer .npz over duplicate mesh when both exist
        if n in seen_n and path.suffix.lower() != ".npz":
            continue
        seen_n.add(n)
        yield n, path


def load_meso_vertices(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        return np.asarray(data["V"], dtype=float)
    mesh = load_mesh(path)
    return np.asarray(mesh.vertices, dtype=float)
