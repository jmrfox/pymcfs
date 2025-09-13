"""
Main mesh class
"""

import logging
from dataclasses import dataclass
from datetime import datetime
import multiprocessing
import traceback
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import trimesh

# Module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())



def example_mesh(
    kind: str = "cylinder",
    *,
    # Cylinder params
    radius: float = 0.5,
    height: float = 2.0,
    sections: int | None = 64,
    # Torus params
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    major_sections: int | None = 64,
    minor_sections: int | None = 32,
    # Common
    transform: np.ndarray | None = None,
    **kwargs,
) -> trimesh.Trimesh:
    """Create a simple demo mesh using trimesh primitives.

    Parameters
    ----------
    kind : {"cylinder", "torus"}
        Type of primitive to generate. Default "cylinder".
    radius : float
        Cylinder radius (when kind="cylinder"). Default 0.5.
    height : float
        Cylinder height (when kind="cylinder"). Default 2.0.
    sections : int or None
        Cylinder radial resolution (pie wedges). Default 64.
    major_radius : float
        Torus major radius (center of hole to centerline of tube). Default 1.0.
    minor_radius : float
        Torus minor radius (tube radius). Default 0.3.
    major_sections : int or None
        Torus resolution around major circle. Default 64.
    minor_sections : int or None
        Torus resolution around tube section. Default 32.
    transform : (4,4) float array, optional
        Transform applied after creation.
    **kwargs : dict
        Passed through to Trimesh constructor via trimesh.creation.* helpers
        (e.g., process=False).

    Returns
    -------
    trimesh.Trimesh
        Generated primitive mesh.

    Examples
    --------
    >>> m = example_mesh("cylinder", radius=0.4, height=1.5)
    >>> t = example_mesh("torus", major_radius=1.0, minor_radius=0.25)
    """
    k = (kind or "cylinder").lower()
    if k == "cylinder":
        return trimesh.creation.cylinder(
            radius=float(radius),
            height=float(height),
            sections=None if sections is None else int(sections),
            transform=transform,
            **kwargs,
        )
    elif k == "torus":
        # trimesh.creation.torus parameters
        return trimesh.creation.torus(
            major_radius=float(major_radius),
            minor_radius=float(minor_radius),
            major_sections=None if major_sections is None else int(major_sections),
            minor_sections=None if minor_sections is None else int(minor_sections),
            transform=transform,
            **kwargs,
        )
    else:
        raise ValueError("example_mesh kind must be 'cylinder' or 'torus'")


class MeshManager:
    """
    Unified mesh class handling loading, processing, and analysis.
    """

    @dataclass
    class Transform:
        name: str
        M: np.ndarray  # 4x4 homogeneous
        params: Optional[Dict[str, Any]]
        timestamp: datetime
        is_uniform_scale: bool = False
        uniform_scale: Optional[float] = None

    def __init__(self, mesh: Optional[trimesh.Trimesh] = None, verbose: bool = True):
        # Core mesh attributes
        self.mesh = mesh
        self.original_mesh = mesh
        self.bounds = self._compute_bounds()

        # Attributes
        self.verbose = verbose
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "volume_fixed": 0,
            "watertight_fixed": 0,
            "degenerate_removed": 0,
        }

        # Transformation tracking
        self.transform_stack: list[MeshManager.Transform] = []
        self.M_world_from_local: np.ndarray = np.eye(4, dtype=float)
        self.M_local_from_world: np.ndarray = np.eye(4, dtype=float)

    def log(self, message: str, level: str = "INFO"):
        """Compatibility shim: delegate to module logger respecting self.verbose.

        Levels: INFO, SUCCESS (maps to INFO), WARNING, ERROR, PROCESSING (maps to INFO)
        """
        if not self.verbose:
            return
        lvl = level.upper()
        if lvl in ("SUCCESS", "PROCESSING"):
            lvl = "INFO"
        if lvl == "INFO":
            logger.info(message)
        elif lvl == "WARNING":
            logger.warning(message)
        elif lvl == "ERROR":
            logger.error(message)
        else:
            logger.info(message)

    # =================================================================
    # MESH LOADING AND BASIC OPERATIONS
    # =================================================================

    def load_mesh(
        self, filepath: str, file_format: Optional[str] = None
    ) -> trimesh.Trimesh:
        """
        Load a mesh from file.

        Args:
            filepath: Path to mesh file
            file_format: Optional format specification (auto-detected if None)

        Returns:
            Loaded trimesh object
        """
        try:
            if file_format:
                mesh = trimesh.load(filepath, file_type=file_format)
            else:
                mesh = trimesh.load(filepath)

            # Ensure we have a single mesh
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, try to get the first geometry
                geometries = list(mesh.geometry.values())
                if geometries:
                    mesh = geometries[0]
                else:
                    raise ValueError("No geometry found in mesh scene")

            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Loaded object is not a mesh: {type(mesh)}")

            self.mesh = mesh
            self.original_mesh = mesh.copy()
            self.bounds = self._compute_bounds()

            # Reset transformation tracking on new load
            self.transform_stack.clear()
            self.M_world_from_local = np.eye(4, dtype=float)
            self.M_local_from_world = np.eye(4, dtype=float)

            if self.verbose:
                logger.info(
                    "Loaded mesh: %d vertices, %d faces",
                    len(mesh.vertices),
                    len(mesh.faces),
                )
                logger.debug("Bounds: %s", self.bounds)

            return mesh

        except Exception as e:
            raise ValueError(f"Failed to load mesh from {filepath}: {str(e)}")

    def save(self, filepath, file_format="obj"):
        self.mesh.export(filepath, file_type=file_format)

    def copy(self):
        return MeshManager(self.mesh.copy(), verbose=self.verbose)

    def to_trimesh(self):
        return self.mesh

    # =================================================================
    # TRANSFORM MANAGEMENT
    # =================================================================

    def _apply_and_record_transform(
        self,
        name: str,
        M: np.ndarray,
        *,
        params: Optional[Dict[str, Any]] = None,
        is_uniform_scale: bool = False,
        uniform_scale: Optional[float] = None,
    ) -> None:
        """Apply a 4x4 transform to the mesh and record it on the stack.

        This updates both the mesh vertices and the cumulative matrices.
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")
        M = np.asarray(M, dtype=float)
        if M.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")

        # Apply to mesh using trimesh's built-in method
        try:
            self.mesh.apply_transform(M)
        except Exception:
            # Fallback manual apply
            v = np.asarray(self.mesh.vertices, dtype=float)
            ones = np.ones((v.shape[0], 1), dtype=float)
            vh = np.hstack([v, ones])
            v2 = (M @ vh.T).T[:, :3]
            self.mesh.vertices = v2

        # Update bounds
        self.bounds = self._compute_bounds()

        # Update composite matrices
        self.M_world_from_local = M @ self.M_world_from_local
        try:
            self.M_local_from_world = np.linalg.inv(self.M_world_from_local)
        except Exception:
            # Maintain previous inverse if singular; still record transform
            pass

        # Record
        self.transform_stack.append(
            MeshManager.Transform(
                name=name,
                M=M.copy(),
                params=None if params is None else dict(params),
                timestamp=datetime.now(),
                is_uniform_scale=bool(is_uniform_scale),
                uniform_scale=float(uniform_scale) if uniform_scale is not None else None,
            )
        )

    def apply_transform(self, M: np.ndarray, *, name: str = "custom", params: Optional[Dict[str, Any]] = None) -> trimesh.Trimesh:
        """Public method to apply and record an arbitrary 4x4 transform."""
        self._apply_and_record_transform(name, M, params=params)
        return self.mesh  # type: ignore[return-value]

    def get_composite_matrix(self) -> np.ndarray:
        """Return the cumulative 4x4 matrix mapping local->world (applied order)."""
        return self.M_world_from_local.copy()

    def get_inverse_matrix(self) -> np.ndarray:
        """Return the inverse composite 4x4 matrix (world->local)."""
        return self.M_local_from_world.copy()

    def _compute_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Compute mesh bounding box."""
        if self.mesh is None:
            return None

        min_coords = self.mesh.vertices.min(axis=0)
        max_coords = self.mesh.vertices.max(axis=0)

        return {
            "x": (min_coords[0], max_coords[0]),
            "y": (min_coords[1], max_coords[1]),
            "z": (min_coords[2], max_coords[2]),
        }

    def get_z_range(self) -> Tuple[float, float]:
        """Get the z-axis range of the mesh."""
        if self.bounds is None:
            raise ValueError("No mesh loaded")
        return self.bounds["z"]

    def center_mesh(self, center_on: str = "centroid") -> trimesh.Trimesh:
        """
        Center the mesh.

        Args:
            center_on: 'centroid', 'bounds_center', or 'origin'

        Returns:
            Centered mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        if center_on == "centroid":
            center = self.mesh.centroid
        elif center_on == "bounds_center":
            center = self.mesh.bounds.mean(axis=0)
        elif center_on == "origin":
            center = np.array([0, 0, 0])
        else:
            raise ValueError(f"Unknown center_on option: {center_on}")

        # Build translation to move selected center to origin
        T = np.eye(4, dtype=float)
        T[:3, 3] = -np.asarray(center, dtype=float)
        self._apply_and_record_transform(
            "center",
            T,
            params={"center_on": center_on, "center": np.asarray(center, dtype=float).tolist()},
        )
        return self.mesh

    def scale_mesh(self, scale_factor: float) -> trimesh.Trimesh:
        """
        Scale the mesh uniformly.

        Args:
            scale_factor: Scaling factor

        Returns:
            Scaled mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        S = np.eye(4, dtype=float)
        sf = float(scale_factor)
        S[0, 0] = sf
        S[1, 1] = sf
        S[2, 2] = sf
        self._apply_and_record_transform(
            "normalize_scale",
            S,
            params={"scale_factor": sf},
            is_uniform_scale=True,
            uniform_scale=sf,
        )
        return self.mesh

    def align_with_z_axis(
        self, target_axis: Optional[np.ndarray] = None
    ) -> trimesh.Trimesh:
        """
        Align the mesh's principal axis with the z-axis.

        Args:
            target_axis: Target direction (default: [0, 0, 1])

        Returns:
            Aligned mesh
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        if target_axis is None:
            target_axis = np.array([0, 0, 1])

        # Compute principal axis using PCA
        vertices_centered = self.mesh.vertices - self.mesh.centroid
        covariance = np.cov(vertices_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Principal axis is the eigenvector with largest eigenvalue
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # Compute rotation to align principal axis with target
        rotation_matrix = self._rotation_matrix_between_vectors(
            principal_axis, target_axis
        )

        # Apply rotation about centroid to preserve position
        centroid = self.mesh.centroid
        R3 = rotation_matrix
        T1 = np.eye(4, dtype=float)
        T2 = np.eye(4, dtype=float)
        T1[:3, 3] = -centroid
        T2[:3, 3] = centroid
        R4 = np.eye(4, dtype=float)
        R4[:3, :3] = R3
        M = T2 @ R4 @ T1
        self._apply_and_record_transform(
            "rotate_to_z_principal_axis",
            M,
            params={
                "method": "PCA",
                "target_axis": np.asarray(target_axis, dtype=float).tolist()
                if target_axis is not None
                else [0.0, 0.0, 1.0],
                "centroid": centroid.tolist(),
            },
        )

        return self.mesh

    def _rotation_matrix_between_vectors(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> np.ndarray:
        """Compute rotation matrix to rotate vec1 to vec2."""
        # Normalize vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        # Check if vectors are already aligned
        if np.allclose(vec1, vec2):
            return np.eye(3)
        if np.allclose(vec1, -vec2):
            # 180-degree rotation - find perpendicular axis
            perp = np.array([1, 0, 0]) if abs(vec1[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(vec1, perp)
            axis = axis / np.linalg.norm(axis)
            return self._rodrigues_rotation(axis, np.pi)

        # General case using Rodrigues' formula
        axis = np.cross(vec1, vec2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(vec1, vec2), -1, 1))

        return self._rodrigues_rotation(axis, angle)

    def _rodrigues_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues' rotation formula."""
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    def _furthest_pair(self, points: np.ndarray) -> Tuple[int, int, np.ndarray]:
        """Return indices (i,j) of a furthest-apart pair and the vector points[j]-points[i]."""
        best_i = 0
        best_j = 1 if points.shape[0] > 1 else 0
        best_d2 = -1.0
        n = points.shape[0]
        for i in range(n - 1):
            diffs = points[i + 1 :] - points[i]
            if diffs.size == 0:
                continue
            d2 = np.einsum("ij,ij->i", diffs, diffs)
            if d2.size == 0:
                continue
            j_rel = int(np.argmax(d2))
            if d2[j_rel] > best_d2:
                best_d2 = float(d2[j_rel])
                best_i = i
                best_j = i + 1 + j_rel
        v = points[best_j] - points[best_i]
        return best_i, best_j, v

    def align_furthest_points_with_z(
        self,
        *,
        prefer_positive_z: bool = True,
        use_convex_hull: bool = True,
    ) -> trimesh.Trimesh:
        """
        Orient the mesh so the vector between the two furthest-apart points
        aligns with the +Z axis.

        This computes a diameter of the point set (by default using the convex
        hull vertices for efficiency), finds the pair of points with maximum
        Euclidean distance, and rotates the mesh about its centroid so that
        this vector is aligned with the positive z-axis.

        Args:
            prefer_positive_z: If True, align the diameter to +Z; otherwise the
                orientation to Z is arbitrary up to a sign.
            use_convex_hull: If True (default), compute the diameter over the
                convex hull vertices. This is typically much smaller than the
                full vertex set and yields the same diameter for convex sets.

        Returns:
            The rotated mesh (also stored in self.mesh).
        """
        if self.mesh is None:
            raise ValueError("No mesh loaded")

        # Select candidate points
        try:
            points = (
                self.mesh.convex_hull.vertices
                if use_convex_hull and hasattr(self.mesh, "convex_hull")
                else self.mesh.vertices
            )
        except Exception:
            points = self.mesh.vertices

        points = np.asarray(points, dtype=float)
        if points.shape[0] < 2:
            # Nothing to do
            return self.mesh

        # Find furthest pair using a vectorized per-anchor pass (O(n^2) time, O(n) mem)
        best_i = 0
        best_j = 1
        best_d2 = -1.0
        n = points.shape[0]
        for i in range(n - 1):
            diffs = points[i + 1 :] - points[i]
            if diffs.size == 0:
                continue
            d2 = np.einsum("ij,ij->i", diffs, diffs)
            j_rel = int(np.argmax(d2))
            if d2[j_rel] > best_d2:
                best_d2 = float(d2[j_rel])
                best_i = i
                best_j = i + 1 + j_rel

        p1 = points[best_i]
        p2 = points[best_j]
        v = p2 - p1
        norm = float(np.linalg.norm(v))
        if not np.isfinite(norm) or norm <= 1e-12:
            # Degenerate case; nothing to rotate
            return self.mesh

        if prefer_positive_z and v[2] < 0:
            v = -v

        # Compute rotation from v to +Z
        target = np.array([0.0, 0.0, 1.0], dtype=float)
        R = self._rotation_matrix_between_vectors(v, target)

        # Rotate about centroid to preserve position via 4x4
        centroid = self.mesh.centroid
        T1 = np.eye(4, dtype=float)
        T2 = np.eye(4, dtype=float)
        T1[:3, 3] = -centroid
        T2[:3, 3] = centroid
        R4 = np.eye(4, dtype=float)
        R4[:3, :3] = R
        M = T2 @ R4 @ T1
        self._apply_and_record_transform(
            "align_furthest_points_with_z",
            M,
            params={
                "prefer_positive_z": bool(prefer_positive_z),
                "use_convex_hull": bool(use_convex_hull),
                "centroid": centroid.tolist(),
            },
        )

        # Post-rotation correction: compute the furthest pair on the rotated
        # vertices and ensure that vector is aligned with +Z as well. This
        # guards against cases where the exact furthest pair differs slightly
        # from the one chosen for the initial rotation.
        try:
            all_pts = np.asarray(self.mesh.vertices, dtype=float)
            if all_pts.shape[0] >= 2:
                _, _, v2 = self._furthest_pair(all_pts)
                if prefer_positive_z and v2[2] < 0:
                    v2 = -v2
                v2n = v2 / (np.linalg.norm(v2) + 1e-12)
                if v2n[2] < 0.999:
                    R2 = self._rotation_matrix_between_vectors(v2, target)
                    T1b = np.eye(4, dtype=float)
                    T2b = np.eye(4, dtype=float)
                    T1b[:3, 3] = -centroid
                    T2b[:3, 3] = centroid
                    R4b = np.eye(4, dtype=float)
                    R4b[:3, :3] = R2
                    M2 = T2b @ R4b @ T1b
                    self._apply_and_record_transform(
                        "align_furthest_points_with_z_correction",
                        M2,
                        params={"centroid": centroid.tolist()},
                    )
        except Exception:
            # If anything goes wrong, we keep the initial rotation which already
            # aligns the chosen diameter to +Z.
            pass

        # Final enforcement of +Z direction if requested: if the furthest-pair
        # vector still points to -Z due to numerical or selection ambiguities,
        # rotate 180 degrees about the X-axis around the centroid.
        try:
            if prefer_positive_z:
                Vfinal = np.asarray(self.mesh.vertices, dtype=float)
                if Vfinal.shape[0] >= 2:
                    _, _, vf = self._furthest_pair(Vfinal)
                    if vf[2] < 0:
                        # 180-degree rotation about X-axis
                        Rx = np.eye(4, dtype=float)
                        Rx[1, 1] = -1.0
                        Rx[2, 2] = -1.0
                        T1c = np.eye(4, dtype=float)
                        T2c = np.eye(4, dtype=float)
                        T1c[:3, 3] = -centroid
                        T2c[:3, 3] = centroid
                        Mfix = T2c @ Rx @ T1c
                        self._apply_and_record_transform(
                            "align_furthest_points_with_z_sign_fix",
                            Mfix,
                            params={"centroid": centroid.tolist()},
                        )
        except Exception:
            pass

        return self.mesh

    # combining the functions from utils into this class

    def analyze_mesh(self) -> dict:
        """
        Analyze and return mesh properties for diagnostic purposes.
        This function performs pure analysis without modifying the input mesh.

        Returns:
            Dictionary of mesh properties including volume, watertightness, winding consistency,
            face count, vertex count, bounds, and potential issues.
        """

        mesh = self.to_trimesh()
        # Initialize results dictionary
        results = {
            "face_count": len(mesh.faces),
            "vertex_count": len(mesh.vertices),
            "bounds": mesh.bounds.tolist() if hasattr(mesh, "bounds") else None,
            "is_watertight": mesh.is_watertight,
            "is_winding_consistent": mesh.is_winding_consistent,
            "issues": [],
        }

        # Calculate volume (report actual value, even if negative)
        try:
            results["volume"] = mesh.volume
            if mesh.volume < 0:
                results["issues"].append(
                    "Negative volume detected - face normals may be inverted"
                )
        except Exception as e:
            results["volume"] = None
            results["issues"].append(f"Volume calculation failed: {str(e)}")

        # Check for non-manifold edges
        try:
            if hasattr(mesh, "is_manifold"):
                results["is_manifold"] = mesh.is_manifold
                if not mesh.is_manifold:
                    results["issues"].append("Non-manifold edges detected")
        except Exception:
            results["is_manifold"] = None

        # Calculate topological properties using trimesh's built-in methods
        try:
            # Use trimesh's built-in euler_number property for correct topology calculation
            # For a sphere: euler_number = 2
            # For a torus: euler_number = 0
            # For a double torus: euler_number = -2
            # Genus = (2 - euler_number) / 2

            results["euler_characteristic"] = mesh.euler_number

            # Only calculate genus for closed (watertight) meshes
            if mesh.is_watertight:
                # For a closed orientable surface: genus = (2 - euler_number) / 2
                results["genus"] = int((2 - mesh.euler_number) / 2)

                # Sanity check - genus should be non-negative for simple shapes
                if results["genus"] < 0:
                    results["genus"] = (
                        0  # Default to 0 for simple shapes like spheres, cylinders
                    )
                    results["issues"].append(
                        "Calculated negative genus, defaulting to 0"
                    )
            else:
                # For non-watertight meshes, genus is not well-defined
                results["genus"] = None
                results["issues"].append("Genus undefined for non-watertight mesh")
        except Exception as e:
            results["genus"] = None
            results["euler_characteristic"] = None
            results["issues"].append(f"Topology calculation failed: {str(e)}")

        # Analyze face normals
        try:
            if hasattr(mesh, "face_normals") and mesh.face_normals is not None:
                # Get statistics on face normal directions
                results["normal_stats"] = {
                    "mean": mesh.face_normals.mean(axis=0).tolist(),
                    "std": mesh.face_normals.std(axis=0).tolist(),
                    "sum": mesh.face_normals.sum(axis=0).tolist(),
                }

                # Check if normals are predominantly pointing inward (negative volume)
                if results.get("volume", 0) < 0:
                    results["normal_direction"] = "inward"
                else:
                    results["normal_direction"] = "outward"
        except Exception as e:
            results["normal_stats"] = None
            results["issues"].append(f"Normal analysis failed: {str(e)}")

        # Check for duplicate vertices and faces
        try:
            unique_verts = np.unique(mesh.vertices, axis=0)
            results["duplicate_vertices"] = len(mesh.vertices) - len(unique_verts)
            if results["duplicate_vertices"] > 0:
                results["issues"].append(
                    f"Found {results['duplicate_vertices']} duplicate vertices"
                )
        except Exception:
            results["duplicate_vertices"] = None

        # Check for degenerate faces (zero area)
        try:
            if hasattr(mesh, "area_faces"):
                degenerate_count = np.sum(mesh.area_faces < 1e-8)
                results["degenerate_faces"] = int(degenerate_count)
                if degenerate_count > 0:
                    results["issues"].append(
                        f"Found {degenerate_count} degenerate faces"
                    )
        except Exception:
            results["degenerate_faces"] = None

        # Check for connected components
        try:
            components = mesh.split(only_watertight=False)
            results["component_count"] = len(components)
            if len(components) > 1:
                results["issues"].append(
                    f"Mesh has {len(components)} disconnected components"
                )
        except Exception:
            results["component_count"] = None

        return results

    def print_mesh_analysis(self, verbose: bool = False) -> None:
        """
        Analyze a mesh and print a formatted report of its properties.

        Args:
            verbose: Whether to print detailed information
        """
        analysis = self.analyze_mesh()

        logger.info("Mesh Analysis Report")
        logger.info("====================")

        # Basic properties
        logger.info("\nGeometry:")
        logger.info("  * Vertices: %s", analysis['vertex_count'])
        logger.info("  * Faces: %s", analysis['face_count'])
        if analysis.get("component_count") is not None:
            logger.info("  * Components: %s", analysis['component_count'])
        if analysis.get("volume") is not None:
            logger.info("  * Volume: %.2f", analysis['volume'])
        if analysis.get("bounds") is not None:
            min_bound, max_bound = analysis["bounds"]
            logger.info(
                "  * Bounds: [%.1f, %.1f, %.1f] to [%.1f, %.1f, %.1f]",
                min_bound[0],
                min_bound[1],
                min_bound[2],
                max_bound[0],
                max_bound[1],
                max_bound[2],
            )

        # Mesh quality
        logger.info("\nMesh Quality:")
        logger.info("  * Watertight: %s", analysis['is_watertight'])
        logger.info("  * Winding Consistent: %s", analysis['is_winding_consistent'])
        if analysis.get("is_manifold") is not None:
            logger.info("  * Manifold: %s", analysis['is_manifold'])
        if analysis.get("normal_direction") is not None:
            logger.info("  * Normal Direction: %s", analysis['normal_direction'])
        if analysis.get("duplicate_vertices") is not None:
            logger.info("  * Duplicate Vertices: %s", analysis['duplicate_vertices'])
        if analysis.get("degenerate_faces") is not None:
            logger.info("  * Degenerate Faces: %s", analysis['degenerate_faces'])

        # Topology
        if (
            analysis.get("genus") is not None
            or analysis.get("euler_characteristic") is not None
        ):
            logger.info("\nTopology:")
            if analysis.get("genus") is not None:
                logger.info("  * Genus: %s", analysis['genus'])
            if analysis.get("euler_characteristic") is not None:
                logger.info("  * Euler Characteristic: %s", analysis['euler_characteristic'])

        # Issues
        if analysis["issues"]:
            logger.info("\nIssues Detected (%d):", len(analysis['issues']))
            for i, issue in enumerate(analysis["issues"]):
                logger.info("  %d. %s", i + 1, issue)
        else:
            logger.info("\nNo issues detected")

        # Detailed stats
        if verbose and analysis.get("normal_stats") is not None:
            logger.info("\nNormal Statistics:")
            mean = analysis["normal_stats"]["mean"]
            sum_val = analysis["normal_stats"]["sum"]
            logger.info("  * Mean: [%.4f, %.4f, %.4f]", mean[0], mean[1], mean[2])
            logger.info("  * Sum: [%.4f, %.4f, %.4f]", sum_val[0], sum_val[1], sum_val[2])

        logger.info("\nRecommendation:")
        if analysis["issues"]:
            logger.info("  Consider using repair_mesh() to fix the detected issues.")
        else:
            logger.info("  Mesh appears to be in good condition.")
        logger.info("====================")

    def repair_mesh(
        self,
        fix_holes: bool = True,
        remove_duplicates: bool = True,
        fix_normals: bool = True,
        remove_degenerate: bool = True,
        fix_negative_volume: bool = True,
        keep_largest_component: bool = False,
        verbose: bool = True,
    ) -> trimesh.Trimesh:
        """
        Attempt to repair common mesh issues to improve watertightness and quality.

        Args:
            mesh_data: Either a Trimesh object or (vertices, faces) tuple
            fix_holes: Whether to attempt filling holes
            remove_duplicates: Whether to remove duplicate faces and vertices
            fix_normals: Whether to fix face normal consistency
            remove_degenerate: Whether to remove degenerate faces
            fix_negative_volume: Whether to invert faces if mesh has negative volume
            keep_largest_component: Whether to keep only the largest connected component
            verbose: Whether to print repair summary

        Returns:
            Repaired mesh (new copy, original is not modified)
        """

        mesh = self.to_trimesh()

        repair_log = []

        # Fix negative volume by inverting faces if needed
        if fix_negative_volume:
            try:
                # Check if the mesh has a negative volume
                if hasattr(mesh, "volume") and mesh.volume < 0:
                    initial_volume = mesh.volume
                    mesh.invert()
                    repair_log.append(
                        f"Inverted faces to fix negative volume: {initial_volume:.2f} → {mesh.volume:.2f}"
                    )
            except Exception as e:
                repair_log.append(f"Failed to fix negative volume: {e}")

        # Remove duplicate and degenerate faces
        if remove_duplicates:
            try:
                initial_faces = len(mesh.faces)
                mesh.remove_duplicate_faces()
                removed_faces = initial_faces - len(mesh.faces)
                if removed_faces > 0:
                    repair_log.append(f"Removed {removed_faces} duplicate faces")
            except Exception as e:
                repair_log.append(f"Failed to remove duplicate faces: {e}")

        if remove_degenerate:
            try:
                initial_faces = len(mesh.faces)
                mesh.remove_degenerate_faces()
                removed_faces = initial_faces - len(mesh.faces)
                if removed_faces > 0:
                    repair_log.append(f"Removed {removed_faces} degenerate faces")
            except Exception as e:
                repair_log.append(f"Failed to remove degenerate faces: {e}")

        # Fix winding consistency
        if fix_normals:
            try:
                if not mesh.is_winding_consistent:
                    mesh.fix_normals()
                    if mesh.is_winding_consistent:
                        repair_log.append("Fixed face normal winding consistency")
                    else:
                        repair_log.append(
                            "Attempted to fix normals but still inconsistent"
                        )
            except Exception as e:
                repair_log.append(f"Failed to fix normals: {e}")

        # Attempt to fill holes
        if fix_holes:
            try:
                if not mesh.is_watertight:
                    initial_watertight = mesh.is_watertight
                    mesh.fill_holes()
                    if mesh.is_watertight and not initial_watertight:
                        repair_log.append(
                            "Successfully filled holes - mesh is now watertight"
                        )
                    elif mesh.is_watertight:
                        repair_log.append("Mesh was already watertight")
                    else:
                        repair_log.append(
                            "Attempted to fill holes but mesh still not watertight"
                        )
            except Exception as e:
                repair_log.append(f"Failed to fill holes: {e}")

        # Keep only the largest component if requested
        if keep_largest_component:
            try:
                components = mesh.split(only_watertight=False)
                if len(components) > 1:
                    # Keep the largest component by volume or face count
                    volumes = [
                        abs(c.volume) if hasattr(c, "volume") else len(c.faces)
                        for c in components
                    ]
                    largest_idx = np.argmax(volumes)
                    mesh = components[largest_idx]
                    repair_log.append(
                        f"Kept largest of {len(components)} components (volume: {volumes[largest_idx]:.2f})"
                    )
            except Exception as e:
                repair_log.append(f"Failed to isolate largest component: {e}")

        # Final processing to ensure consistency
        try:
            mesh.process(validate=True)
            repair_log.append("Applied final mesh processing and validation")
        except Exception as e:
            repair_log.append(f"Final processing failed: {e}")

        # Store repair log as mesh metadata
        if not hasattr(mesh, "metadata"):
            mesh.metadata = {}
        mesh.metadata["repair_log"] = repair_log

        # Log repair summary
        if verbose:
            if repair_log:
                logger.info("Mesh Repair Summary:")
                for log_entry in repair_log:
                    logger.info("  • %s", log_entry)

                # Final mesh status
                logger.info("\nFinal Mesh Status:")
                logger.info(
                    "  • Volume: %s",
                    mesh.volume if hasattr(mesh, "volume") else "N/A",
                )
                logger.info("  • Watertight: %s", mesh.is_watertight)
                logger.info("  • Winding consistent: %s", mesh.is_winding_consistent)
                logger.info("  • Faces: %d", len(mesh.faces))
                logger.info("  • Vertices: %d", len(mesh.vertices))
            else:
                logger.info("No repairs needed - mesh is in good condition")

        self.mesh = mesh
        return mesh


    def visualize_mesh_3d(
        self,
        title: str = "3D Mesh Visualization",
        color: str = "lightblue",
        backend: str = "auto",
        show_axes: bool = True,
        show_wireframe: bool = False,
        width: int = 800,
        height: int = 600,
    ) -> Optional[object]:
        """
        Create a 3D visualization of a mesh.

        Args:
            title: Plot title
            color: Mesh color (named color or RGB tuple)
            backend: Visualization backend ('plotly' or 'matplotlib')
            show_axes: Whether to show coordinate axes
            show_wireframe: Whether to show wireframe overlay

        Returns:
            Figure object (backend-dependent) or None if visualization fails
        """
        if backend == "auto":
            # Try plotly first, then fallback to others
            try:
                import plotly.graph_objects as go

                backend = "plotly"
            except ImportError:
                try:
                    import matplotlib.pyplot as plt

                    backend = "matplotlib"
                except ImportError:
                    backend = "plotly"

        if backend == "plotly":
            return self._visualize_mesh_plotly(
                title, color, show_axes, show_wireframe, width, height
            )
        elif backend == "matplotlib":
            return self._visualize_mesh_matplotlib(
                title, color, show_axes, show_wireframe
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _visualize_mesh_plotly(
        self, title, color, show_axes, show_wireframe, width=800, height=600
    ):
        """Plotly-based mesh visualization."""
        try:
            import plotly.graph_objects as go

            vertices = self.mesh.vertices
            faces = self.mesh.faces

            # Create mesh trace
            mesh_trace = go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.8,
                color=color,
                name="Mesh",
            )

            fig = go.Figure(data=[mesh_trace])

            # Add wireframe if requested
            if show_wireframe:
                # Create wireframe edges
                edge_trace = []
                for face in faces:
                    for i in range(3):
                        v1, v2 = face[i], face[(i + 1) % 3]
                        edge_trace.extend(
                            [
                                vertices[v1][0],
                                vertices[v2][0],
                                None,
                                vertices[v1][1],
                                vertices[v2][1],
                                None,
                                vertices[v1][2],
                                vertices[v2][2],
                                None,
                            ]
                        )

                if edge_trace:
                    fig.add_trace(
                        go.Scatter3d(
                            x=edge_trace[::3],
                            y=edge_trace[1::3],
                            z=edge_trace[2::3],
                            mode="lines",
                            line=dict(color="black", width=1),
                            name="Wireframe",
                        )
                    )

            # Configure layout
            fig.update_layout(
                title=title,
                autosize=False,
                width=width,
                height=height,
                scene=dict(
                    aspectmode="data",
                    xaxis=dict(visible=show_axes),
                    yaxis=dict(visible=show_axes),
                    zaxis=dict(visible=show_axes),
                ),
            )

            return fig

        except ImportError:
            print("Plotly not available")
            return None
        except Exception as e:
            print(f"Plotly visualization failed: {e}")
            return None

    def _visualize_mesh_matplotlib(self, title, color, show_axes, show_wireframe):
        """Matplotlib-based mesh visualization."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            vertices = self.mesh.vertices
            faces = self.mesh.faces

            # Create mesh surface
            poly3d = Poly3DCollection(
                vertices[faces],
                alpha=0.7,
                facecolor=color,
                edgecolor="black" if show_wireframe else None,
            )
            ax.add_collection3d(poly3d)

            ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
            ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
            ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

            ax.set_xlabel("X (µm)")
            ax.set_ylabel("Y (µm)")
            ax.set_zlabel("Z (µm)")
            ax.set_title(title)

            if not show_axes:
                ax.set_axis_off()

            plt.tight_layout()
            return fig

        except ImportError:
            print("Matplotlib not available")
            return None
        except Exception as e:
            print(f"Matplotlib visualization failed: {e}")
            return None

    def visualize_mesh_slice_interactive(
        self,
        title: str = "Interactive Mesh Slice",
        z_range: Optional[Tuple[float, float]] = None,
        num_slices: int = 50,
        slice_color: str = "red",
        mesh_color: str = "lightblue",
        mesh_opacity: float = 0.3,
    ) -> Optional[object]:
        """
        Create an interactive 3D visualization of a mesh with a controllable slice plane.

        This function displays a 3D mesh and calculates the intersection of the mesh
        with an xy-plane at a user-controlled z-value. The intersection is shown as a
        colored line on the mesh. A slider allows the user to interactively change the
        z-value of the intersection plane.

        Args:
            title: Plot title
            z_range: Tuple of (min_z, max_z) for slice range. Auto-detected if None.
            num_slices: Number of positions for the slider
            slice_color: Color for the intersection line
            mesh_color: Color for the 3D mesh
            mesh_opacity: Opacity of the 3D mesh (0-1)

        Returns:
            Plotly figure with interactive slider for controlling the z-value
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly is required for interactive visualization")
            return None

        mesh = self.mesh

        # Determine z-range if not provided
        if z_range is None:
            z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
            # Add small padding
            padding = (z_max - z_min) * 0.05
            z_min -= padding
            z_max += padding
        else:
            z_min, z_max = z_range

        # Create the base figure with the mesh
        fig = go.Figure()

        # Add the mesh to the figure
        fig.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                opacity=mesh_opacity,
                color=mesh_color,
                name="Mesh",
            )
        )

        # Function to create a slice at a given z-value
        def create_slice_trace(z_value):
            # Calculate intersection with plane at z_value
            section = mesh.section(plane_origin=[0, 0, z_value], plane_normal=[0, 0, 1])

            # If no intersection, return None
            if (
                section is None
                or not hasattr(section, "entities")
                or len(section.entities) == 0
            ):
                return None

            # Process all entities in the section to get 3D coordinates
            all_points = []

            for entity in section.entities:
                if hasattr(entity, "points") and len(entity.points) > 0:
                    # Get the actual 2D coordinates
                    points_2d = section.vertices[entity.points]

                    # Convert to 3D by adding z_value
                    points_3d = np.column_stack(
                        [points_2d, np.full(len(points_2d), z_value)]
                    )

                    # Add closing point if needed (to complete the loop)
                    if len(points_2d) > 2 and not np.array_equal(
                        points_2d[0], points_2d[-1]
                    ):
                        closing_point = np.array(
                            [points_2d[0][0], points_2d[0][1], z_value]
                        )
                        points_3d = np.vstack([points_3d, closing_point])

                    # Add to all points list
                    all_points.extend(points_3d.tolist())

                    # Add None to create a break between separate entities
                    all_points.append([None, None, None])

            # If we have points, create a scatter trace
            if all_points:
                x_coords = [p[0] if p is not None else None for p in all_points]
                y_coords = [p[1] if p is not None else None for p in all_points]
                z_coords = [p[2] if p is not None else None for p in all_points]

                return go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="lines",
                    line=dict(color=slice_color, width=5),
                    name=f"Slice at z={z_value:.2f}",
                )

            return None

        # Create initial slice
        initial_z = (z_min + z_max) / 2
        initial_slice = create_slice_trace(initial_z)

        # Add initial slice to figure if it exists
        if initial_slice:
            fig.add_trace(initial_slice)

        # Create frames for animation
        frames = []
        for i, z_val in enumerate(np.linspace(z_min, z_max, num_slices)):
            # Create a slice at this z-value
            slice_trace = create_slice_trace(z_val)

            # If we have a valid slice, add it to frames
            if slice_trace:
                frame_data = [fig.data[0], slice_trace]  # Mesh and slice
            else:
                frame_data = [fig.data[0]]  # Just the mesh

            frames.append(
                go.Frame(
                    data=frame_data,
                    name=f"frame_{i}",
                    traces=[0, 1],  # Update both traces
                )
            )

        # Create slider steps
        steps = []
        for i, z_val in enumerate(np.linspace(z_min, z_max, num_slices)):
            step = dict(
                args=[
                    [f"frame_{i}"],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                ],
                label=f"{z_val:.2f}",
                method="animate",
            )
            steps.append(step)

        # Configure the slider
        sliders = [
            dict(
                active=num_slices // 2,  # Start in the middle
                currentvalue={
                    "prefix": "Z-value: ",
                    "visible": True,
                    "xanchor": "right",
                },
                pad={"t": 50, "b": 10},
                len=0.9,
                x=0.1,
                y=0,
                steps=steps,
            )
        ]

        # Configure the figure layout
        fig.update_layout(
            title=title,
            scene=dict(aspectmode="data", camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
            height=800,  # Taller to make room for slider
            margin=dict(l=50, r=50, b=100, t=100),  # Add margin at bottom for slider
            sliders=sliders,
            # Add animation controls
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0,
                    xanchor="left",
                    yanchor="top",
                    pad=dict(t=60, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        ),
                        dict(
                            label="Reset View",
                            method="relayout",
                            args=[{"scene.camera.eye": dict(x=1.5, y=1.5, z=1.5)}],
                        ),
                    ],
                )
            ],
        )

        # Set frames
        fig.frames = frames

        return fig

    def visualize_mesh_slice_grid(
        self,
        vertices: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
        title: str = "Mesh Slice Grid",
        num_slices: int = 9,
        z_range: Optional[Tuple[float, float]] = None,
    ) -> Optional[object]:
        """
        Create a grid visualization showing multiple cross-sections of a 3D mesh.

        Args:
            vertices: Vertex array (alternative to mesh_data)
            faces: Face array (alternative to mesh_data)
            title: Plot title
            num_slices: Number of slices to show (should be perfect square for grid)
            z_range: Tuple of (min_z, max_z) for slice range. Auto-detected if None.

        Returns:
            Plotly figure with subplot grid
        """
        try:
            import math

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not available for slice grid visualization")
            return None

        mesh = self.mesh

        # Determine grid size
        grid_size = int(math.sqrt(num_slices))
        if grid_size * grid_size != num_slices:
            grid_size = int(math.ceil(math.sqrt(num_slices)))
            num_slices = grid_size * grid_size

        # Determine Z range
        if z_range is None:
            z_min, z_max = mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()
        else:
            z_min, z_max = z_range

        # Create Z levels
        z_levels = np.linspace(z_min, z_max, num_slices)

        # Create subplots
        fig = make_subplots(
            rows=grid_size,
            cols=grid_size,
            subplot_titles=[f"Z = {z:.2f}" for z in z_levels],
            specs=[[{"type": "xy"}] * grid_size for _ in range(grid_size)],
        )

        # Generate slices and add to subplots
        for i, z_level in enumerate(z_levels):
            row = i // grid_size + 1
            col = i % grid_size + 1

            try:
                # Get 2D cross-section
                slice_2d = mesh.section(
                    plane_origin=[0, 0, z_level], plane_normal=[0, 0, 1]
                )

                if (
                    slice_2d is not None
                    and hasattr(slice_2d, "entities")
                    and len(slice_2d.entities) > 0
                ):
                    # Plot each entity in the slice
                    for entity in slice_2d.entities:
                        if hasattr(entity, "points"):
                            points = slice_2d.vertices[entity.points]
                            # Close the loop
                            points_closed = np.vstack([points, points[0]])

                            fig.add_trace(
                                go.Scatter(
                                    x=points_closed[:, 0],
                                    y=points_closed[:, 1],
                                    mode="lines",
                                    line=dict(color="red", width=2),
                                    showlegend=False,
                                ),
                                row=row,
                                col=col,
                            )

                # Set equal aspect ratio for each subplot
                fig.update_xaxes(scaleanchor="y", scaleratio=1, row=row, col=col)
                fig.update_xaxes(title_text="X (µm)", row=row, col=col)
                fig.update_yaxes(title_text="Y (µm)", row=row, col=col)

            except Exception as e:
                # If slicing fails, just leave subplot empty
                pass

        fig.update_layout(
            title=title,
            height=150 * grid_size,
            showlegend=False,
        )

        return fig

