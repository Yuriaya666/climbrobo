"""从现有机器人 STL 标定参数化碰撞代理尺寸。

该模块只读取 ``robots_model/meshes``，不修改 URDF 或 STL。连杆半径使用
每个 STL 的 PCA 主轴横截面包络；关节 proxy 使用对应 child-link 原点附近
的局部 STL 邻域稳健包络。由于仓库没有独立电机 CAD，关节尺寸明确标记为
``mesh_local_envelope``，不是电机外壳的精确 CAD 测量。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh


JOINT_NEIGHBORHOOD_RADIUS_M = 0.055
JOINT_QUANTILE = 0.99
JOINT_PROXY_AUDIT_RADIUS_M = 0.040


@dataclass(frozen=True)
class MeshMeasurement:
    """一个 STL 的可复核几何测量。"""

    name: str
    path: Path
    vertex_count: int
    local_bounds_m: np.ndarray
    pca_extents_m: np.ndarray
    pca_axis_world_m: np.ndarray
    axial_length_m: float
    cross_section_width_m: float
    cross_section_height_m: float
    equivalent_radius_m: float
    radial_max_m: float
    radial_p99_m: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "vertex_count": self.vertex_count,
            "local_bounds_m": self.local_bounds_m.tolist(),
            "pca_extents_m": self.pca_extents_m.tolist(),
            "pca_axis_m": self.pca_axis_world_m.tolist(),
            "axial_length_m": self.axial_length_m,
            "cross_section_width_m": self.cross_section_width_m,
            "cross_section_height_m": self.cross_section_height_m,
            "equivalent_radius_m": self.equivalent_radius_m,
            "radial_max_m": self.radial_max_m,
            "radial_p99_m": self.radial_p99_m,
        }


@dataclass(frozen=True)
class JointLocalMeasurement:
    """child-link 原点附近的关节 proxy 测量。"""

    joint_name: str
    child_link: str
    mesh_name: str
    sample_count: int
    neighborhood_radius_m: float
    quantile: float
    raw_radius_m: float
    max_radius_m: float

    def as_dict(self) -> dict[str, object]:
        return {
            "joint_name": self.joint_name,
            "child_link": self.child_link,
            "mesh_name": self.mesh_name,
            "sample_count": self.sample_count,
            "neighborhood_radius_m": self.neighborhood_radius_m,
            "quantile": self.quantile,
            "raw_radius_m": self.raw_radius_m,
            "max_radius_m": self.max_radius_m,
            "source": "child-link STL vertices within Euclidean neighborhood of link origin",
        }


@dataclass(frozen=True)
class ProxyCalibration:
    """对称参数化模型所需的完整尺寸 profile。"""

    mesh_measurements: dict[str, MeshMeasurement]
    joint_measurements: dict[str, JointLocalMeasurement]
    full_link_radii_m: np.ndarray
    full_joint_radii_m: np.ndarray
    link_sources: tuple[str, ...]
    joint_sources: tuple[str, ...]

    def for_active_indices(self, active_indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(active_indices, dtype=int)
        return self.full_link_radii_m[indices].copy(), self.full_joint_radii_m[indices].copy()

    def dimension_rows(self, *, collision_inflation_m: float) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        link_pairs = (("link_1", "L3", "L5"), ("link_2", "L2", "L6"), ("link_3", "L1", "L7"), ("terminal", "base_link", "L8"))
        for component, left_name, right_name in link_pairs:
            left = self.mesh_measurements[left_name]
            right = self.mesh_measurements[right_name]
            radius = float(max(left.equivalent_radius_m, right.equivalent_radius_m))
            rows.append(
                {
                    "component": component,
                    "source_geometry": f"{left_name}.STL + {right_name}.STL",
                    "proxy_type": "Capsule" if component != "terminal" else "endpoint mesh / optional terminal capsule",
                    "kinematic_length": "design variable L1/L2/L3/L4",
                    "physical_proxy_length": "current parameterized segment length",
                    "nominal_radius_m": radius,
                    "safety_inflation_m": float(collision_inflation_m),
                    "effective_radius_m": radius + float(collision_inflation_m),
                    "source_derivation": "PCA main axis; max orthogonal extent / 2, mirrored pair max",
                }
            )
        joint_pairs = (("joint_1", "J4", "J5"), ("joint_2", "J3", "J6"), ("joint_3", "J2", "J7"), ("joint_4", "J1", "J8"))
        for component, left_name, right_name in joint_pairs:
            left = self.joint_measurements[left_name]
            right = self.joint_measurements[right_name]
            radius = float(self.full_joint_radii_m[int(component[-1]) - 1])
            rows.append(
                {
                    "component": component,
                    "source_geometry": f"{left.child_link}.STL + {right.child_link}.STL",
                    "proxy_type": "Sphere",
                    "kinematic_length": "point at joint center",
                    "physical_proxy_length": "n/a",
                    "nominal_radius_m": radius,
                    "safety_inflation_m": float(collision_inflation_m),
                    "effective_radius_m": radius + float(collision_inflation_m),
                    "source_derivation": "local STL 99th-percentile upper bound, capped at 40 mm raw because motor housing is not separable in the STL; mirrored pair and baseline joint-spacing audit",
                }
            )
        return rows


def _as_mesh(path: Path):
    mesh = trimesh.load_mesh(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


@lru_cache(maxsize=16)
def _measure_mesh_cached(path_text: str) -> MeshMeasurement:
    path = Path(path_text)
    mesh = _as_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"{path} 的 STL 顶点无效")
    centered = vertices - vertices.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(vertices))
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order]
    # 统一 PCA 轴符号只为让报告稳定，不影响尺寸。
    for index in range(3):
        largest = int(np.argmax(np.abs(axes[:, index])))
        if axes[largest, index] < 0.0:
            axes[:, index] *= -1.0
    projected = centered @ axes
    extents = projected.max(axis=0) - projected.min(axis=0)
    radial = np.linalg.norm(projected[:, 1:], axis=1)
    return MeshMeasurement(
        name=path.stem,
        path=path,
        vertex_count=int(len(vertices)),
        local_bounds_m=np.asarray(mesh.bounds, dtype=float),
        pca_extents_m=np.asarray(extents, dtype=float),
        pca_axis_world_m=np.asarray(axes[:, 0], dtype=float),
        axial_length_m=float(extents[0]),
        cross_section_width_m=float(extents[1]),
        cross_section_height_m=float(extents[2]),
        equivalent_radius_m=float(max(extents[1], extents[2]) / 2.0),
        radial_max_m=float(radial.max()),
        radial_p99_m=float(np.quantile(radial, 0.99)),
    )


def measure_mesh(path: Path) -> MeshMeasurement:
    return _measure_mesh_cached(str(path.resolve()))


@lru_cache(maxsize=16)
def _joint_measurement_cached(mesh_path_text: str, joint_name: str, child_link: str) -> JointLocalMeasurement:
    path = Path(mesh_path_text)
    mesh = _as_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=float)
    distances = np.linalg.norm(vertices, axis=1)
    selected = vertices[distances <= JOINT_NEIGHBORHOOD_RADIUS_M]
    if len(selected) < 16:
        raise ValueError(f"{joint_name} 的 {child_link}.STL 在关节局部邻域内样本不足")
    local_radius = np.linalg.norm(selected, axis=1)
    return JointLocalMeasurement(
        joint_name=joint_name,
        child_link=child_link,
        mesh_name=path.name,
        sample_count=int(len(selected)),
        neighborhood_radius_m=JOINT_NEIGHBORHOOD_RADIUS_M,
        quantile=JOINT_QUANTILE,
        raw_radius_m=float(np.quantile(local_radius, JOINT_QUANTILE)),
        max_radius_m=float(local_radius.max()),
    )


def calibrate_proxy_dimensions(mesh_dir: Path) -> ProxyCalibration:
    names = ("base_link", *(f"L{i}" for i in range(1, 9)))
    measurements = {
        name: measure_mesh(mesh_dir / f"{name}.STL") for name in names
    }
    child_by_joint = {f"J{i}": f"L{i}" for i in range(1, 9)}
    joints = {
        joint: _joint_measurement_cached(
            str((mesh_dir / f"{child}.STL").resolve()), joint, child
        )
        for joint, child in child_by_joint.items()
    }
    link_pairs = (("L3", "L5"), ("L2", "L6"), ("L1", "L7"), ("base_link", "L8"))
    full_link = np.asarray(
        [max(measurements[left].equivalent_radius_m, measurements[right].equivalent_radius_m) for left, right in link_pairs],
        dtype=float,
    )
    joint_pairs = (("J4", "J5"), ("J3", "J6"), ("J2", "J7"), ("J1", "J8"))
    # The local mesh measurement is an upper bound on the geometry around a
    # joint origin, but the STL does not identify the motor housing separately
    # from the outgoing arm.  Cap the usable spherical motor proxy at the
    # previously audited 40 mm raw envelope; the safety inflation remains an
    # independent parameter.
    full_joint = np.asarray(
        [
            min(max(joints[left].raw_radius_m, joints[right].raw_radius_m), JOINT_PROXY_AUDIT_RADIUS_M)
            for left, right in joint_pairs
        ],
        dtype=float,
    )
    return ProxyCalibration(
        mesh_measurements=measurements,
        joint_measurements=joints,
        full_link_radii_m=full_link,
        full_joint_radii_m=full_joint,
        link_sources=tuple(f"{left}.STL + {right}.STL" for left, right in link_pairs),
        joint_sources=tuple(f"{left} local + {right} local" for left, right in joint_pairs),
    )
