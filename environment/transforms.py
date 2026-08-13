from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def normalize(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    """单位化向量，并显式拒绝零向量。"""

    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError(f"{name}不能是零向量")
    return value / norm


def project_onto_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """把一个方向投影到法向为normal的平面内。"""

    normal = normalize(normal, name="plane normal")
    vector = np.asarray(vector, dtype=float)
    return vector - float(np.dot(vector, normal)) * normal


def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """将旋转矩阵转为PyBullet使用的[x, y, z, w]四元数。"""

    m = np.asarray(rotation_matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError("rotation_matrix必须是3x3矩阵")

    trace = float(np.trace(m))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m[2, 1] - m[1, 2]) / scale
        qy = (m[0, 2] - m[2, 0]) / scale
        qz = (m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / scale
        qx = 0.25 * scale
        qy = (m[0, 1] + m[1, 0]) / scale
        qz = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / scale
        qx = (m[0, 1] + m[1, 0]) / scale
        qy = 0.25 * scale
        qz = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / scale
        qx = (m[0, 2] + m[2, 0]) / scale
        qy = (m[1, 2] + m[2, 1]) / scale
        qz = 0.25 * scale

    quaternion = np.array([qx, qy, qz, qw], dtype=float)
    return normalize(quaternion, name="quaternion")


def quaternion_to_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """将[x, y, z, w]四元数转成3x3旋转矩阵。"""

    x, y, z, w = normalize(quaternion_xyzw, name="quaternion")
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def build_frame_from_z_and_y_reference(
    z_axis: np.ndarray,
    y_reference: np.ndarray,
) -> np.ndarray:
    """
    由Z轴和面内Y参考方向构造右手坐标系。

    返回矩阵的三列依次是X、Y、Z轴在父坐标系下的方向。
    """

    z_axis = normalize(z_axis, name="z_axis")
    y_projected = project_onto_plane(y_reference, z_axis)
    y_axis = normalize(y_projected, name="projected y_reference")

    x_axis = normalize(np.cross(y_axis, z_axis), name="x_axis")
    y_axis = normalize(np.cross(z_axis, x_axis), name="y_axis")
    return np.column_stack((x_axis, y_axis, z_axis))


def rotate_vector_about_axis(
    vector: np.ndarray,
    axis: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """用Rodrigues公式绕单位轴旋转一个向量。"""

    axis = normalize(axis, name="rotation axis")
    vector = np.asarray(vector, dtype=float)
    cos_value = math.cos(angle_rad)
    sin_value = math.sin(angle_rad)
    return (
        vector * cos_value
        + np.cross(axis, vector) * sin_value
        + axis * float(np.dot(axis, vector)) * (1.0 - cos_value)
    )


def angle_between_vectors_rad(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个方向之间的夹角，返回弧度。"""

    a = normalize(a, name="first direction")
    b = normalize(b, name="second direction")
    dot_value = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.acos(dot_value))


@dataclass(frozen=True)
class RigidTransform:
    """刚体变换，表示 T_parent_child。"""

    position: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        quaternion = normalize(self.quaternion_xyzw, name="transform quaternion")
        if position.shape != (3,):
            raise ValueError("position必须是长度为3的向量")
        if quaternion.shape != (4,):
            raise ValueError("quaternion_xyzw必须是长度为4的向量")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_xyzw", quaternion)

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(
            position=np.zeros(3, dtype=float),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        )

    @classmethod
    def from_rotation_matrix(
        cls,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> "RigidTransform":
        return cls(
            position=np.asarray(position, dtype=float),
            quaternion_xyzw=rotation_matrix_to_quaternion(rotation_matrix),
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quaternion_to_rotation_matrix(self.quaternion_xyzw)

    @property
    def x_axis(self) -> np.ndarray:
        return self.rotation_matrix[:, 0]

    @property
    def y_axis(self) -> np.ndarray:
        return self.rotation_matrix[:, 1]

    @property
    def z_axis(self) -> np.ndarray:
        return self.rotation_matrix[:, 2]

    def inverse(self) -> "RigidTransform":
        rotation_inv = self.rotation_matrix.T
        position_inv = -rotation_inv @ self.position
        return RigidTransform.from_rotation_matrix(position_inv, rotation_inv)

    def multiply(self, other: "RigidTransform") -> "RigidTransform":
        rotation = self.rotation_matrix @ other.rotation_matrix
        position = self.position + self.rotation_matrix @ other.position
        return RigidTransform.from_rotation_matrix(position, rotation)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        return self.position + self.rotation_matrix @ np.asarray(point, dtype=float)

    def as_pybullet(self) -> tuple[list[float], list[float]]:
        return self.position.tolist(), self.quaternion_xyzw.tolist()
