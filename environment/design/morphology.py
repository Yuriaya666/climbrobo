"""对称6R/8R机构的参数化运动学描述。

本模块不修改现有URDF。它从当前URDF和吸盘配置提取中央本体、关节轴、
关节限位和零位几何，再建立左右镜像的参数化侧链。新设计的连杆只在这里
使用长度参数和后续的Capsule碰撞代理表示。
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from environment.paths import ProjectPaths
from environment.design.proxy_calibration import calibrate_proxy_dimensions
from environment.suction_frames import SuctionFrameSet
from environment.transforms import RigidTransform, normalize, rotation_matrix_to_quaternion


# The link and joint proxies represent different physical envelopes.  The
# former is taken from the observed arm cross-section; the latter is an audit
# envelope constrained by the 96.702 mm baseline joint spacing.  It is not a
# substitute for a motor CAD model and remains configurable per candidate.
DEFAULT_LINK_PROXY_RADIUS_M = 0.065
DEFAULT_JOINT_PROXY_RADIUS_M = 0.040


def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """按URDF固定轴顺序生成roll-pitch-yaw旋转矩阵。"""

    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues旋转矩阵。"""

    unit = normalize(axis, name="joint axis")
    x, y, z = unit
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3) * math.cos(angle_rad)
        + (1.0 - math.cos(angle_rad)) * np.outer(unit, unit)
        + math.sin(angle_rad) * skew
    )


def _pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(rotation, dtype=float)
    result[:3, 3] = np.asarray(position, dtype=float)
    return result


def _pose_from_transform(transform: RigidTransform) -> np.ndarray:
    return _pose_matrix(transform.position, transform.rotation_matrix)


def _transform_from_pose(pose: np.ndarray) -> RigidTransform:
    return RigidTransform.from_rotation_matrix(pose[:3, 3], pose[:3, :3])


def _inverse_pose(pose: np.ndarray) -> np.ndarray:
    rotation = pose[:3, :3]
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ pose[:3, 3]
    return result


@dataclass(frozen=True)
class JointTemplate:
    """一个可保留关节的轴和真实限位。"""

    name: str
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class BranchTemplate:
    """从中央本体向一个吸盘端展开的零位侧链模板。"""

    side: str
    endpoint_name: str
    joint_templates: tuple[JointTemplate, ...]
    # body frame到第一关节frame的固定安装变换。
    mount_pose: np.ndarray
    # 每个关节旋转之后，到下一关节/端部参考frame的零位平移方向。
    span_directions: np.ndarray
    nominal_lengths_m: np.ndarray
    endpoint_link_to_suction: np.ndarray
    endpoint_mesh_path: Path

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError(f"未知侧链名称：{self.side}")
        n = len(self.joint_templates)
        if n == 0:
            raise ValueError("侧链至少需要一个关节")
        if self.span_directions.shape != (n, 3):
            raise ValueError("span_directions必须是(n,3)")
        if self.nominal_lengths_m.shape != (n,):
            raise ValueError("nominal_lengths_m必须是(n,)")
        if np.any(self.nominal_lengths_m <= 0.0):
            raise ValueError("侧链杆长必须大于0")
        norms = np.linalg.norm(self.span_directions, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError("侧链平移方向不能有零向量")


@dataclass(frozen=True)
class BaselineGeometry:
    """从当前真实URDF提取出的机构基线几何。"""

    central_link_name: str
    central_mesh_path: Path
    central_mass_kg: float
    left: BranchTemplate
    right: BranchTemplate
    original_joint_names: tuple[str, ...]
    original_joint_limits: dict[str, tuple[float, float]]

    @classmethod
    def from_project(cls, paths: ProjectPaths) -> "BaselineGeometry":
        """读取URDF，并按当前左右4R串联表达提取模板。"""

        tree = ET.parse(paths.robot_urdf)
        root = tree.getroot()
        links: dict[str, ET.Element] = {
            link.attrib["name"]: link for link in root.findall("link")
        }
        joints: dict[str, ET.Element] = {
            joint.attrib["name"]: joint for joint in root.findall("joint")
        }
        expected = [f"J{i}" for i in range(1, 9)]
        missing = [name for name in expected if name not in joints]
        if missing:
            raise ValueError(f"当前URDF缺少基线关节：{missing}")
        if "L4" not in links:
            raise ValueError("当前URDF中找不到中央本体link L4")

        frames = SuctionFrameSet.load(paths.suction_config)

        def parse_origin(joint_name: str) -> RigidTransform:
            origin = joints[joint_name].find("origin")
            if origin is None:
                return RigidTransform.identity()
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            if xyz.shape != (3,) or rpy.shape != (3,):
                raise ValueError(f"{joint_name}的origin不是长度为3的xyz/rpy")
            return RigidTransform.from_rotation_matrix(xyz, _rotation_from_rpy(rpy))

        def parse_joint(name: str) -> JointTemplate:
            joint = joints[name]
            axis_element = joint.find("axis")
            limit_element = joint.find("limit")
            if axis_element is None or limit_element is None:
                raise ValueError(f"{name}缺少axis或limit")
            axis = np.fromstring(axis_element.attrib.get("xyz", ""), sep=" ")
            if axis.shape != (3,) or np.linalg.norm(axis) < 1e-12:
                raise ValueError(f"{name}的axis无效")
            lower = float(limit_element.attrib["lower"])
            upper = float(limit_element.attrib["upper"])
            return JointTemplate(name, normalize(axis, name=f"{name} axis"), lower, upper)

        parsed = {name: parse_joint(name) for name in expected}
        origins = {name: parse_origin(name) for name in expected}

        # 中央L4的frame位于当前本体左侧安装frame；J5的origin给出右侧
        # 安装frame相对于L4的真实间距。该间距属于中央本体，不属于杆长变量。
        left_names = ("J4", "J3", "J2", "J1")
        right_names = ("J5", "J6", "J7", "J8")

        left_joint_templates = tuple(
            JointTemplate(
                name=name,
                # 由L4向base端反向展开，原关节变量的旋转方向取逆。
                axis=-parsed[name].axis,
                lower=parsed[name].lower,
                upper=parsed[name].upper,
            )
            for name in left_names
        )
        right_joint_templates = tuple(parsed[name] for name in right_names)

        left_vectors = np.asarray(
            [origins[name].inverse().position for name in left_names], dtype=float
        )
        right_vectors = np.asarray(
            [origins[name].position for name in ("J6", "J7", "J8")]
            + [frames.l8_end.position],
            dtype=float,
        )
        left_lengths = np.linalg.norm(left_vectors, axis=1)
        right_lengths = np.linalg.norm(right_vectors, axis=1)
        if not np.allclose(left_lengths, right_lengths, atol=2e-5):
            raise ValueError(
                "当前左右两侧零位有效杆长不镜像："
                f"left={left_lengths.tolist()}, right={right_lengths.tolist()}"
            )

        central_link = links["L4"]
        central_mesh = central_link.find("collision/geometry/mesh")
        central_visual = central_link.find("visual/geometry/mesh")
        if central_mesh is None and central_visual is None:
            raise ValueError("L4缺少真实mesh")
        mesh_element = central_mesh if central_mesh is not None else central_visual
        mesh_name = str(mesh_element.attrib["filename"]).split("/meshes/")[-1]
        mass_element = central_link.find("inertial/mass")
        central_mass = float(mass_element.attrib["value"]) if mass_element is not None else 0.0

        def mesh_path(name: str) -> Path:
            return paths.robot_mesh_dir / name

        return cls(
            central_link_name="L4",
            central_mesh_path=mesh_path(mesh_name),
            central_mass_kg=central_mass,
            left=BranchTemplate(
                side="left",
                endpoint_name="base_end",
                joint_templates=left_joint_templates,
                mount_pose=np.eye(4, dtype=float),
                span_directions=np.asarray(
                    [normalize(value, name="left span") for value in left_vectors]
                ),
                nominal_lengths_m=left_lengths,
                endpoint_link_to_suction=_pose_from_transform(frames.base_end.transform_link_to_suction),
                endpoint_mesh_path=mesh_path("base_link.STL"),
            ),
            right=BranchTemplate(
                side="right",
                endpoint_name="l8_end",
                joint_templates=right_joint_templates,
                mount_pose=_pose_from_transform(origins["J5"]),
                span_directions=np.asarray(
                    [normalize(value, name="right span") for value in right_vectors]
                ),
                nominal_lengths_m=right_lengths,
                endpoint_link_to_suction=_pose_from_transform(frames.l8_end.transform_link_to_suction),
                endpoint_mesh_path=mesh_path("L8.STL"),
            ),
            original_joint_names=tuple(expected),
            original_joint_limits={
                name: (parsed[name].lower, parsed[name].upper) for name in expected
            },
        )

    def baseline_joint_vector(self, urdf_ordered_q: np.ndarray) -> np.ndarray:
        """把URDF的J1..J8顺序转换为左外向、右外向顺序。"""

        values = np.asarray(urdf_ordered_q, dtype=float)
        if values.shape != (8,):
            raise ValueError("BASELINE_8R关节向量必须是(8,)")
        return np.concatenate((values[[3, 2, 1, 0]], values[[4, 5, 6, 7]]))


@dataclass(frozen=True)
class MorphologySpec:
    """一个左右对称6R或8R机构设计。"""

    name: str
    dof: int
    per_side_dof: int
    topology_id: str
    left_joint_templates: tuple[JointTemplate, ...]
    right_joint_templates: tuple[JointTemplate, ...]
    left_axes: np.ndarray
    right_axes: np.ndarray
    left_span_directions: np.ndarray
    right_span_directions: np.ndarray
    link_lengths_m: np.ndarray
    nominal_link_lengths_m: np.ndarray
    left_mount_pose: np.ndarray
    right_mount_pose: np.ndarray
    left_endpoint_link_to_suction: np.ndarray
    right_endpoint_link_to_suction: np.ndarray
    central_mesh_path: Path
    left_endpoint_mesh_path: Path
    right_endpoint_mesh_path: Path
    left_full_axes: np.ndarray
    right_full_axes: np.ndarray
    left_full_span_directions: np.ndarray
    right_full_span_directions: np.ndarray
    left_full_nominal_lengths_m: np.ndarray
    right_full_nominal_lengths_m: np.ndarray
    left_active_indices: tuple[int, ...]
    right_active_indices: tuple[int, ...]
    left_terminal_span_is_suction_offset: bool
    right_terminal_span_is_suction_offset: bool
    joint_proxy_radius_m: float = 0.065
    link_proxy_radius_m: float = 0.065
    collision_inflation_m: float = 0.005
    # Per-original-span profiles.  The scalar fields above remain for legacy
    # reports and old callers; collision code uses these arrays when present.
    link_proxy_radii_m: np.ndarray | None = None
    joint_proxy_radii_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.per_side_dof not in {3, 4} or self.dof != 2 * self.per_side_dof:
            raise ValueError("本轮只允许对称6R或对称8R")
        if len(self.link_lengths_m) != self.per_side_dof:
            raise ValueError("link_lengths_m长度必须等于每侧DOF")
        if np.any(np.asarray(self.link_lengths_m) <= 0.0):
            raise ValueError("所有设计杆长必须大于0")
        if self.collision_inflation_m < 0.0:
            raise ValueError("collision_inflation_m不能为负")
        if len(self.left_active_indices) != self.per_side_dof:
            raise ValueError("left_active_indices数量必须等于每侧DOF")
        if len(self.right_active_indices) != self.per_side_dof:
            raise ValueError("right_active_indices数量必须等于每侧DOF")
        for name, values in (
            ("link_proxy_radii_m", self.link_proxy_radii_m),
            ("joint_proxy_radii_m", self.joint_proxy_radii_m),
        ):
            if values is not None:
                array = np.asarray(values, dtype=float)
                if array.shape != (4,) or np.any(array <= 0.0):
                    raise ValueError(f"{name}必须是4个正数的原始4R profile")

    @property
    def lower_limits(self) -> np.ndarray:
        return np.asarray(
            [joint.lower for joint in self.left_joint_templates]
            + [joint.lower for joint in self.right_joint_templates],
            dtype=float,
        )

    @property
    def upper_limits(self) -> np.ndarray:
        return np.asarray(
            [joint.upper for joint in self.left_joint_templates]
            + [joint.upper for joint in self.right_joint_templates],
            dtype=float,
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(
            [f"left_{joint.name}" for joint in self.left_joint_templates]
            + [f"right_{joint.name}" for joint in self.right_joint_templates]
        )

    def link_proxy_radius_for_original_index(self, index: int) -> float:
        if index not in range(4):
            raise ValueError("原始侧链span index必须是0到3")
        if self.link_proxy_radii_m is not None:
            return float(np.asarray(self.link_proxy_radii_m, dtype=float)[index])
        return float(self.link_proxy_radius_m)

    def joint_proxy_radius_for_original_index(self, index: int) -> float:
        if index not in range(4):
            raise ValueError("原始侧链joint index必须是0到3")
        if self.joint_proxy_radii_m is not None:
            return float(np.asarray(self.joint_proxy_radii_m, dtype=float)[index])
        return float(self.joint_proxy_radius_m)

    @classmethod
    def baseline_8r(cls, geometry: BaselineGeometry, *, collision_inflation_m: float = 0.005) -> "MorphologySpec":
        return cls.from_geometry(
            geometry,
            topology_id="baseline_8r",
            remove_pair_index=None,
            link_lengths_m=geometry.left.nominal_lengths_m,
            collision_inflation_m=collision_inflation_m,
        )

    @classmethod
    def six_r_topology(
        cls,
        geometry: BaselineGeometry,
        remove_pair_index: int,
        *,
        link_lengths_m: np.ndarray | None = None,
        collision_inflation_m: float = 0.005,
    ) -> "MorphologySpec":
        if remove_pair_index not in range(4):
            raise ValueError("remove_pair_index必须是0到3")
        active_indices = tuple(index for index in range(4) if index != remove_pair_index)
        nominal = geometry.left.nominal_lengths_m[np.asarray(active_indices, dtype=int)]
        lengths = nominal if link_lengths_m is None else np.asarray(link_lengths_m, dtype=float)
        return cls.from_geometry(
            geometry,
            topology_id=f"6r_drop_pair_{remove_pair_index}",
            remove_pair_index=remove_pair_index,
            link_lengths_m=lengths,
            collision_inflation_m=collision_inflation_m,
        )

    @classmethod
    def optimized_8r(
        cls,
        geometry: BaselineGeometry,
        link_lengths_m: np.ndarray,
        *,
        collision_inflation_m: float = 0.005,
    ) -> "MorphologySpec":
        return cls.from_geometry(
            geometry,
            topology_id="optimized_8r",
            remove_pair_index=None,
            link_lengths_m=np.asarray(link_lengths_m, dtype=float),
            collision_inflation_m=collision_inflation_m,
        )

    @classmethod
    def from_geometry(
        cls,
        geometry: BaselineGeometry,
        *,
        topology_id: str,
        remove_pair_index: int | None,
        link_lengths_m: np.ndarray,
        collision_inflation_m: float,
    ) -> "MorphologySpec":
        left = geometry.left
        right = geometry.right
        if remove_pair_index is None:
            left_joints = left.joint_templates
            right_joints = right.joint_templates
            left_dirs = left.span_directions
            right_dirs = right.span_directions
            nominal = left.nominal_lengths_m
            active_indices = tuple(range(4))
        else:
            active_indices = tuple(index for index in range(4) if index != remove_pair_index)
            left_joints = tuple(left.joint_templates[index] for index in active_indices)
            right_joints = tuple(right.joint_templates[index] for index in active_indices)
            left_dirs = left.span_directions[np.asarray(active_indices, dtype=int)]
            right_dirs = right.span_directions[np.asarray(active_indices, dtype=int)]
            nominal = left.nominal_lengths_m[np.asarray(active_indices, dtype=int)]

        lengths = np.asarray(link_lengths_m, dtype=float)
        if lengths.shape != (len(left_joints),):
            raise ValueError(
                f"{topology_id}的杆长数量应为{len(left_joints)}，实际为{lengths.shape}"
            )
        calibration = calibrate_proxy_dimensions(geometry.central_mesh_path.parent)
        return cls(
            name=topology_id.upper(),
            dof=2 * len(left_joints),
            per_side_dof=len(left_joints),
            topology_id=topology_id,
            left_joint_templates=left_joints,
            right_joint_templates=right_joints,
            left_axes=np.asarray([joint.axis for joint in left_joints], dtype=float),
            right_axes=np.asarray([joint.axis for joint in right_joints], dtype=float),
            left_span_directions=np.asarray(left_dirs, dtype=float),
            right_span_directions=np.asarray(right_dirs, dtype=float),
            link_lengths_m=lengths.copy(),
            nominal_link_lengths_m=np.asarray(nominal, dtype=float),
            left_mount_pose=left.mount_pose.copy(),
            right_mount_pose=right.mount_pose.copy(),
            left_endpoint_link_to_suction=left.endpoint_link_to_suction.copy(),
            right_endpoint_link_to_suction=right.endpoint_link_to_suction.copy(),
            central_mesh_path=geometry.central_mesh_path,
            left_endpoint_mesh_path=left.endpoint_mesh_path,
            right_endpoint_mesh_path=right.endpoint_mesh_path,
            left_full_axes=np.asarray([joint.axis for joint in left.joint_templates], dtype=float),
            right_full_axes=np.asarray([joint.axis for joint in right.joint_templates], dtype=float),
            left_full_span_directions=np.asarray(left.span_directions, dtype=float),
            right_full_span_directions=np.asarray(right.span_directions, dtype=float),
            left_full_nominal_lengths_m=np.asarray(left.nominal_lengths_m, dtype=float),
            right_full_nominal_lengths_m=np.asarray(right.nominal_lengths_m, dtype=float),
            left_active_indices=active_indices,
            right_active_indices=active_indices,
            left_terminal_span_is_suction_offset=False,
            right_terminal_span_is_suction_offset=True,
            joint_proxy_radius_m=float(np.mean(calibration.full_joint_radii_m)),
            link_proxy_radius_m=float(np.mean(calibration.full_link_radii_m)),
            collision_inflation_m=collision_inflation_m,
            link_proxy_radii_m=calibration.full_link_radii_m.copy(),
            joint_proxy_radii_m=calibration.full_joint_radii_m.copy(),
        )


def _merged_span_vectors(vectors: np.ndarray, remove_pair_index: int) -> np.ndarray:
    """删除一个侧链关节后合并其相邻刚性段，生成真正的3R侧链。"""

    values = np.asarray(vectors, dtype=float)
    if values.shape != (4, 3):
        raise ValueError("当前4R基线的span向量必须是(4,3)")
    merge_index = min(remove_pair_index, 2)
    output: list[np.ndarray] = []
    index = 0
    while index < 4:
        if index == merge_index:
            output.append(values[index] + values[index + 1])
            index += 2
        else:
            output.append(values[index])
            index += 1
    result = np.asarray(output, dtype=float)
    if result.shape != (3, 3):
        raise AssertionError("合并4段杆后必须得到3段杆")
    return result


@dataclass(frozen=True)
class MorphologyState:
    """一个关节状态下的世界几何和两个末端位姿。"""

    q: np.ndarray
    body_pose: np.ndarray
    left_suction_pose: np.ndarray
    right_suction_pose: np.ndarray
    left_endpoint_link_pose: np.ndarray
    right_endpoint_link_pose: np.ndarray
    left_joint_poses: tuple[np.ndarray, ...]
    right_joint_poses: tuple[np.ndarray, ...]
    left_span_segments: tuple[tuple[np.ndarray, np.ndarray], ...]
    right_span_segments: tuple[tuple[np.ndarray, np.ndarray], ...]

    def suction_pose(self, endpoint_name: str) -> np.ndarray:
        if endpoint_name == "base_end":
            return self.left_suction_pose
        if endpoint_name == "l8_end":
            return self.right_suction_pose
        raise ValueError(f"未知物理端：{endpoint_name}")

    def endpoint_link_pose(self, endpoint_name: str) -> np.ndarray:
        if endpoint_name == "base_end":
            return self.left_endpoint_link_pose
        if endpoint_name == "l8_end":
            return self.right_endpoint_link_pose
        raise ValueError(f"未知物理端：{endpoint_name}")


class MorphologyModel:
    """给定MorphologySpec的快速双侧正运动学模型。"""

    def __init__(self, spec: MorphologySpec) -> None:
        self.spec = spec

    def forward(
        self,
        q: np.ndarray,
        *,
        body_pose: np.ndarray | None = None,
    ) -> MorphologyState:
        values = np.asarray(q, dtype=float)
        if values.shape != (self.spec.dof,):
            raise ValueError(f"q必须是({self.spec.dof},)，实际为{values.shape}")
        body = np.eye(4, dtype=float) if body_pose is None else np.asarray(body_pose, dtype=float)
        if body.shape != (4, 4):
            raise ValueError("body_pose必须是4x4矩阵")
        left = self._forward_branch(
            body,
            self.spec.left_mount_pose,
            self.spec.left_full_axes,
            self.spec.left_full_span_directions,
            self.spec.left_full_nominal_lengths_m,
            self.spec.left_active_indices,
            values[: self.spec.per_side_dof],
            self.spec.link_lengths_m,
            self.spec.left_endpoint_link_to_suction,
            self.spec.left_terminal_span_is_suction_offset,
        )
        right = self._forward_branch(
            body,
            self.spec.right_mount_pose,
            self.spec.right_full_axes,
            self.spec.right_full_span_directions,
            self.spec.right_full_nominal_lengths_m,
            self.spec.right_active_indices,
            values[self.spec.per_side_dof :],
            self.spec.link_lengths_m,
            self.spec.right_endpoint_link_to_suction,
            self.spec.right_terminal_span_is_suction_offset,
        )
        return MorphologyState(
            q=values.copy(),
            body_pose=body.copy(),
            left_suction_pose=left[0],
            right_suction_pose=right[0],
            left_endpoint_link_pose=left[1],
            right_endpoint_link_pose=right[1],
            left_joint_poses=tuple(left[2]),
            right_joint_poses=tuple(right[2]),
            left_span_segments=tuple(left[3]),
            right_span_segments=tuple(right[3]),
        )

    @staticmethod
    def _forward_branch(
        body_pose: np.ndarray,
        mount_pose: np.ndarray,
        axes: np.ndarray,
        span_directions: np.ndarray,
        nominal_lengths: np.ndarray,
        active_indices: tuple[int, ...],
        q: np.ndarray,
        lengths: np.ndarray,
        endpoint_link_to_suction: np.ndarray,
        terminal_span_is_suction_offset: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
        current = body_pose @ mount_pose
        joint_poses: list[np.ndarray] = []
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        active_position = {original: position for position, original in enumerate(active_indices)}
        endpoint_link_pose: np.ndarray | None = None
        for original_index, (axis, direction) in enumerate(zip(axes, span_directions)):
            rotated = current.copy()
            if original_index in active_position:
                joint_poses.append(current.copy())
                angle = float(q[active_position[original_index]])
                rotated[:3, :3] = current[:3, :3] @ _rotation_about_axis(axis, angle)
            start = rotated[:3, 3].copy()
            if original_index in active_position:
                length = float(lengths[active_position[original_index]])
            else:
                length = float(nominal_lengths[original_index])
            end = start + rotated[:3, :3] @ (normalize(direction, name="span direction") * length)
            next_pose = rotated.copy()
            next_pose[:3, 3] = end
            segments.append((start, end))
            if terminal_span_is_suction_offset and original_index == len(span_directions) - 1:
                endpoint_link_pose = rotated.copy()
            current = next_pose

        if endpoint_link_pose is None:
            endpoint_link_pose = current.copy()
        rotation_only = np.eye(4, dtype=float)
        rotation_only[:3, :3] = endpoint_link_to_suction[:3, :3]
        suction_pose = current @ rotation_only
        # 当前最后一段已经把真实吸盘中心的平移包含进来了；因此这里
        # 只追加吸盘frame的旋转，避免重复计算末端position offset。
        return suction_pose, endpoint_link_pose, joint_poses, segments

    def body_pose_for_support(
        self,
        q: np.ndarray,
        support_endpoint: str,
        support_target_pose: np.ndarray,
    ) -> np.ndarray:
        """给定关节角和支撑吸盘目标，反算中央本体世界位姿。"""

        relative = self.forward(q).suction_pose(support_endpoint)
        return np.asarray(support_target_pose, dtype=float) @ _inverse_pose(relative)

    def world_state_for_support(
        self,
        q: np.ndarray,
        support_endpoint: str,
        support_target_pose: np.ndarray,
    ) -> MorphologyState:
        body_pose = self.body_pose_for_support(q, support_endpoint, support_target_pose)
        return self.forward(q, body_pose=body_pose)

    def moving_suction_pose_for_support(
        self,
        q: np.ndarray,
        support_endpoint: str,
        moving_endpoint: str,
        support_target_pose: np.ndarray,
    ) -> np.ndarray:
        state = self.world_state_for_support(q, support_endpoint, support_target_pose)
        return state.suction_pose(moving_endpoint)

    def copy_with_lengths(self, lengths_m: np.ndarray) -> "MorphologyModel":
        return MorphologyModel(replace(self.spec, link_lengths_m=np.asarray(lengths_m, dtype=float)))


def with_axis_architecture(
    spec: MorphologySpec,
    left_full_axes: np.ndarray,
    *,
    architecture_id: str,
) -> MorphologySpec:
    """应用一个有限离散的左侧轴序列，并按真实镜像规则生成右侧轴。

    当前 outward branch 的镜像关系是 ``right = -M @ left``，其中M为
    关于中央本体左右平面的反射。该关系与从URDF反向展开左链得到的
    BASELINE_8R轴序列一致。
    """

    left = np.asarray(left_full_axes, dtype=float)
    if left.shape != (4, 3):
        raise ValueError("离散轴架构必须包含4个左侧轴")
    left = left / np.linalg.norm(left, axis=1, keepdims=True)
    mirror = np.diag([-1.0, 1.0, 1.0])
    right = np.asarray([-mirror @ axis for axis in left], dtype=float)
    left_active = left[np.asarray(spec.left_active_indices, dtype=int)]
    right_active = right[np.asarray(spec.right_active_indices, dtype=int)]
    return replace(
        spec,
        name=f"{spec.name}_{architecture_id.upper()}",
        topology_id=f"{spec.topology_id}_{architecture_id}",
        left_axes=left_active,
        right_axes=right_active,
        left_full_axes=left,
        right_full_axes=right,
    )
