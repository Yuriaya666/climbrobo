from __future__ import annotations

import time
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p

from environment.paths import ProjectPaths
from environment.suction_frames import SuctionFrame
from environment.transforms import RigidTransform, angle_between_vectors_rad
from environment.urdf_resolver import ResolvedUrdf


@dataclass(frozen=True)
class JointInfo:
    """PyBullet中一个可动关节的基本信息。"""

    index: int
    name: str
    lower: float
    upper: float


@dataclass
class SupportAnchor:
    """记录一个需要在世界坐标中保持不动的吸盘支撑端。"""

    suction_frame: SuctionFrame
    target_pose: RigidTransform


class PyBulletScene:
    """负责加载铁塔、机器人，并封装常用PyBullet查询。"""

    def __init__(self, paths: ProjectPaths, *, gui: bool = False) -> None:
        self.paths = paths
        self.gui = gui
        self.client_id: int | None = None
        self.robot_id: int | None = None
        self.tower_id: int | None = None
        self.link_index_by_name: dict[str, int] = {}
        self.link_name_by_index: dict[int, str] = {}
        self.joints: list[JointInfo] = []
        self.parent_link_by_child: dict[int, int] = {}
        self.joint_origin_by_child: dict[int, np.ndarray] = {}
        self._visual_marker_ids: list[int] = []
        self._support_anchor: SupportAnchor | None = None

    def __enter__(self) -> "PyBulletScene":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.disconnect()

    def connect(self) -> None:
        mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(mode)
        if self.client_id < 0:
            raise RuntimeError("无法连接PyBullet")

        p.resetSimulation()
        p.setGravity(0.0, 0.0, 0.0)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    def disconnect(self) -> None:
        if p.isConnected():
            p.disconnect()
        self.client_id = None
        self._support_anchor = None

    def load_tower(self) -> int:
        """加载铁塔，碰撞用高精度网格，显示用简化网格。"""

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=str(self.paths.tower_collision_mesh),
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(self.paths.tower_visual_mesh),
            rgbaColor=[0.65, 0.65, 0.65, 0.55],
        )
        self.tower_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        return self.tower_id

    def load_robot(self, base_pose: RigidTransform) -> int:
        """加载机器人，使用临时URDF解决package://路径问题。"""

        flags = (
            p.URDF_USE_INERTIA_FROM_FILE
            | p.URDF_MAINTAIN_LINK_ORDER
            | p.URDF_USE_SELF_COLLISION
        )
        flags |= getattr(p, "URDF_USE_SELF_COLLISION_EXCLUDE_PARENT", 0)

        base_position, base_orientation = base_pose.as_pybullet()
        with ResolvedUrdf(self.paths.robot_urdf, self.paths.robot_mesh_dir) as urdf_path:
            self.robot_id = p.loadURDF(
                fileName=str(urdf_path),
                basePosition=base_position,
                baseOrientation=base_orientation,
                useFixedBase=True,
                flags=flags,
            )
        self._support_anchor = None

        self._refresh_robot_metadata()
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            # GUI渲染线程刚启动时，PyBullet的user debug draw偶尔会失败。
            # 这里推进一帧并短暂等待，让OpenGL上下文完成初始化。
            p.stepSimulation()
            time.sleep(0.05)
        return self.robot_id

    def _refresh_robot_metadata(self) -> None:
        if self.robot_id is None:
            raise RuntimeError("机器人尚未加载")

        self.link_index_by_name = {"base_link": -1}
        self.link_name_by_index = {-1: "base_link"}
        self.joints = []
        self.parent_link_by_child = {}
        self.joint_origin_by_child = {}
        urdf_joint_origins = self._read_urdf_joint_origins_by_child_name()

        for joint_index in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, joint_index)
            joint_name = info[1].decode("utf-8", errors="ignore")
            joint_type = info[2]
            lower = float(info[8])
            upper = float(info[9])
            child_link_name = info[12].decode("utf-8", errors="ignore")
            joint_origin_parent = urdf_joint_origins.get(
                child_link_name,
                np.asarray(info[14], dtype=float),
            )
            parent_link_index = int(info[16])

            self.link_index_by_name[child_link_name] = joint_index
            self.link_name_by_index[joint_index] = child_link_name
            self.parent_link_by_child[joint_index] = parent_link_index
            self.joint_origin_by_child[joint_index] = joint_origin_parent

            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joints.append(
                    JointInfo(
                        index=joint_index,
                        name=joint_name,
                        lower=lower,
                        upper=upper,
                    )
                )

    def _read_urdf_joint_origins_by_child_name(self) -> dict[str, np.ndarray]:
        """
        从原始URDF读取joint origin。

        PyBullet jointInfo里的parentFramePos会受惯性帧处理影响，不适合估计
        串联链最大长度，所以这里直接读取URDF文本中的关节平移。
        """

        origins: dict[str, np.ndarray] = {}
        tree = ET.parse(self.paths.robot_urdf)
        root = tree.getroot()
        for joint in root.findall("joint"):
            child = joint.find("child")
            origin = joint.find("origin")
            if child is None or origin is None:
                continue
            child_name = child.attrib.get("link")
            xyz_text = origin.attrib.get("xyz", "0 0 0")
            if not child_name:
                continue
            origins[child_name] = np.fromstring(xyz_text, sep=" ", dtype=float)
        return origins

    def link_index(self, link_name: str) -> int:
        if link_name not in self.link_index_by_name:
            raise KeyError(f"URDF中找不到link：{link_name}")
        return self.link_index_by_name[link_name]

    def link_name(self, link_index: int) -> str:
        return self.link_name_by_index.get(link_index, f"link_{link_index}")

    def adjacent_link_pairs(self) -> set[tuple[int, int]]:
        """用于过滤父子相邻link的自碰撞误报。"""

        pairs = set()
        for child, parent in self.parent_link_by_child.items():
            pairs.add(tuple(sorted((child, parent))))
        return pairs

    def estimate_serial_chain_upper_bound(
        self,
        *,
        end_link_index: int,
        base_suction_frame: SuctionFrame,
        end_suction_frame: SuctionFrame,
        base_link_index: int = -1,
        base_suction_link_index: int | None = None,
    ) -> float:
        """
        根据URDF关节origin估算两个吸盘中心的最大可能距离上界。

        这是一个保守上界，用来跳过明显超过机器人长度的候选点。
        """

        distance = float(np.linalg.norm(base_suction_frame.position))
        distance += float(np.linalg.norm(end_suction_frame.position))

        # 当前模型的两只吸盘位于串联链两端。运动端是base_link时，
        # 不能因为它本身没有父关节就漏掉整条J1~J8链；此时从支撑端
        # 的link反向累计到URDF base。调用方应显式传入支撑吸盘所在link。
        if end_link_index == base_link_index:
            if base_suction_link_index is None:
                # 兼容旧调用：该机器人只有一条完整串联链，累加全部URDF关节origin。
                distance += sum(
                    float(np.linalg.norm(origin))
                    for origin in self.joint_origin_by_child.values()
                )
                return distance
            link_index = int(base_suction_link_index)
            stop_index = base_link_index
        else:
            link_index = end_link_index
            stop_index = base_link_index
        while link_index != stop_index and link_index != -1:
            if link_index not in self.parent_link_by_child:
                break
            distance += float(np.linalg.norm(self.joint_origin_by_child[link_index]))
            link_index = self.parent_link_by_child[link_index]
        return distance

    def joint_positions(self) -> np.ndarray:
        if self.robot_id is None:
            raise RuntimeError("机器人尚未加载")
        return np.array(
            [p.getJointState(self.robot_id, joint.index)[0] for joint in self.joints],
            dtype=float,
        )

    def get_base_pose(self) -> RigidTransform:
        """读取机器人URDF base link在世界坐标中的位姿。"""

        return self.get_link_pose(-1)

    def set_base_pose(self, pose: RigidTransform) -> None:
        """设置固定基座的世界位姿，不改变任何关节角。"""

        if self.robot_id is None:
            raise RuntimeError("机器人尚未加载")
        position, orientation = pose.as_pybullet()
        p.resetBasePositionAndOrientation(self.robot_id, position, orientation)

    def enable_support_anchor(
        self,
        suction_frame: SuctionFrame,
        target_pose: RigidTransform,
    ) -> None:
        """启用支撑锚定，使后续关节更新保持该吸盘位姿不变。"""

        if self.robot_id is None:
            raise RuntimeError("启用支撑锚定前必须先加载机器人")
        self._support_anchor = SupportAnchor(suction_frame, target_pose)

    def disable_support_anchor(self) -> None:
        """关闭支撑锚定，恢复普通固定base的关节更新行为。"""

        self._support_anchor = None

    @property
    def support_anchor(self) -> SupportAnchor | None:
        """返回当前支撑锚定配置，供回放和诊断使用。"""

        return self._support_anchor

    def support_anchor_errors(self) -> tuple[float, float] | None:
        """返回当前支撑吸盘的位置误差和法向误差。"""

        if self._support_anchor is None:
            return None
        actual = self.get_suction_pose(self._support_anchor.suction_frame)
        position_error = float(np.linalg.norm(actual.position - self._support_anchor.target_pose.position))
        normal_error = angle_between_vectors_rad(actual.z_axis, self._support_anchor.target_pose.z_axis)
        return position_error, normal_error

    def capture_link_poses(self) -> dict[int, RigidTransform]:
        """保存base和全部link的世界位姿，用于变基座连续性验证。"""

        if self.robot_id is None:
            raise RuntimeError("机器人尚未加载")
        return {
            link_index: self.get_link_pose(link_index)
            for link_index in range(-1, p.getNumJoints(self.robot_id))
        }

    def capture_base_poses_for_trajectory(
        self,
        trajectory: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """逐状态记录轨迹对应的base位姿，支持锚定轨迹的脱离场景回放。"""

        positions: list[np.ndarray] = []
        orientations: list[np.ndarray] = []
        for joints in np.asarray(trajectory, dtype=float):
            self.reset_joints(joints)
            pose = self.get_base_pose()
            positions.append(pose.position.copy())
            orientations.append(pose.quaternion_xyzw.copy())
        return np.asarray(positions), np.asarray(orientations)

    def reset_joints(self, positions: np.ndarray) -> None:
        if self.robot_id is None:
            raise RuntimeError("机器人尚未加载")
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (len(self.joints),):
            raise ValueError(f"关节角shape应为({len(self.joints)},)，实际为{positions.shape}")

        for joint, value in zip(self.joints, positions):
            p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=joint.index,
                targetValue=float(value),
                targetVelocity=0.0,
            )

        # 先设置关节，再根据当前关节下的base->吸盘相对变换反算新的base。
        # 这样第二步的新支撑脚在整个IK、直线和RRT检查中都保持固定。
        if self._support_anchor is not None:
            # base_end直接位于base_link原点时，base到支撑吸盘的变换与
            # 关节状态无关。数值IK会高频调用reset_joints，这里跳过重复
            # 的正运动学查询，仍使用完全相同的锚定公式。
            if (
                self._support_anchor.suction_frame.link_name == "base_link"
                and np.linalg.norm(
                    self._support_anchor.suction_frame.transform_link_to_suction.position
                )
                < 1e-12
            ):
                desired_base = self._support_anchor.target_pose.multiply(
                    self._support_anchor.suction_frame.transform_link_to_suction.inverse()
                )
                self.set_base_pose(desired_base)
                return

            base_pose = self.get_base_pose()
            support_pose = self.get_suction_pose(self._support_anchor.suction_frame)
            base_to_support = base_pose.inverse().multiply(support_pose)
            desired_base = self._support_anchor.target_pose.multiply(base_to_support.inverse())
            self.set_base_pose(desired_base)

    def joint_lower_limits(self) -> np.ndarray:
        return np.array([joint.lower for joint in self.joints], dtype=float)

    def joint_upper_limits(self) -> np.ndarray:
        return np.array([joint.upper for joint in self.joints], dtype=float)

    def normalize_revolute_solutions(self, positions: np.ndarray) -> np.ndarray:
        """
        把IK给出的等价大角度折回URDF限位。

        只处理接近整圈范围的关节，例如[-3.14, 3.14]；像J2/J7这种
        [-1, 1]窄限位关节不做周期折返，避免掩盖真实越界。
        """

        values = np.asarray(positions, dtype=float).copy()
        for i, joint in enumerate(self.joints):
            lower = joint.lower
            upper = joint.upper
            joint_range = upper - lower
            if joint_range < 2.0 * np.pi - 0.02:
                continue

            center = 0.5 * (lower + upper)
            values[i] = ((values[i] - center + np.pi) % (2.0 * np.pi)) - np.pi + center
        return values

    def within_joint_limits(self, positions: np.ndarray, margin: float = 1e-7) -> bool:
        positions = np.asarray(positions, dtype=float)
        return bool(
            np.all(positions >= self.joint_lower_limits() - margin)
            and np.all(positions <= self.joint_upper_limits() + margin)
        )

    def get_link_pose(self, link_index: int) -> RigidTransform:
        if self.robot_id is None:
            raise RuntimeError("机器人尚未加载")
        if link_index == -1:
            position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        else:
            state = p.getLinkState(
                self.robot_id,
                link_index,
                computeForwardKinematics=True,
            )
            position, orientation = state[4], state[5]
        return RigidTransform(
            position=np.asarray(position, dtype=float),
            quaternion_xyzw=np.asarray(orientation, dtype=float),
        )

    def get_link_inertial_pose(self, link_index: int) -> RigidTransform:
        """
        获取 T_link_inertial。

        PyBullet的IK目标使用link惯性/质心帧。规划代码更自然地使用URDF
        link frame，所以在calculate_ik里会用这个固定变换做一次转换。
        """

        if self.robot_id is None:
            raise RuntimeError("机器人尚未加载")
        if link_index < 0:
            dynamics = p.getDynamicsInfo(self.robot_id, -1)
            position, orientation = dynamics[3], dynamics[4]
        else:
            state = p.getLinkState(
                self.robot_id,
                link_index,
                computeForwardKinematics=True,
            )
            position, orientation = state[2], state[3]
        return RigidTransform(
            position=np.asarray(position, dtype=float),
            quaternion_xyzw=np.asarray(orientation, dtype=float),
        )

    def get_suction_pose(self, suction_frame: SuctionFrame) -> RigidTransform:
        link_pose = self.get_link_pose(self.link_index(suction_frame.link_name))
        return link_pose.multiply(suction_frame.transform_link_to_suction)

    def calculate_suction_jacobian(
        self,
        suction_frame: SuctionFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算吸盘中心在世界坐标中的线速度和角速度雅可比。

        PyBullet的``calculateJacobian``返回的是link原点的雅可比，且
        输出轴向随base坐标系表达。这里先转换到世界坐标，再补上
        吸盘中心相对于link原点的偏移，结果与项目的吸盘FK定义一致。
        """

        if self.robot_id is None:
            raise RuntimeError("计算雅可比前必须先加载机器人")

        link_index = self.link_index(suction_frame.link_name)
        joint_positions = self.joint_positions().tolist()
        zeros = [0.0] * len(joint_positions)
        linear_base, angular_base = p.calculateJacobian(
            self.robot_id,
            link_index,
            [0.0, 0.0, 0.0],
            joint_positions,
            zeros,
            zeros,
        )
        base_rotation = self.get_base_pose().rotation_matrix
        linear_link = base_rotation @ np.asarray(linear_base, dtype=float)
        angular_world = base_rotation @ np.asarray(angular_base, dtype=float)
        link_pose = self.get_link_pose(link_index)
        suction_pose = self.get_suction_pose(suction_frame)
        offset_world = suction_pose.position - link_pose.position
        linear_suction = linear_link + np.cross(
            angular_world.T,
            offset_world,
        ).T
        return linear_suction, angular_world

    def calculate_ik(
        self,
        link_index: int,
        target_link_pose: RigidTransform,
        rest_positions: np.ndarray,
        *,
        max_iterations: int,
        residual_threshold: float,
    ) -> np.ndarray:
        if self.robot_id is None:
            raise RuntimeError("机器人尚未加载")

        # 外部传入的是URDF link frame目标；PyBullet IK实际求的是惯性帧目标。
        target_inertial_pose = target_link_pose.multiply(
            self.get_link_inertial_pose(link_index)
        )

        lower = self.joint_lower_limits()
        upper = self.joint_upper_limits()
        ranges = upper - lower
        solution = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=link_index,
            targetPosition=target_inertial_pose.position.tolist(),
            targetOrientation=target_inertial_pose.quaternion_xyzw.tolist(),
            lowerLimits=lower.tolist(),
            upperLimits=upper.tolist(),
            jointRanges=ranges.tolist(),
            restPoses=np.asarray(rest_positions, dtype=float).tolist(),
            maxNumIterations=max_iterations,
            residualThreshold=residual_threshold,
        )
        return np.asarray(solution[: len(self.joints)], dtype=float)

    def draw_point(
        self,
        position: np.ndarray,
        color: list[float],
        size: float = 10.0,
    ) -> int:
        """用小球标记候选点，避免PyBullet debug draw warning。"""

        radius = 0.004 * max(float(size), 1.0)
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[color[0], color[1], color[2], 1.0],
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            baseCollisionShapeIndex=-1,
            basePosition=np.asarray(position, dtype=float).tolist(),
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        self._visual_marker_ids.append(body_id)
        return body_id

    def draw_frame(
        self,
        pose: RigidTransform,
        label: str,
        *,
        axis_length: float = 0.15,
        line_width: float = 3.0,
    ) -> list[int]:
        """用圆柱体画坐标轴：X红、Y绿、Z蓝。"""

        origin = pose.position
        rotation = pose.rotation_matrix
        colors = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        ids: list[int] = []
        for axis_index, color in enumerate(colors):
            end = origin + axis_length * rotation[:, axis_index]
            body_id = self._create_cylinder_between_points(
                start=origin,
                end=end,
                radius=0.0015 * max(float(line_width), 1.0),
                rgba=[color[0], color[1], color[2], 1.0],
            )
            ids.append(body_id)
            self._visual_marker_ids.append(body_id)
        return ids

    def draw_polyline(
        self,
        points: np.ndarray,
        color: list[float],
        *,
        radius: float = 0.0015,
    ) -> list[int]:
        """用细圆柱显示一条仅有可视几何的折线。"""

        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
            raise ValueError("points必须是至少包含两个点的(N,3)数组")
        ids: list[int] = []
        for start, end in zip(values[:-1], values[1:]):
            if np.linalg.norm(end - start) < 1e-10:
                continue
            body_id = self._create_cylinder_between_points(
                start=start,
                end=end,
                radius=radius,
                rgba=[color[0], color[1], color[2], 1.0],
            )
            ids.append(body_id)
            self._visual_marker_ids.append(body_id)
        return ids

    def _create_cylinder_between_points(
        self,
        *,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        rgba: list[float],
    ) -> int:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-12:
            raise ValueError("坐标轴长度不能为0")

        visual_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=rgba,
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=(0.5 * (start + end)).tolist(),
            baseOrientation=self._quaternion_from_z_to_direction(delta),
        )
        return body_id

    @staticmethod
    def _quaternion_from_z_to_direction(direction: np.ndarray) -> list[float]:
        """返回把圆柱局部Z轴转到direction方向的四元数。"""

        target = np.asarray(direction, dtype=float)
        target = target / np.linalg.norm(target)
        source = np.array([0.0, 0.0, 1.0], dtype=float)
        dot_value = float(np.clip(np.dot(source, target), -1.0, 1.0))

        if dot_value > 1.0 - 1e-12:
            return [0.0, 0.0, 0.0, 1.0]
        if dot_value < -1.0 + 1e-12:
            return [1.0, 0.0, 0.0, 0.0]

        axis = np.cross(source, target)
        axis = axis / np.linalg.norm(axis)
        angle = float(np.arccos(dot_value))
        sin_half = float(np.sin(0.5 * angle))
        cos_half = float(np.cos(0.5 * angle))
        return [
            float(axis[0] * sin_half),
            float(axis[1] * sin_half),
            float(axis[2] * sin_half),
            cos_half,
        ]

    def highlight_robot(self, rgba: list[float] | None = None) -> None:
        """把机器人改成醒目颜色，方便在大塔模型旁观察。"""

        if self.robot_id is None:
            raise RuntimeError("机器人尚未加载")
        if rgba is None:
            rgba = [1.0, 0.05, 0.02, 1.0]

        for link_index in range(-1, p.getNumJoints(self.robot_id)):
            p.changeVisualShape(
                objectUniqueId=self.robot_id,
                linkIndex=link_index,
                rgbaColor=rgba,
            )

    def focus_camera_on_robot(
        self,
        *,
        distance: float = 0.8,
        yaw: float = 45.0,
        pitch: float = -25.0,
    ) -> None:
        """把GUI相机对准机器人当前包围盒中心。"""

        if self.robot_id is None:
            raise RuntimeError("机器人尚未加载")

        aabb_min, aabb_max = p.getAABB(self.robot_id)
        center = 0.5 * (np.asarray(aabb_min, dtype=float) + np.asarray(aabb_max, dtype=float))
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=center.tolist(),
        )

    def play_joint_trajectory(
        self,
        trajectory: np.ndarray,
        *,
        seconds_per_state: float = 0.05,
        repeats: int = 3,
        pause_between_repeats_s: float = 0.8,
        step_simulation: bool = False,
    ) -> None:
        """
        按固定关节序列播放轨迹。

        默认不推进物理仿真，只做运动学显示，避免吸盘接触塔面时由接触
        约束引入抖动。
        """

        if repeats < 1:
            repeats = 1

        for repeat_index in range(repeats):
            for positions in trajectory:
                self.reset_joints(positions)
                if step_simulation:
                    p.stepSimulation()
                time.sleep(seconds_per_state)

            if repeat_index != repeats - 1:
                self.reset_joints(trajectory[0])
                if step_simulation:
                    p.stepSimulation()
                time.sleep(pause_between_repeats_s)
