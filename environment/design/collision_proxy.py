"""参数化6R/8R机构的保守碰撞代理。

中央本体和两个末端继续使用仓库中的真实mesh；尚未有CAD的杆件使用
Capsule，关节/电机使用有半径的球体。该模块只服务机构设计评价，不修改
现有URDF和正式规划器。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pybullet as p

from environment.design.morphology import MorphologyModel, MorphologyState
from environment.paths import ProjectPaths
from environment.transforms import RigidTransform, rotation_matrix_to_quaternion


@dataclass(frozen=True)
class ProxyCollisionResult:
    """一次参数化机构状态碰撞检查结果。"""

    ok: bool
    minimum_clearance_m: float
    critical_link: str | None
    kind: str | None
    critical_position_m: np.ndarray | None = None
    details: str = ""
    point_on_a_m: np.ndarray | None = None
    point_on_b_m: np.ndarray | None = None
    penetration_m: float | None = None


def _quaternion_from_z_axis(direction: np.ndarray) -> np.ndarray:
    """返回把局部+Z轴转到direction的四元数。"""

    z = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(z))
    if norm < 1e-12:
        raise ValueError("胶囊方向不能为零")
    z /= norm
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, z))) > 0.999999:
        reference = np.array([1.0, 0.0, 0.0])
    x = np.cross(reference, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return rotation_matrix_to_quaternion(np.column_stack((x, y, z)))


class MorphologyCollisionWorld:
    """持有一组可重复更新的PyBullet碰撞代理。"""

    def __init__(
        self,
        paths: ProjectPaths,
        model: MorphologyModel,
        *,
        gui: bool = False,
        load_tower: bool = True,
    ) -> None:
        self.paths = paths
        self.model = model
        self.spec = model.spec
        self.gui = gui
        self.load_tower_flag = load_tower
        self.client_id: int | None = None
        self.tower_id: int | None = None
        self.central_id: int | None = None
        self.endpoint_ids: dict[str, int] = {}
        self.segment_ids: dict[str, list[int]] = {"left": [], "right": []}
        self.joint_ids: dict[str, list[int]] = {"left": [], "right": []}
        self._body_names: dict[int, str] = {}
        self._allowed_adjacent_pairs: set[tuple[int, int]] = set()

    def __enter__(self) -> "MorphologyCollisionWorld":
        self.client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("无法连接PyBullet碰撞代理世界")
        p.resetSimulation()
        p.setGravity(0.0, 0.0, 0.0)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        self._build_world()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if p.isConnected():
            p.disconnect()
        self.client_id = None

    def _mesh_shape(self, path, *, collision: bool = True, visual: bool = False) -> tuple[int, int]:
        collision_id = -1
        visual_id = -1
        if collision:
            collision_id = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=str(path),
                flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            )
        if visual:
            visual_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=str(path),
                rgbaColor=[0.72, 0.72, 0.72, 1.0],
            )
        return collision_id, visual_id

    def _new_body(
        self,
        name: str,
        collision_id: int,
        visual_id: int = -1,
        *,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ) -> int:
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=(np.zeros(3) if position is None else position).tolist(),
            baseOrientation=(np.array([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation).tolist(),
        )
        self._body_names[body_id] = name
        return body_id

    def _build_world(self) -> None:
        if self.load_tower_flag:
            tower_collision, tower_visual = self._mesh_shape(
                self.paths.tower_collision_mesh,
                collision=True,
                visual=self.gui,
            )
            self.tower_id = self._new_body("Tower", tower_collision, tower_visual)

        central_collision, central_visual = self._mesh_shape(
            self.spec.central_mesh_path,
            collision=True,
            visual=self.gui,
        )
        self.central_id = self._new_body("central_body", central_collision, central_visual)

        for endpoint, mesh_path in (
            ("base_end", self.spec.left_endpoint_mesh_path),
            ("l8_end", self.spec.right_endpoint_mesh_path),
        ):
            collision, visual = self._mesh_shape(mesh_path, collision=True, visual=self.gui)
            self.endpoint_ids[endpoint] = self._new_body(f"{endpoint}_mesh", collision, visual)

        for side, count in (("left", 4), ("right", 4)):
            active_indices = self.spec.left_active_indices if side == "left" else self.spec.right_active_indices
            lengths = self.spec.link_lengths_m
            nominal = (
                self.spec.left_full_nominal_lengths_m
                if side == "left"
                else self.spec.right_full_nominal_lengths_m
            )
            active_position = {original: position for position, original in enumerate(active_indices)}
            # The right terminal span is the existing L8-to-suction-frame
            # offset.  L8.STL already represents that physical end structure
            # at the J8 link frame, so adding another capsule here double
            # counts the same geometry and creates false endpoint collisions.
            segment_count = count
            # Both endpoint meshes already represent the physical terminal
            # structure.  The left base-link mesh is placed at the terminal
            # frame after its kinematic offset, while the right L8 mesh is
            # placed at J8 and owns its suction offset.  In either case a
            # second terminal capsule would double-count endpoint geometry.
            terminal_is_suction_offset = (
                True
                if side == "left"
                else self.spec.right_terminal_span_is_suction_offset
            )
            if terminal_is_suction_offset:
                segment_count -= 1
            for index in range(segment_count):
                segment_length = float(
                    lengths[active_position[index]] if index in active_position else nominal[index]
                )
                radius = (
                    self.spec.link_proxy_radius_for_original_index(index)
                    + self.spec.collision_inflation_m
                )
                shape = p.createCollisionShape(
                    p.GEOM_CAPSULE,
                    radius=radius,
                    height=max(1e-4, segment_length - 2.0 * radius),
                )
                visual = -1
                if self.gui:
                    visual = p.createVisualShape(
                        p.GEOM_CAPSULE,
                        radius=radius,
                        length=max(1e-4, segment_length - 2.0 * radius),
                        rgbaColor=[0.16, 0.45, 0.85, 1.0] if side == "left" else [0.85, 0.38, 0.16, 1.0],
                    )
                body_id = self._new_body(f"{side}_link_{index + 1}", shape, visual)
                self.segment_ids[side].append(body_id)
            for index in range(self.spec.per_side_dof):
                original_joint_index = active_indices[index]
                joint_radius = (
                    self.spec.joint_proxy_radius_for_original_index(original_joint_index)
                    + self.spec.collision_inflation_m
                )
                shape = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=joint_radius,
                )
                visual = -1
                if self.gui:
                    visual = p.createVisualShape(
                        p.GEOM_SPHERE,
                        radius=joint_radius,
                        rgbaColor=[0.2, 0.2, 0.2, 1.0],
                    )
                body_id = self._new_body(f"{side}_joint_{index + 1}", shape, visual)
                self.joint_ids[side].append(body_id)

        # 这些接触属于同一实体的装配邻接关系，不能被自碰撞检查误报。
        for side in ("left", "right"):
            groups = self.segment_ids[side], self.joint_ids[side]
            for first, second in combinations(groups[0], 2):
                if abs(self.segment_ids[side].index(first) - self.segment_ids[side].index(second)) <= 1:
                    self._allow_pair(first, second)
            # A joint at boundary ``j`` is adjacent to the span before it and
            # the span after it.  The old implementation paired joint[j]
            # with span[j] and span[j+1], which shifted this relation by one
            # segment and produced false self-collisions at normal poses.
            for joint_index, joint in enumerate(groups[1]):
                for segment_index in (joint_index - 1, joint_index):
                    if 0 <= segment_index < len(groups[0]):
                        self._allow_pair(joint, groups[0][segment_index])
            self._allow_pair(self.central_id, groups[0][0])
            self._allow_pair(self.central_id, groups[1][0])
            self._allow_pair(self.endpoint_ids["base_end" if side == "left" else "l8_end"], groups[0][-1])
            self._allow_pair(self.endpoint_ids["base_end" if side == "left" else "l8_end"], groups[1][-1])
            if side == "left" and len(groups[0]) >= 2:
                # base_link与J1之前的左侧末段属于父子装配邻接；
                # 与右侧L8↔J7的对称局部重叠同样不能作为机构自碰撞。
                self._allow_pair(self.endpoint_ids["base_end"], groups[0][-2])
            if side == "right" and len(groups[0]) >= 2:
                # 右端L8真实mesh位于J8之后，同时与J8对应段和末端吸盘
                # 偏置段相邻；两者都是装配邻接，不应被自碰撞重复报告。
                self._allow_pair(self.endpoint_ids["l8_end"], groups[0][-2])
        self._allow_pair(self.central_id, self.endpoint_ids["base_end"])
        self._allow_pair(self.central_id, self.endpoint_ids["l8_end"])

    def _allow_pair(self, first: int | None, second: int | None) -> None:
        if first is None or second is None:
            return
        self._allowed_adjacent_pairs.add(tuple(sorted((first, second))))

    def _set_pose(self, body_id: int, pose: np.ndarray) -> None:
        p.resetBasePositionAndOrientation(
            body_id,
            pose[:3, 3].tolist(),
            rotation_matrix_to_quaternion(pose[:3, :3]).tolist(),
        )

    def _set_segment_pose(self, body_id: int, start: np.ndarray, end: np.ndarray) -> None:
        vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(vector))
        if length < 1e-9:
            length = 1e-9
        midpoint = 0.5 * (np.asarray(start) + np.asarray(end))
        p.resetBasePositionAndOrientation(body_id, midpoint.tolist(), _quaternion_from_z_axis(vector).tolist())

    def update(self, state: MorphologyState) -> None:
        """把代理更新为给定参数化机构状态。"""

        self._set_pose(self.central_id, state.body_pose)
        for endpoint, pose in (
            ("base_end", state.left_endpoint_link_pose),
            ("l8_end", state.right_endpoint_link_pose),
        ):
            self._set_pose(self.endpoint_ids[endpoint], pose)

        for side, segments, joints in (
            ("left", state.left_span_segments, state.left_joint_poses),
            ("right", state.right_span_segments, state.right_joint_poses),
        ):
            for body_id, (start, end) in zip(self.segment_ids[side], segments):
                self._set_segment_pose(body_id, start, end)
            for body_id, pose in zip(self.joint_ids[side], joints):
                p.resetBasePositionAndOrientation(
                    body_id,
                    pose[:3, 3].tolist(),
                    [0.0, 0.0, 0.0, 1.0],
                )
        p.performCollisionDetection()

    def check(
        self,
        *,
        allowed_endpoint_positions: dict[str, np.ndarray] | None = None,
        allowed_contact_radius_m: float = 0.09,
        distance_margin_m: float = 0.0,
    ) -> ProxyCollisionResult:
        """检查Tower和代理自碰撞，允许指定吸盘附近的合法Tower接触。"""

        allowed_endpoint_positions = allowed_endpoint_positions or {}
        minimum = float("inf")
        critical: str | None = None
        kind: str | None = None
        critical_position: np.ndarray | None = None

        def consider(distance: float, name: str, collision_kind: str) -> bool:
            nonlocal minimum, critical, kind, critical_position
            if distance < minimum:
                minimum = float(distance)
                critical = name
                kind = collision_kind
                critical_position = None
            return distance <= distance_margin_m

        if self.tower_id is not None:
            robot_items = [(self.central_id, "central_body")]
            robot_items.extend((body, name) for body, name in self._body_names.items() if name.startswith(("left_", "right_")))
            robot_items.extend((body, f"{endpoint}_mesh") for endpoint, body in self.endpoint_ids.items())
            for body_id, name in robot_items:
                query_distance_m = max(float(distance_margin_m), 0.02)
                points = p.getClosestPoints(body_id, self.tower_id, distance=query_distance_m)
                for point in points:
                    distance = float(point[8])
                    endpoint = "base_end" if name == "base_end_mesh" else "l8_end" if name == "l8_end_mesh" else None
                    if endpoint is None and name in {"left_link_4", "left_joint_" + str(self.spec.per_side_dof)}:
                        endpoint = "base_end"
                    # right_link_3 is J7->J8 and is a real intermediate link;
                    # it must never inherit the moving suction contact waiver.
                    if endpoint is None and name in {"right_joint_" + str(self.spec.per_side_dof)}:
                        endpoint = "l8_end"
                    if endpoint is not None and endpoint in allowed_endpoint_positions:
                        contact_point = np.asarray(point[5], dtype=float)
                        if np.linalg.norm(contact_point - allowed_endpoint_positions[endpoint]) <= allowed_contact_radius_m:
                            continue
                    if consider(distance, name, "TOWER_COLLISION"):
                        critical_position = np.asarray(point[5], dtype=float)
                        return ProxyCollisionResult(
                            False,
                            minimum,
                            critical,
                            kind,
                            critical_position,
                            "机器人代理与Tower发生碰撞",
                            np.asarray(point[5], dtype=float),
                            np.asarray(point[6], dtype=float),
                            max(0.0, -distance),
                        )

        robot_ids = [self.central_id, *self.endpoint_ids.values(), *self.segment_ids["left"], *self.segment_ids["right"], *self.joint_ids["left"], *self.joint_ids["right"]]
        robot_ids = [body for body in robot_ids if body is not None]
        for first, second in combinations(robot_ids, 2):
            if tuple(sorted((first, second))) in self._allowed_adjacent_pairs:
                continue
            # Query a small fixed neighborhood even when the acceptance margin
            # is zero.  PyBullet's concave-mesh distance query can otherwise
            # return no record for a real overlap after a reset/update.
            query_distance_m = max(float(distance_margin_m), 0.02)
            points = p.getClosestPoints(first, second, distance=query_distance_m)
            if not points:
                continue
            point = min(points, key=lambda item: float(item[8]))
            distance = float(point[8])
            if consider(distance, f"{self._body_names.get(first, first)}↔{self._body_names.get(second, second)}", "SELF_COLLISION"):
                critical_position = np.asarray(point[5], dtype=float)
                return ProxyCollisionResult(
                    False,
                    minimum,
                    critical,
                    kind,
                    critical_position,
                    "机构代理发生自碰撞",
                    np.asarray(point[5], dtype=float),
                    np.asarray(point[6], dtype=float),
                    max(0.0, -distance),
                )

        if not np.isfinite(minimum):
            minimum = float("inf")
        return ProxyCollisionResult(True, minimum, critical, kind, critical_position, "")
