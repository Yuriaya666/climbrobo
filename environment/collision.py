from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrame
from environment.transforms import RigidTransform, angle_between_vectors_rad


@dataclass(frozen=True)
class CollisionItem:
    """一次碰撞或安全距离违规的简要记录。"""

    kind: str
    link_a: str
    link_b: str
    distance_m: float
    position_m: np.ndarray


@dataclass(frozen=True)
class CollisionReport:
    """某个轨迹状态的碰撞检测结果。"""

    ok: bool
    items: list[CollisionItem]

    @property
    def first_item(self) -> CollisionItem | None:
        return self.items[0] if self.items else None


class CollisionChecker:
    """检查机器人自碰撞和机器人与铁塔的碰撞。"""

    def __init__(
        self,
        scene: PyBulletScene,
        *,
        support_link_index: int,
        moving_link_index: int,
        support_point_m: np.ndarray,
        target_point_m: np.ndarray,
        collision_margin_m: float,
        allowed_contact_radius_m: float,
        support_suction_frame: SuctionFrame | None = None,
        support_suction_pose: RigidTransform | None = None,
        moving_start_point_m: np.ndarray | None = None,
        support_position_tolerance_m: float = 0.005,
        support_normal_tolerance_deg: float = 3.0,
    ) -> None:
        if scene.robot_id is None or scene.tower_id is None:
            raise RuntimeError("碰撞检查前必须先加载机器人和铁塔")

        self.scene = scene
        self.robot_id = scene.robot_id
        self.tower_id = scene.tower_id
        self.support_link_index = support_link_index
        self.moving_link_index = moving_link_index
        self.support_point_m = np.asarray(support_point_m, dtype=float)
        self.target_point_m = np.asarray(target_point_m, dtype=float)
        self.collision_margin_m = float(collision_margin_m)
        self.allowed_contact_radius_m = float(allowed_contact_radius_m)
        self._adjacent_pairs = scene.adjacent_link_pairs()
        self.support_suction_frame = support_suction_frame
        self.support_suction_pose = support_suction_pose
        self.moving_start_point_m = (
            None if moving_start_point_m is None else np.asarray(moving_start_point_m, dtype=float)
        )
        self.support_position_tolerance_m = float(support_position_tolerance_m)
        self.support_normal_tolerance_rad = np.deg2rad(float(support_normal_tolerance_deg))

    def check_state(
        self,
        *,
        allow_goal_contact: bool,
        allow_start_contact: bool = False,
    ) -> CollisionReport:
        p.performCollisionDetection()
        items: list[CollisionItem] = []
        items.extend(
            self._robot_tower_violations(
                allow_goal_contact=allow_goal_contact,
                allow_start_contact=allow_start_contact,
            )
        )
        items.extend(self._self_collision_violations())
        items.extend(self._support_constraint_violations())
        return CollisionReport(ok=(len(items) == 0), items=items)

    def _support_constraint_violations(self) -> list[CollisionItem]:
        if self.support_suction_frame is None or self.support_suction_pose is None:
            return []
        actual = self.scene.get_suction_pose(self.support_suction_frame)
        position_error = float(np.linalg.norm(actual.position - self.support_suction_pose.position))
        normal_error = angle_between_vectors_rad(actual.z_axis, self.support_suction_pose.z_axis)
        if position_error <= self.support_position_tolerance_m and normal_error <= self.support_normal_tolerance_rad:
            return []
        return [
            CollisionItem(
                kind="support_constraint",
                link_a=self.scene.link_name(self.support_link_index),
                link_b="support_target",
                distance_m=position_error,
                position_m=actual.position,
            )
        ]

    def _robot_tower_violations(
        self,
        *,
        allow_goal_contact: bool,
        allow_start_contact: bool,
    ) -> list[CollisionItem]:
        points = p.getClosestPoints(
            bodyA=self.robot_id,
            bodyB=self.tower_id,
            distance=self.collision_margin_m,
        )
        violations: list[CollisionItem] = []
        for point in points:
            link_index = int(point[3])
            distance = float(point[8])
            position_on_tower = np.asarray(point[6], dtype=float)
            position_on_robot = np.asarray(point[5], dtype=float)
            position = 0.5 * (position_on_tower + position_on_robot)

            if self._is_allowed_tower_contact(
                link_index=link_index,
                contact_position_m=position_on_tower,
                allow_goal_contact=allow_goal_contact,
                allow_start_contact=allow_start_contact,
            ):
                continue

            violations.append(
                CollisionItem(
                    kind="robot_tower",
                    link_a=self.scene.link_name(link_index),
                    link_b="Tower",
                    distance_m=distance,
                    position_m=position,
                )
            )
        return violations

    def _is_allowed_tower_contact(
        self,
        *,
        link_index: int,
        contact_position_m: np.ndarray,
        allow_goal_contact: bool,
        allow_start_contact: bool,
    ) -> bool:
        """只允许吸盘中心附近的预期接触，不放行整条link。"""

        if link_index == self.support_link_index:
            distance_to_support = float(
                np.linalg.norm(contact_position_m - self.support_point_m)
            )
            if distance_to_support <= self.allowed_contact_radius_m:
                return True

        if allow_goal_contact and link_index == self.moving_link_index:
            distance_to_target = float(np.linalg.norm(contact_position_m - self.target_point_m))
            if distance_to_target <= self.allowed_contact_radius_m:
                return True

        # 换脚的第一帧中，旧支撑脚仍可能与铁塔接触；只允许其原吸盘
        # 中心附近的接触，后续状态不默认放行运动脚接触。
        if (
            allow_start_contact
            and link_index == self.moving_link_index
            and self.moving_start_point_m is not None
        ):
            distance_to_start = float(
                np.linalg.norm(contact_position_m - self.moving_start_point_m)
            )
            if distance_to_start <= self.allowed_contact_radius_m:
                return True

        return False

    def _self_collision_violations(self) -> list[CollisionItem]:
        points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        violations: list[CollisionItem] = []
        seen_pairs: set[tuple[int, int]] = set()
        for point in points:
            link_a = int(point[3])
            link_b = int(point[4])
            pair = tuple(sorted((link_a, link_b)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # 父子相邻link在CAD网格边界处容易产生无意义接触，先排除。
            if pair in self._adjacent_pairs:
                continue

            violations.append(
                CollisionItem(
                    kind="self",
                    link_a=self.scene.link_name(link_a),
                    link_b=self.scene.link_name(link_b),
                    distance_m=float(point[8]),
                    position_m=np.asarray(point[5], dtype=float),
                )
            )
        return violations
