"""支撑脚切换和变基座连续性检查。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrame
from environment.transforms import RigidTransform


@dataclass(frozen=True)
class RebaseResult:
    """一次支撑脚切换的验证结果。"""

    old_support_name: str
    new_support_name: str
    target_support_pose: RigidTransform
    base_pose: RigidTransform
    max_link_position_change_m: float
    max_link_orientation_change_rad: float
    new_support_position_before: np.ndarray
    new_support_position_after: np.ndarray

    @property
    def continuous(self) -> bool:
        """判断变基座是否只产生浮点误差级别的变化。"""

        return (
            self.max_link_position_change_m <= 1e-7
            and self.max_link_orientation_change_rad <= 1e-7
        )


def rebase_to_support(
    scene: PyBulletScene,
    *,
    joints: np.ndarray,
    old_support_name: str,
    new_support_name: str,
    new_support_frame: SuctionFrame,
    target_support_pose: RigidTransform | None = None,
) -> RebaseResult:
    """把当前机器人状态切换到新的吸盘支撑端，并验证世界几何连续。

    变基座不是重新加载机器人，也不是重置关节。这里保留同一组关节角，
    先记录所有link世界位姿，再启用新的吸盘锚定并重复设置同一状态，最后
    比较所有link的世界位置和姿态。
    """

    scene.disable_support_anchor()
    scene.reset_joints(np.asarray(joints, dtype=float))
    before = scene.capture_link_poses()
    support_before = scene.get_suction_pose(new_support_frame)
    anchor_pose = target_support_pose or support_before

    scene.enable_support_anchor(new_support_frame, anchor_pose)
    # 重新应用同一关节状态，触发锚定逻辑计算新的world base pose。
    scene.reset_joints(np.asarray(joints, dtype=float))
    after = scene.capture_link_poses()
    support_after = scene.get_suction_pose(new_support_frame)

    position_changes = [
        float(np.linalg.norm(after[index].position - pose.position))
        for index, pose in before.items()
    ]
    orientation_changes = [
        _orientation_distance_rad(pose, after[index])
        for index, pose in before.items()
    ]
    result = RebaseResult(
        old_support_name=old_support_name,
        new_support_name=new_support_name,
        target_support_pose=anchor_pose,
        base_pose=scene.get_base_pose(),
        max_link_position_change_m=max(position_changes, default=0.0),
        max_link_orientation_change_rad=max(orientation_changes, default=0.0),
        new_support_position_before=support_before.position.copy(),
        new_support_position_after=support_after.position.copy(),
    )

    print("Rebase continuity check", flush=True)
    print(f"old support: {result.old_support_name}", flush=True)
    print(f"new support: {result.new_support_name}", flush=True)
    print(f"max link position change: {result.max_link_position_change_m:.12e} m", flush=True)
    print(
        "max link orientation change: "
        f"{result.max_link_orientation_change_rad:.12e} rad",
        flush=True,
    )
    print(
        "new support world position before: "
        f"{result.new_support_position_before.tolist()}",
        flush=True,
    )
    print(
        "new support world position after: "
        f"{result.new_support_position_after.tolist()}",
        flush=True,
    )
    if not result.continuous:
        raise RuntimeError("变基座连续性检查失败，停止第二步规划")
    return result


def _orientation_distance_rad(first: RigidTransform, second: RigidTransform) -> float:
    """使用旋转矩阵相对旋转角比较完整姿态，避免四元数正负号问题。"""

    relative = first.rotation_matrix.T @ second.rotation_matrix
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))
