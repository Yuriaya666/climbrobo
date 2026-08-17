"""成功单步轨迹的持久化读写。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environment.attachment_semantics import canonical_surface_name


@dataclass(frozen=True)
class SavedTrajectory:
    """可脱离规划过程直接播放的一条完整关节轨迹。"""

    trajectory_rad: np.ndarray
    joint_names: tuple[str, ...]
    support_xyz_m: np.ndarray
    support_normal: np.ndarray
    support_point_id: int
    support_region_id: int
    target_xyz_m: np.ndarray
    target_normal: np.ndarray
    target_yaw_rad: float
    target_segment_id: int
    target_s_m: float
    start_joints_rad: np.ndarray
    goal_joints_rad: np.ndarray
    vertical_progress_m: float
    trajectory_method: str
    unit: str = "m/rad"
    coordinate_frame: str = "tower_stl_global"
    support_foot_name: str = "foot1"
    moving_foot_name: str = "foot2"
    support_frame_name: str = "base_end"
    moving_frame_name: str = "l8_end"
    base_position_m: np.ndarray | None = None
    base_orientation_xyzw: np.ndarray | None = None
    # 新语义：物理机器人端与附着面分开保存；旧轨迹缺失时沿用历史默认映射。
    support_surface_name: str = "surface1"
    target_surface_name: str = "surface2"

    def validate(self) -> None:
        trajectory = np.asarray(self.trajectory_rad, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[0] < 2:
            raise ValueError("trajectory_rad必须是至少包含两个状态的(N,J)数组")
        joint_count = trajectory.shape[1]
        if len(self.joint_names) != joint_count:
            raise ValueError("joint_names数量与轨迹关节数量不一致")
        if np.asarray(self.start_joints_rad).shape != (joint_count,):
            raise ValueError("start_joints_rad shape不正确")
        if np.asarray(self.goal_joints_rad).shape != (joint_count,):
            raise ValueError("goal_joints_rad shape不正确")
        for name, value, shape in (
            ("support_xyz_m", self.support_xyz_m, (3,)),
            ("support_normal", self.support_normal, (3,)),
            ("target_xyz_m", self.target_xyz_m, (3,)),
            ("target_normal", self.target_normal, (3,)),
        ):
            if np.asarray(value).shape != shape:
                raise ValueError(f"{name} shape应为{shape}")
        if self.unit != "m/rad":
            raise ValueError(f"轨迹单位应为m/rad，实际为{self.unit!r}")
        if not self.trajectory_method:
            raise ValueError("trajectory_method不能为空")
        canonical_surface_name(self.support_surface_name)
        canonical_surface_name(self.target_surface_name)
        for name in ("support_foot_name", "moving_foot_name", "support_frame_name", "moving_frame_name"):
            if not getattr(self, name):
                raise ValueError(f"{name}不能为空")
        if self.base_position_m is not None:
            base_position = np.asarray(self.base_position_m, dtype=float)
            if base_position.shape != (len(trajectory), 3):
                raise ValueError("base_position_m必须是(N,3)，且N等于轨迹状态数")
        if self.base_orientation_xyzw is not None:
            base_orientation = np.asarray(self.base_orientation_xyzw, dtype=float)
            if base_orientation.shape != (len(trajectory), 4):
                raise ValueError("base_orientation_xyzw必须是(N,4)，且N等于轨迹状态数")

    def save(self, npz_path: Path, csv_path: Path) -> None:
        """同时保存机器可读NPZ和便于查看的CSV。"""

        self.validate()
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            trajectory_rad=np.asarray(self.trajectory_rad, dtype=float),
            joint_names=np.asarray(self.joint_names),
            support_xyz_m=np.asarray(self.support_xyz_m, dtype=float),
            support_normal=np.asarray(self.support_normal, dtype=float),
            support_point_id=np.asarray(self.support_point_id, dtype=np.int64),
            support_region_id=np.asarray(self.support_region_id, dtype=np.int64),
            target_xyz_m=np.asarray(self.target_xyz_m, dtype=float),
            target_normal=np.asarray(self.target_normal, dtype=float),
            target_yaw_rad=np.asarray(self.target_yaw_rad, dtype=float),
            target_segment_id=np.asarray(self.target_segment_id, dtype=np.int64),
            target_s_m=np.asarray(self.target_s_m, dtype=float),
            start_joints_rad=np.asarray(self.start_joints_rad, dtype=float),
            goal_joints_rad=np.asarray(self.goal_joints_rad, dtype=float),
            vertical_progress_m=np.asarray(self.vertical_progress_m, dtype=float),
            trajectory_method=np.asarray(self.trajectory_method),
            unit=np.asarray(self.unit),
            coordinate_frame=np.asarray(self.coordinate_frame),
            support_foot_name=np.asarray(self.support_foot_name),
            moving_foot_name=np.asarray(self.moving_foot_name),
            support_frame_name=np.asarray(self.support_frame_name),
            moving_frame_name=np.asarray(self.moving_frame_name),
            support_surface_name=np.asarray(self.support_surface_name),
            target_surface_name=np.asarray(self.target_surface_name),
        )
        if self.base_position_m is not None:
            # base轨迹是变基座回放所需的附加数据，旧文件不受影响。
            arrays["base_position_m"] = np.asarray(self.base_position_m, dtype=float)
            if self.base_orientation_xyzw is not None:
                arrays["base_orientation_xyzw"] = np.asarray(self.base_orientation_xyzw, dtype=float)
        np.savez(npz_path, **arrays)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "alpha", *self.joint_names])
            count = len(self.trajectory_rad)
            for index, joints in enumerate(np.asarray(self.trajectory_rad, dtype=float)):
                alpha = index / (count - 1)
                writer.writerow([index, alpha, *joints.tolist()])

    @classmethod
    def load(cls, npz_path: Path) -> "SavedTrajectory":
        if not npz_path.exists():
            raise FileNotFoundError(f"找不到已保存轨迹：{npz_path}（可直接播放的轨迹数据）")
        data = np.load(npz_path, allow_pickle=False)
        required = {
            "trajectory_rad", "joint_names", "support_xyz_m", "support_normal",
            "support_point_id", "support_region_id", "target_xyz_m", "target_normal",
            "target_yaw_rad", "target_segment_id", "target_s_m", "start_joints_rad",
            "goal_joints_rad", "vertical_progress_m", "trajectory_method", "unit",
            "coordinate_frame",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{npz_path}缺少轨迹字段：{missing}")
        support_foot_name = (
            str(data["support_foot_name"].item())
            if "support_foot_name" in data
            else "foot1"
        )
        moving_foot_name = (
            str(data["moving_foot_name"].item())
            if "moving_foot_name" in data
            else "foot2"
        )
        result = cls(
            trajectory_rad=np.asarray(data["trajectory_rad"], dtype=float),
            joint_names=tuple(str(value) for value in data["joint_names"].tolist()),
            support_xyz_m=np.asarray(data["support_xyz_m"], dtype=float),
            support_normal=np.asarray(data["support_normal"], dtype=float),
            support_point_id=int(data["support_point_id"].item()),
            support_region_id=int(data["support_region_id"].item()),
            target_xyz_m=np.asarray(data["target_xyz_m"], dtype=float),
            target_normal=np.asarray(data["target_normal"], dtype=float),
            target_yaw_rad=float(data["target_yaw_rad"].item()),
            target_segment_id=int(data["target_segment_id"].item()),
            target_s_m=float(data["target_s_m"].item()),
            start_joints_rad=np.asarray(data["start_joints_rad"], dtype=float),
            goal_joints_rad=np.asarray(data["goal_joints_rad"], dtype=float),
            vertical_progress_m=float(data["vertical_progress_m"].item()),
            trajectory_method=str(data["trajectory_method"].item()),
            unit=str(data["unit"].item()),
            coordinate_frame=str(data["coordinate_frame"].item()),
            support_foot_name=support_foot_name,
            moving_foot_name=moving_foot_name,
            support_frame_name=str(data["support_frame_name"].item()) if "support_frame_name" in data else "base_end",
            moving_frame_name=str(data["moving_frame_name"].item()) if "moving_frame_name" in data else "l8_end",
            base_position_m=np.asarray(data["base_position_m"], dtype=float) if "base_position_m" in data else None,
            base_orientation_xyzw=np.asarray(data["base_orientation_xyzw"], dtype=float) if "base_orientation_xyzw" in data else None,
            support_surface_name=(
                str(data["support_surface_name"].item())
                if "support_surface_name" in data
                else canonical_surface_name(support_foot_name)
            ),
            target_surface_name=(
                str(data["target_surface_name"].item())
                if "target_surface_name" in data
                else canonical_surface_name(moving_foot_name)
            ),
        )
        result.validate()
        return result
