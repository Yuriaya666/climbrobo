"""只播放已经保存的单步轨迹，不重新进行规划、IK或碰撞搜索。"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as p

from environment.candidates import CandidatePoint
from environment.one_step_planner import AttachmentPoseBuilder
from environment.paths import ProjectPaths
from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrameSet
from environment.trajectory_io import SavedTrajectory
from environment.transforms import normalize
from environment.urdf_resolver import ResolvedUrdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="播放已保存的单步关节轨迹")
    parser.add_argument("--repo-root", type=str, default=None, help="仓库根目录")
    parser.add_argument("--trajectory", type=str, default=None, help="轨迹NPZ路径")
    parser.add_argument("--seconds-per-state", type=float, default=0.08, help="每个轨迹状态停留秒数")
    parser.add_argument("--repeats", type=int, default=1, help="重复播放次数")
    parser.add_argument("--keep-open", action="store_true", help="播放结束后保持GUI打开")
    return parser


def _candidate(foot_name: str, point_id: int, region_id: int, xyz: np.ndarray, normal: np.ndarray) -> CandidatePoint:
    return CandidatePoint(
        foot_name=foot_name,
        point_id=point_id,
        region_id=region_id,
        xyz_m=np.asarray(xyz, dtype=float),
        normal=normalize(normal, name=f"{foot_name}保存法向"),
        uv_m=np.zeros(2, dtype=float),
    )


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    trajectory_path = paths.successful_trajectory_npz if args.trajectory is None else paths.repo_root / args.trajectory
    saved = SavedTrajectory.load(trajectory_path)
    frames = SuctionFrameSet.load(paths.suction_config)
    support = _candidate("foot1", saved.support_point_id, saved.support_region_id, saved.support_xyz_m, saved.support_normal)
    target = _candidate("foot2", -1, saved.target_segment_id, saved.target_xyz_m, saved.target_normal)
    pose_builder = AttachmentPoseBuilder()
    support_pose = pose_builder.build(
        support,
        preferred_y_reference_world=np.array([0.0, 0.0, 1.0], dtype=float),
    )
    base_pose = support_pose.multiply(frames.base_end.transform_link_to_suction.inverse())

    print(f"读取轨迹：{trajectory_path}（不重新规划）")
    print(f"轨迹方法：{saved.trajectory_method}，状态数：{len(saved.trajectory_rad)}，向上位移：{saved.vertical_progress_m:.6f} m")
    print("启动GUI前使用 DISPLAY=:1。")

    with PyBulletScene(paths, gui=True) as scene:
        scene.load_tower()
        scene.load_robot(base_pose)
        scene.reset_joints(saved.start_joints_rad)
        scene.highlight_robot()
        scene.focus_camera_on_robot()
        scene.draw_point(support.xyz_m, [1.0, 0.1, 0.1], size=12.0)
        scene.draw_point(target.xyz_m, [0.1, 0.3, 1.0], size=12.0)
        scene.draw_frame(support_pose, "base support")
        target_pose = pose_builder.build(
            target,
            preferred_y_reference_world=np.array([0.0, 0.0, 1.0], dtype=float),
            yaw_rad=saved.target_yaw_rad,
        )
        scene.draw_frame(target_pose, "L8 target")
        scene.play_joint_trajectory(
            saved.trajectory_rad,
            seconds_per_state=max(float(args.seconds_per_state), 0.0),
            repeats=max(int(args.repeats), 1),
            step_simulation=False,
        )
        scene.reset_joints(saved.goal_joints_rad)
        if args.keep_open:
            print("已播放保存轨迹，GUI保持打开，按Ctrl+C退出。")
            try:
                while p.isConnected():
                    scene.reset_joints(saved.goal_joints_rad)
                    time.sleep(1.0 / 60.0)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
