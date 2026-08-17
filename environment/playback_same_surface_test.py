"""只回放已经保存的同面附着测试轨迹，不重新规划。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pybullet as p

from environment.candidates import CandidatePoint
from environment.one_step_planner import AttachmentPoseBuilder
from environment.paths import ProjectPaths
from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrameSet
from environment.trajectory_io import SavedTrajectory
from environment.transforms import normalize, RigidTransform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回放Test A或Test B已保存轨迹")
    parser.add_argument("--case", choices=("A", "B"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--seconds-per-state", type=float, default=0.08)
    parser.add_argument("--keep-open", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    case_path = paths.same_surface_test_a_npz if args.case == "A" else paths.same_surface_test_b_npz
    saved = SavedTrajectory.load(case_path)
    if saved.base_position_m is None or saved.base_orientation_xyzw is None:
        raise ValueError("同面测试轨迹缺少base pose序列，无法回放")
    frames = SuctionFrameSet.load(paths.suction_config)
    support_frame = frames.base_end if saved.support_frame_name == "base_end" else frames.l8_end
    moving_frame = frames.base_end if saved.moving_frame_name == "base_end" else frames.l8_end
    support = CandidatePoint(
        foot_name=saved.support_frame_name,
        point_id=saved.support_point_id,
        region_id=saved.support_region_id,
        xyz_m=saved.support_xyz_m,
        normal=normalize(saved.support_normal, name="同面测试支撑法向"),
        uv_m=[0.0, 0.0],
        surface_name=saved.support_surface_name,
    )
    support_pose = AttachmentPoseBuilder().build(
        support,
        preferred_y_reference_world=[0.0, 0.0, 1.0],
    )
    with PyBulletScene(paths, gui=True) as scene:
        scene.load_tower()
        scene.load_robot(
            RigidTransform(
                position=saved.base_position_m[0],
                quaternion_xyzw=saved.base_orientation_xyzw[0],
            )
        )
        scene.highlight_robot()
        scene.focus_camera_on_robot()
        scene.draw_point(saved.support_xyz_m, [1.0, 0.1, 0.1], size=13.0)
        scene.draw_point(saved.target_xyz_m, [0.1, 0.9, 0.2], size=13.0)
        scene.draw_frame(support_pose, "support surface target")
        for index, joints in enumerate(saved.trajectory_rad):
            scene.set_base_pose(
                RigidTransform(
                    position=saved.base_position_m[index],
                    quaternion_xyzw=saved.base_orientation_xyzw[index],
                )
            )
            scene.reset_joints(joints)
            time.sleep(max(args.seconds_per_state, 0.0))
        scene.reset_joints(saved.goal_joints_rad)
        print(
            f"Test {args.case}轨迹播放完成：support={saved.support_frame_name} "
            f"on {saved.support_surface_name}, moving={saved.moving_frame_name} "
            f"to {saved.target_surface_name}",
            flush=True,
        )
        if args.keep_open:
            try:
                while p.isConnected():
                    scene.set_base_pose(
                        RigidTransform(
                            position=saved.base_position_m[-1],
                            quaternion_xyzw=saved.base_orientation_xyzw[-1],
                        )
                    )
                    scene.reset_joints(saved.goal_joints_rad)
                    time.sleep(1.0 / 60.0)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
