"""读取第一步已保存轨迹，变更支撑脚后只规划第二步。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as p

from environment.candidates import CandidatePoint
from environment.one_step_planner import (
    AttachmentPoseBuilder,
    OneStepPlanner,
    PlannerSettings,
    format_plan_result,
)
from environment.paths import ProjectPaths
from environment.rebase import rebase_to_support
from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrameSet
from environment.trajectory_io import SavedTrajectory
from environment.transforms import normalize
from environment.transforms import angle_between_vectors_rad, RigidTransform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从第一步保存终点开始规划第二步")
    parser.add_argument("--repo-root", type=Path, default=None, help="仓库根目录")
    parser.add_argument("--gui", action="store_true", help="第二步规划完成后打开GUI回放两步")
    parser.add_argument("--keep-open", action="store_true", help="回放结束后保持GUI打开")
    parser.add_argument("--seconds-per-state", type=float, default=0.08, help="每个轨迹状态停留秒数")
    parser.add_argument("--yaw-samples", type=int, default=16, help="第二步yaw采样数")
    parser.add_argument("--attach-search-spacing-m", type=float, default=0.25, help="连续线粗搜索间隔，单位m")
    parser.add_argument("--attach-refinement-spacing-m", type=float, default=0.125, help="连续线细化间隔，单位m")
    parser.add_argument("--ik-random-seeds", type=int, default=3, help="第二步随机IK初值数量")
    parser.add_argument("--progress-interval", type=int, default=50, help="目标搜索进度打印间隔")
    return parser


def _candidate_from_saved(saved: SavedTrajectory) -> CandidatePoint:
    """把第一步终点保存信息包装成第二步的固定支撑候选点。"""

    return CandidatePoint(
        foot_name=saved.moving_foot_name,
        point_id=int(saved.target_segment_id),
        region_id=int(saved.target_segment_id),
        xyz_m=np.asarray(saved.target_xyz_m, dtype=float),
        normal=normalize(saved.target_normal, name="第一步目标法向"),
        uv_m=np.zeros(2, dtype=float),
    )


def _load_step1_roles(saved: SavedTrajectory) -> tuple[str, str, str, str]:
    """兼容旧NPZ，同时优先使用新轨迹中的显式角色元数据。"""

    return (
        saved.support_foot_name or "foot1",
        saved.moving_foot_name or "foot2",
        saved.support_frame_name or "base_end",
        saved.moving_frame_name or "l8_end",
    )


def _combine_trajectories(
    paths: ProjectPaths,
    step1: SavedTrajectory,
    step2: SavedTrajectory,
) -> None:
    """保存两步拼接轨迹，并去掉步间重复的终点/起点状态。"""

    if not np.allclose(step1.goal_joints_rad, step2.start_joints_rad, atol=1e-8):
        raise ValueError("第二步起点关节角与第一步终点不一致，不能拼接")
    combined = np.vstack((step1.trajectory_rad, step2.trajectory_rad[1:]))
    boundary = np.asarray([len(step1.trajectory_rad) - 1, len(combined) - 1], dtype=np.int64)
    np.savez(
        paths.successful_two_step_trajectory_npz,
        trajectory_rad=combined,
        step1_trajectory_rad=np.asarray(step1.trajectory_rad, dtype=float),
        step2_trajectory_rad=np.asarray(step2.trajectory_rad, dtype=float),
        step_boundary_indices=boundary,
        joint_names=np.asarray(step1.joint_names),
        support_foot_names=np.asarray([step1.support_foot_name, step2.support_foot_name]),
        moving_foot_names=np.asarray([step1.moving_foot_name, step2.moving_foot_name]),
        support_frame_names=np.asarray([step1.support_frame_name, step2.support_frame_name]),
        moving_frame_names=np.asarray([step1.moving_frame_name, step2.moving_frame_name]),
        unit=np.asarray("m/rad"),
        coordinate_frame=np.asarray(step1.coordinate_frame),
    )
    print(
        f"两步连续轨迹已保存：{paths.successful_two_step_trajectory_npz}（两步拼接轨迹数据）",
        flush=True,
    )


def _measure_support_drift(
    paths: ProjectPaths,
    saved: SavedTrajectory,
    support_pose,
) -> tuple[float, float]:
    """重新加载第二步轨迹，逐状态测量固定支撑吸盘漂移。"""

    frames = SuctionFrameSet.load(paths.suction_config)
    support_frame = frames.l8_end if saved.support_frame_name == "l8_end" else frames.base_end
    base_position = saved.base_position_m
    base_orientation = saved.base_orientation_xyzw
    if base_position is None or base_orientation is None:
        raise ValueError("第二步轨迹缺少变基座回放所需的base pose序列")

    position_errors: list[float] = []
    orientation_errors: list[float] = []
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_tower()
        scene.load_robot(
            RigidTransform(
                position=base_position[0],
                quaternion_xyzw=base_orientation[0],
            )
        )
        scene.enable_support_anchor(support_frame, support_pose)
        for index, joints in enumerate(saved.trajectory_rad):
            scene.set_base_pose(
                RigidTransform(
                    position=base_position[index],
                    quaternion_xyzw=base_orientation[index],
                )
            )
            scene.reset_joints(joints)
            actual = scene.get_suction_pose(support_frame)
            position_errors.append(float(np.linalg.norm(actual.position - support_pose.position)))
            orientation_errors.append(angle_between_vectors_rad(actual.z_axis, support_pose.z_axis))
    return max(position_errors, default=0.0), max(orientation_errors, default=0.0)


def _playback_two_steps(paths: ProjectPaths, step1: SavedTrajectory, step2: SavedTrajectory, *, keep_open: bool, seconds_per_state: float) -> None:
    """只回放已保存的第一步和第二步，不重新规划。"""

    frames = SuctionFrameSet.load(paths.suction_config)
    pose_builder = AttachmentPoseBuilder()
    support = CandidatePoint(
        foot_name=step1.support_foot_name,
        point_id=step1.support_point_id,
        region_id=step1.support_region_id,
        xyz_m=step1.support_xyz_m,
        normal=normalize(step1.support_normal, name="第一步支撑法向"),
        uv_m=np.zeros(2, dtype=float),
    )
    support_pose = pose_builder.build(
        support,
        preferred_y_reference_world=np.array([0.0, 0.0, 1.0], dtype=float),
    )
    base_pose = support_pose.multiply(frames.base_end.transform_link_to_suction.inverse())

    with PyBulletScene(paths, gui=True) as scene:
        scene.load_tower()
        scene.load_robot(base_pose)
        scene.reset_joints(step1.start_joints_rad)
        scene.highlight_robot()
        scene.focus_camera_on_robot()
        scene.draw_point(step1.support_xyz_m, [1.0, 0.1, 0.1], size=12.0)
        scene.draw_point(step1.target_xyz_m, [0.1, 0.3, 1.0], size=12.0)
        scene.play_joint_trajectory(step1.trajectory_rad, seconds_per_state=seconds_per_state, repeats=1)
        print("STEP 1 FINISHED", flush=True)
        print(
            "SUPPORT SWITCH: "
            f"{step1.support_foot_name}/{step1.support_frame_name} -> "
            f"{step2.support_foot_name}/{step2.support_frame_name}",
            flush=True,
        )

        # 第二步NPZ包含每个状态的base pose。回放时逐帧恢复，避免把
        # 第二步误当作仍然固定原始URDF base。
        scene.draw_point(step2.target_xyz_m, [0.1, 0.9, 0.2], size=12.0)
        for index, joints in enumerate(step2.trajectory_rad):
            if step2.base_position_m is not None and step2.base_orientation_xyzw is not None:
                from environment.transforms import RigidTransform

                scene.set_base_pose(
                    RigidTransform(
                        position=step2.base_position_m[index],
                        quaternion_xyzw=step2.base_orientation_xyzw[index],
                    )
                )
            scene.reset_joints(joints)
            time.sleep(max(seconds_per_state, 0.0))
        scene.reset_joints(step2.goal_joints_rad)
        if keep_open:
            print("两步轨迹已播放，GUI保持打开，按Ctrl+C退出。", flush=True)
            try:
                while p.isConnected():
                    scene.reset_joints(step2.goal_joints_rad)
                    time.sleep(1.0 / 60.0)
            except KeyboardInterrupt:
                pass


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    paths.validate_required_files()
    step1 = SavedTrajectory.load(paths.successful_trajectory_npz)
    support_foot, moving_foot, support_frame_name, moving_frame_name = _load_step1_roles(step1)
    frames = SuctionFrameSet.load(paths.suction_config)
    pose_builder = AttachmentPoseBuilder()

    step1_support = CandidatePoint(
        foot_name=support_foot,
        point_id=step1.support_point_id,
        region_id=step1.support_region_id,
        xyz_m=step1.support_xyz_m,
        normal=normalize(step1.support_normal, name="第一步支撑法向"),
        uv_m=np.zeros(2, dtype=float),
    )
    step1_support_pose = pose_builder.build(
        step1_support,
        preferred_y_reference_world=np.array([0.0, 0.0, 1.0], dtype=float),
    )
    step1_base_pose = step1_support_pose.multiply(
        frames.base_end.transform_link_to_suction.inverse()
    )
    q_end_step1 = np.asarray(step1.trajectory_rad[-1], dtype=float)
    step2_support = _candidate_from_saved(step1)

    # 仅恢复并检查第一步终点，不重新运行第一步搜索。
    step2_start_moving_z = None
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_tower()
        scene.load_robot(step1_base_pose)
        scene.reset_joints(q_end_step1)
        actual_step1_end = scene.get_suction_pose(frames.l8_end)
        rebase = rebase_to_support(
            scene,
            joints=q_end_step1,
            old_support_name=f"{support_foot}/{support_frame_name}",
            new_support_name=f"{moving_foot}/{moving_frame_name}",
            new_support_frame=frames.l8_end,
            target_support_pose=actual_step1_end,
        )
        step2_start_moving_z = float(scene.get_suction_pose(frames.base_end).position[2])

    print(
        f"Step 1: support={support_foot}, moving={moving_foot}, "
        f"vertical progress={step1.vertical_progress_m:.6f} m, "
        f"target z={step1.target_xyz_m[2]:.6f} m, planner={step1.trajectory_method}",
        flush=True,
    )

    settings = PlannerSettings(
        yaw_samples=args.yaw_samples,
        attach_search_spacing_m=args.attach_search_spacing_m,
        attach_refinement_spacing_m=args.attach_refinement_spacing_m,
        ik_random_seeds=args.ik_random_seeds,
        progress_interval=args.progress_interval,
    )
    planner = OneStepPlanner(paths=paths, settings=settings, gui=False)
    result = planner.plan(
        support_foot_name=moving_foot,
        moving_foot_name=support_foot,
        support_frame_name=moving_frame_name,
        moving_frame_name=support_frame_name,
        support_candidate=step2_support,
        start_joints=q_end_step1,
        initial_base_pose=rebase.base_pose,
        support_suction_pose_override=rebase.target_support_pose,
        trajectory_output_npz=paths.successful_step2_trajectory_npz,
        trajectory_output_csv=paths.successful_step2_trajectory_csv,
    )
    print(format_plan_result(result))
    if not result.success:
        raise RuntimeError("第二步规划失败，未生成第二步成功轨迹")
    step2 = SavedTrajectory.load(paths.successful_step2_trajectory_npz)
    _combine_trajectories(paths, step1, step2)

    print(
        f"Step 2: support={step2.support_foot_name}, moving={step2.moving_foot_name}, "
        f"start moving-foot z={step2_start_moving_z:.6f} m, "
        f"target z={step2.target_xyz_m[2]:.6f} m, "
        f"vertical progress={step2.vertical_progress_m:.6f} m, "
        f"planner={step2.trajectory_method}",
        flush=True,
    )
    max_position_drift, max_orientation_drift = _measure_support_drift(
        paths,
        step2,
        rebase.target_support_pose,
    )
    print(
        "Support constraint: "
        f"max position drift={max_position_drift:.12e} m, "
        f"max orientation drift={max_orientation_drift:.12e} rad",
        flush=True,
    )

    if args.gui:
        print("启动GUI前使用 DISPLAY=:1。", flush=True)
        _playback_two_steps(
            paths,
            step1,
            step2,
            keep_open=args.keep_open,
            seconds_per_state=args.seconds_per_state,
        )


if __name__ == "__main__":
    main()
