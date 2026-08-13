"""第三步高处终点诊断实验，不自动规划第四步。"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

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
from environment.transforms import RigidTransform, normalize


MIN_UPWARD_PROGRESS_M = 0.02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="诊断第三步是否存在更高的合法落脚终点")
    parser.add_argument("--repo-root", type=Path, default=None, help="仓库根目录")
    parser.add_argument("--min-upward-progress-m", type=float, default=MIN_UPWARD_PROGRESS_M, help="最低向上进度，单位m")
    parser.add_argument("--yaw-samples", type=int, default=32, help="第三步yaw粗采样数量")
    parser.add_argument("--ik-random-seeds", type=int, default=12, help="第三步随机IK初值数量")
    parser.add_argument("--numerical-ik-iterations", type=int, default=240, help="第三步数值IK迭代次数")
    parser.add_argument("--ik-max-iterations", type=int, default=600, help="第三步PyBullet IK迭代次数")
    parser.add_argument("--attach-search-spacing-m", type=float, default=0.12, help="第三步粗搜索弧长间隔，单位m")
    parser.add_argument("--attach-refinement-spacing-m", type=float, default=0.04, help="第三步细化弧长间隔，单位m")
    parser.add_argument("--attach-refinement-window-m", type=float, default=0.8, help="第三步局部细化窗口，单位m")
    parser.add_argument("--rrt-max-iterations", type=int, default=6000, help="每个RRT随机种子的最大迭代数")
    parser.add_argument("--rrt-seed-count", type=int, default=5, help="独立RRT随机种子数量")
    parser.add_argument("--progress-interval", type=int, default=50, help="搜索进度打印间隔")
    parser.add_argument("--run-trajectory", action="store_true", help="高处合法终点存在时继续运行直线/RRT")
    return parser


def _candidate_from_saved(saved: SavedTrajectory, foot_name: str) -> CandidatePoint:
    return CandidatePoint(
        foot_name=foot_name,
        point_id=int(saved.target_segment_id),
        region_id=int(saved.target_segment_id),
        xyz_m=np.asarray(saved.target_xyz_m, dtype=float),
        normal=normalize(saved.target_normal, name="保存目标法向"),
        uv_m=np.zeros(2, dtype=float),
    )


def _load_step2_state(paths: ProjectPaths, step2: SavedTrajectory):
    """从第二步末帧恢复真实base和关节状态，再把foot1设为支撑端。"""

    frames = SuctionFrameSet.load(paths.suction_config)
    if step2.base_position_m is None or step2.base_orientation_xyzw is None:
        raise ValueError("第二步轨迹缺少base pose序列，无法严格恢复第二步终点")
    base_pose = RigidTransform(
        position=step2.base_position_m[-1],
        quaternion_xyzw=step2.base_orientation_xyzw[-1],
    )
    q_end_step2 = np.asarray(step2.trajectory_rad[-1], dtype=float)
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_tower()
        scene.load_robot(base_pose)
        scene.reset_joints(q_end_step2)
        old_support_pose = scene.get_suction_pose(frames.l8_end)
        new_support_pose = scene.get_suction_pose(frames.base_end)
        rebase = rebase_to_support(
            scene,
            joints=q_end_step2,
            old_support_name="foot2/l8_end",
            new_support_name="foot1/base_end",
            new_support_frame=frames.base_end,
            target_support_pose=new_support_pose,
        )
        current_moving_pose = scene.get_suction_pose(frames.l8_end)
        link_pose_after_rebase = scene.capture_link_poses()
    return frames, q_end_step2, rebase, current_moving_pose, link_pose_after_rebase


def _write_high_goal_diagnostics(
    path: Path,
    internal_path: Path,
    minimum_z: float,
) -> tuple[Counter, dict[str, float], int]:
    """从逐yaw内部诊断生成高处目标统计和最终CSV。"""

    rows: list[dict[str, object]] = []
    counts: Counter = Counter()
    unique_targets: set[tuple[int, float]] = set()
    highest = {
        "attach": float("nan"),
        "geometric": float("nan"),
        "ik": float("nan"),
        "pose": float("nan"),
        "goal": float("nan"),
    }
    ik_satisfied = {"SELF_COLLISION", "GOAL_TOWER_COLLISION", "GOAL_SUCCESS"}

    with internal_path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    for source in source_rows:
        z = float(source["z_m"])
        if z <= minimum_z:
            continue
        segment_id = int(source["segment_id"])
        s_m = float(source["s_m"])
        unique_targets.add((segment_id, round(s_m, 9)))
        highest["attach"] = z if not np.isfinite(highest["attach"]) else max(highest["attach"], z)

        raw_reason = source["failure_reason"]
        if raw_reason == "SUCCESS":
            category = "GOAL_SUCCESS"
        elif raw_reason in {
            "GEOMETRICALLY_UNREACHABLE", "IK_FAILED", "IK_POSITION_ERROR",
            "IK_NORMAL_ERROR", "JOINT_LIMIT_FAILED", "SELF_COLLISION",
            "GOAL_TOWER_COLLISION",
        }:
            category = raw_reason
        elif raw_reason.startswith("IK_POSITION_ERROR"):
            category = "IK_POSITION_ERROR"
        elif raw_reason.startswith("IK_NORMAL_ERROR"):
            category = "IK_NORMAL_ERROR"
        else:
            category = "IK_FAILED"
        counts[category] += 1

        if category != "GEOMETRICALLY_UNREACHABLE":
            highest["geometric"] = (
                z if not np.isfinite(highest["geometric"])
                else max(highest["geometric"], z)
            )

        # SELF_COLLISION/GOAL_TOWER_COLLISION在当前planner中表示IK已满足，
        # 但终点碰撞检查没有通过，因此可用于区分“终点不存在”和“姿态未收敛”。
        if category in ik_satisfied:
            highest["ik"] = z if not np.isfinite(highest["ik"]) else max(highest["ik"], z)
            highest["pose"] = z if not np.isfinite(highest["pose"]) else max(highest["pose"], z)
        if category == "GOAL_SUCCESS":
            highest["goal"] = z if not np.isfinite(highest["goal"]) else max(highest["goal"], z)

        rows.append(
            {
                "segment_id": segment_id,
                "s_m": s_m,
                "x_m": source["x_m"],
                "y_m": source["y_m"],
                "z_m": z,
                "yaw_rad": source["yaw_rad"],
                "vertical_progress_m": source["vertical_progress_m"],
                "position_error_m": source["position_error_m"],
                "normal_error_deg": source["normal_error_deg"],
                "failure_reason": category,
                "detail": raw_reason,
            }
        )

    counts["HIGHER_ATTACH_TARGETS"] = len(unique_targets)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "segment_id", "s_m", "x_m", "y_m", "z_m", "yaw_rad",
                "vertical_progress_m", "position_error_m", "normal_error_deg",
                "failure_reason", "detail",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return counts, highest, len(unique_targets)


def _save_goal_candidates(path: Path, result, minimum_progress_m: float) -> None:
    """保存高处合法终点的q_goal和目标几何，失败时不生成成功轨迹。"""

    endpoints = [item for item in result.endpoint_candidates if item.vertical_progress_m > minimum_progress_m]
    if not endpoints:
        if path.exists():
            path.unlink()
        return
    np.savez(
        path,
        segment_id=np.asarray([item.segment_id for item in endpoints], dtype=np.int64),
        s_m=np.asarray([item.s_m for item in endpoints], dtype=float),
        yaw_rad=np.asarray([item.yaw_rad for item in endpoints], dtype=float),
        target_xyz_m=np.asarray([item.target.xyz_m for item in endpoints], dtype=float),
        target_normal=np.asarray([item.target.normal for item in endpoints], dtype=float),
        goal_joints_rad=np.asarray([item.goal_joints for item in endpoints], dtype=float),
        position_error_m=np.asarray([item.position_error_m for item in endpoints], dtype=float),
        normal_error_deg=np.asarray([item.normal_error_deg for item in endpoints], dtype=float),
        vertical_progress_m=np.asarray([item.vertical_progress_m for item in endpoints], dtype=float),
    )


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    paths.validate_required_files()
    step2 = SavedTrajectory.load(paths.successful_step2_trajectory_npz)
    frames, q_end_step2, rebase, current_moving_pose, _ = _load_step2_state(paths, step2)
    current_z = float(current_moving_pose.position[2])
    minimum_z = current_z + float(args.min_upward_progress_m)

    settings = PlannerSettings(
        yaw_samples=max(32, int(args.yaw_samples)),
        ik_random_seeds=max(12, int(args.ik_random_seeds)),
        numerical_ik_iterations=max(240, int(args.numerical_ik_iterations)),
        ik_max_iterations=max(600, int(args.ik_max_iterations)),
        attach_search_spacing_m=float(args.attach_search_spacing_m),
        attach_refinement_spacing_m=float(args.attach_refinement_spacing_m),
        attach_refinement_window_m=float(args.attach_refinement_window_m),
        endpoint_only=not args.run_trajectory,
        skip_trajectory_planning=not args.run_trajectory,
        min_vertical_progress_m=float(args.min_upward_progress_m),
        rrt_max_iterations=max(6000, int(args.rrt_max_iterations)),
        rrt_seed_count=max(5, int(args.rrt_seed_count)),
        pybullet_ik_seed_count=2,
        warm_start_seed_count=3,
        use_pybullet_ik_seeds=False,
        # 第三步诊断只约束吸盘位置和接触法向，不把绕法向的切向轴
        # 当作额外接触条件，避免过度约束高处终点。
        ik_orientation_mode="normal_only",
        ik_jacobian_mode="pybullet",
        progress_interval=int(args.progress_interval),
        diagnostic_output_name="step3_internal_ik_diagnostics.csv",
        planning_diagnostic_output_name="step3_internal_planning_diagnostics.csv",
    )

    # 第二步末尾的旧支撑脚foot2已作为支撑；第三步重新交换为foot1支撑、foot2运动。
    support = _candidate_from_saved(step2, "foot1")
    planner = OneStepPlanner(paths=paths, settings=settings, gui=False)
    result = planner.plan(
        support_foot_name="foot1",
        moving_foot_name="foot2",
        support_frame_name="base_end",
        moving_frame_name="l8_end",
        support_candidate=support,
        start_joints=q_end_step2,
        initial_base_pose=rebase.base_pose,
        support_suction_pose_override=rebase.target_support_pose,
        trajectory_output_npz=paths.step3_trajectory_npz,
        trajectory_output_csv=paths.step3_trajectory_csv,
    )

    internal_path = paths.candidate_dir / settings.diagnostic_output_name
    counts, highest, higher_target_count = _write_high_goal_diagnostics(
        paths.step3_higher_goal_diagnostics_csv,
        internal_path,
        minimum_z,
    )
    _save_goal_candidates(
        paths.step3_higher_goal_candidates_npz,
        result,
        float(args.min_upward_progress_m),
    )

    higher_endpoints = [item for item in result.endpoint_candidates if item.target.xyz_m[2] > minimum_z]
    print("========== Step 3 higher-goal diagnostic ==========")
    print("Step 3 support: foot1/base_end")
    print("Step 3 moving foot: foot2/l8_end")
    print(f"当前moving foot高度: {current_z:.6f} m")
    print(f"强制最低目标高度: {minimum_z:.6f} m")
    print(f"更高attach目标数量: {higher_target_count}")
    print(f"最高IK/终点合法目标z: {max((item.target.xyz_m[2] for item in higher_endpoints), default=float('nan')):.6f} m")
    print(f"高处合法q_goal数量: {len(higher_endpoints)}")
    print("失败原因统计:")
    for key in (
        "HIGHER_ATTACH_TARGETS", "GEOMETRICALLY_UNREACHABLE", "IK_FAILED",
        "IK_POSITION_ERROR", "IK_NORMAL_ERROR", "JOINT_LIMIT_FAILED",
        "SELF_COLLISION", "GOAL_TOWER_COLLISION", "GOAL_SUCCESS",
    ):
        print(f"  {key}: {counts.get(key, 0)}")
    print(f"最高附着目标z: {highest['attach']:.6f} m")
    print(f"最高通过几何上界并进入IK的目标z: {highest['geometric']:.6f} m")
    print(f"最高IK收敛z: {highest['ik']:.6f} m")
    print(f"最高姿态满足z: {highest['pose']:.6f} m")
    print(f"最高无碰撞q_goal z: {highest['goal']:.6f} m")
    print(f"诊断CSV: {paths.step3_higher_goal_diagnostics_csv}（第三步高处终点诊断表）")
    if higher_endpoints:
        best = max(higher_endpoints, key=lambda item: item.target.xyz_m[2])
        print("是否存在高于当前高度的合法终点: 是")
        print(f"最佳target z: {best.target.xyz_m[2]:.6f} m")
        print(f"vertical progress: {best.vertical_progress_m:.6f} m")
        if args.run_trajectory:
            print(f"轨迹结果: {result.trajectory_method or '失败'}")
        else:
            print("轨迹结果: 本轮未运行，使用--run-trajectory后才执行straight/RRT")
        if args.run_trajectory and result.success:
            print(f"第三步轨迹NPZ: {paths.step3_trajectory_npz}（第三步成功轨迹数据）")
            print(f"第三步轨迹CSV: {paths.step3_trajectory_csv}（第三步逐状态关节角）")
        print("最后判断: 高处终点存在，是否能找到轨迹取决于后续轨迹阶段结果。")
    else:
        print("是否存在高于当前高度的合法终点: 否")
        print("主要失败原因: 详见诊断CSV中的统计；没有生成假的第三步成功轨迹。")
        print("最后判断: 高处终点本身找不到。")


if __name__ == "__main__":
    main()
