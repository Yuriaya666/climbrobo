"""固定跨障目标实验：只验证一个指定高处落脚点，不搜索最高点。"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p

from environment.attach_lines import AttachLineSample, AttachLineSet
from environment.candidates import CandidatePoint
from environment.collision import CollisionChecker, CollisionReport
from environment.ik import NumericalSuctionIKSolver
from environment.one_step_planner import AttachmentPoseBuilder, OneStepPlanner, PlannerSettings
from environment.paths import ProjectPaths
from environment.rebase import RebaseResult, rebase_to_support
from environment.rrt_connect import RRTConnect
from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrameSet
from environment.trajectory_io import SavedTrajectory
from environment.transforms import RigidTransform, angle_between_vectors_rad, normalize


@dataclass(frozen=True)
class FixedTarget:
    """本实验唯一允许使用的跨障目标。"""

    segment_id: int
    s_m: float
    sample: AttachLineSample


@dataclass(frozen=True)
class IKSearchResult:
    """固定目标的IK搜索结果，包含best-effort和所有合法终点候选。"""

    best_joints: np.ndarray
    best_position_error_m: float
    best_normal_error_deg: float
    best_yaw_rad: float
    valid_goals: list[tuple[np.ndarray, float, float, float]]
    counts: Counter
    rows: list[dict[str, object]]


@dataclass(frozen=True)
class RRTAttempt:
    """一次独立RRT运行的结果和诊断。"""

    seed: int
    success: bool
    iterations: int
    tree_nodes: int
    planning_time_s: float
    nearest_goal_distance: float
    first_collision_link: str
    first_collision_kind: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="固定障碍后第一个落脚点的跨障实验")
    parser.add_argument("--repo-root", type=Path, default=None, help="仓库根目录")
    parser.add_argument("--gui", action="store_true", help="用PyBullet GUI显示实验状态和结果")
    parser.add_argument("--keep-open", action="store_true", help="GUI显示结束后保持打开")
    parser.add_argument("--seconds-per-state", type=float, default=0.08, help="成功轨迹每状态停留时间")
    parser.add_argument("--yaw-samples", type=int, default=32, help="固定目标yaw粗采样数")
    parser.add_argument("--ik-random-seeds", type=int, default=24, help="每个yaw的随机IK种子数")
    parser.add_argument("--numerical-ik-iterations", type=int, default=420, help="数值IK最大迭代数")
    parser.add_argument("--ik-max-iterations", type=int, default=1000, help="PyBullet IK最大迭代数")
    parser.add_argument("--rrt-runs", type=int, default=10, help="独立RRT运行次数")
    parser.add_argument("--rrt-max-iterations", type=int, default=10000, help="每次RRT最大迭代数")
    parser.add_argument("--rrt-seed-base", type=int, default=20260813, help="RRT随机种子起点")
    parser.add_argument("--target-s-m", type=float, default=None, help="仅用于复现实验的目标弧长覆盖值")
    return parser


def _candidate_from_saved(saved: SavedTrajectory, foot_name: str) -> CandidatePoint:
    """把第二步终点的运动脚目标保存信息变成第三步支撑候选点。"""

    return CandidatePoint(
        foot_name=foot_name,
        point_id=int(saved.target_segment_id),
        region_id=int(saved.target_segment_id),
        xyz_m=np.asarray(saved.target_xyz_m, dtype=float),
        normal=normalize(saved.target_normal, name="第二步目标法向"),
        uv_m=np.zeros(2, dtype=float),
    )


def _restore_step2_and_rebase(
    paths: ProjectPaths,
    step2: SavedTrajectory,
    frames: SuctionFrameSet,
) -> tuple[np.ndarray, RebaseResult, CandidatePoint, RigidTransform, float, float]:
    """恢复第二步末帧，并把foot1/base_end变成第三步固定支撑端。"""

    if step2.base_position_m is None or step2.base_orientation_xyzw is None:
        raise ValueError("第二步轨迹缺少末帧base pose，无法严格恢复")

    q_start = np.asarray(step2.trajectory_rad[-1], dtype=float)
    base_pose = RigidTransform(
        position=step2.base_position_m[-1],
        quaternion_xyzw=step2.base_orientation_xyzw[-1],
    )
    support = _candidate_from_saved(step2, "foot1")

    with PyBulletScene(paths, gui=False) as scene:
        scene.load_tower()
        scene.load_robot(base_pose)
        scene.reset_joints(q_start)
        new_support_before = scene.get_suction_pose(frames.base_end)
        old_support_before = scene.get_suction_pose(frames.l8_end)
        rebase = rebase_to_support(
            scene,
            joints=q_start,
            old_support_name="foot2/l8_end",
            new_support_name="foot1/base_end",
            new_support_frame=frames.base_end,
            target_support_pose=new_support_before,
        )
        moving_pose = scene.get_suction_pose(frames.l8_end)
        moving_z = float(moving_pose.position[2])

    print("Step 2 end state restored", flush=True)
    print(f"current foot1/base_end: {new_support_before.position.tolist()}", flush=True)
    print(f"current foot2/l8_end: {moving_pose.position.tolist()}", flush=True)
    print(f"old support before rebase: foot2/l8_end", flush=True)
    print(f"new support after rebase: foot1/base_end", flush=True)
    return q_start, rebase, support, rebase.target_support_pose, moving_z, float(old_support_before.position[2])


def _segment_summary(lines: AttachLineSet, current_xyz: np.ndarray, current_z: float) -> tuple[int, float, int, float, float]:
    """根据当前脚位置识别低处段，再选择其上方最低的连续段。"""

    summaries: list[tuple[int, float, float, float]] = []
    current_segment_id = -1
    current_segment_distance = float("inf")
    current_segment_max_z = float("nan")
    for segment_id, start, end in zip(lines.segment_ids, lines.offsets[:-1], lines.offsets[1:]):
        points = lines.polyline_xyz_m[int(start):int(end)]
        min_z = float(np.min(points[:, 2]))
        max_z = float(np.max(points[:, 2]))
        distance = float(np.min(np.linalg.norm(points - current_xyz[None, :], axis=1)))
        summaries.append((int(segment_id), min_z, max_z, distance))
        if distance < current_segment_distance:
            current_segment_id = int(segment_id)
            current_segment_distance = distance
            current_segment_max_z = max_z

    above = [item for item in summaries if item[1] > current_z + 1e-6]
    if not above:
        raise RuntimeError("当前高度以上没有新的连续attach-line区域")
    above.sort(key=lambda item: (item[1], item[0]))
    next_segment_id, next_min_z, _, _ = above[0]
    return current_segment_id, current_segment_max_z, next_segment_id, next_min_z, next_min_z - current_segment_max_z


def _select_fixed_target(lines: AttachLineSet, current_xyz: np.ndarray, current_z: float, override_s: float | None) -> tuple[FixedTarget, int, float, float, float]:
    """选择障碍后第一个连续段的最低合法点，并固定其segment和s。"""

    current_segment_id, current_top_z, next_segment_id, next_min_z, gap = _segment_summary(
        lines, current_xyz, current_z
    )
    start, end = lines._slice(lines._segment_index(next_segment_id))
    segment_s = lines.polyline_s_m[start:end]
    segment_z = lines.polyline_xyz_m[start:end, 2]
    if override_s is None:
        # 附着线已经是吸盘半径和安全裕量收缩后的合法中心线；在该段中
        # 取最低z点，不再为了IK困难改换目标。
        min_z = float(np.min(segment_z))
        candidates = np.flatnonzero(np.isclose(segment_z, min_z, atol=1e-7))
        index = int(candidates[0])
        s_m = float(segment_s[index])
    else:
        s_m = float(override_s)
    sample = lines.evaluate(next_segment_id, s_m)
    target = FixedTarget(next_segment_id, sample.s_m, sample)
    print(f"current attach segment: {current_segment_id}", flush=True)
    print(f"obstacle/attach gap lower boundary: {current_top_z:.6f} m", flush=True)
    print(f"next attachable segment: {next_segment_id}", flush=True)
    print(f"next attachable segment minimum z: {next_min_z:.6f} m", flush=True)
    print(f"attach gap height: {gap:.6f} m", flush=True)
    print(f"selected fixed target: segment={target.segment_id}, s={target.s_m:.9f} m", flush=True)
    print(f"selected target xyz: {target.sample.xyz_m.tolist()}", flush=True)
    print(f"selected target normal: {target.sample.normal.tolist()}", flush=True)
    return target, current_segment_id, current_top_z, next_min_z, gap


def _yaw_values(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("yaw采样数必须大于0")
    return np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)


def _classify_collision(report: CollisionReport | None) -> tuple[str, str]:
    if report is None or report.ok:
        return "", ""
    item = report.first_item
    if item is None:
        return "UNKNOWN_COLLISION", ""
    if item.kind == "self":
        return "SELF_COLLISION", item.link_a
    if item.kind == "support_constraint":
        return "SUPPORT_DRIFT", item.link_a
    return "GOAL_TOWER_COLLISION", item.link_a


def _run_fixed_target_ik(
    *,
    scene: PyBulletScene,
    planner: OneStepPlanner,
    target: FixedTarget,
    support: CandidatePoint,
    support_pose: RigidTransform,
    moving_frame,
    support_frame,
    start_joints: np.ndarray,
    moving_start_xyz: np.ndarray,
    settings: PlannerSettings,
) -> IKSearchResult:
    """只对固定目标做yaw+multi-start IK，并保留全局best-effort结果。"""

    target_candidate = planner._candidate_from_line_sample(target.sample, "foot2", "surface2")
    moving_link_index = scene.link_index(moving_frame.link_name)
    support_link_index = scene.link_index(support_frame.link_name)
    preferred_y = scene.get_suction_pose(moving_frame).y_axis.copy()
    solver = NumericalSuctionIKSolver(
        scene,
        moving_frame,
        position_tolerance_m=settings.position_tolerance_m,
        normal_tolerance_deg=settings.normal_tolerance_deg,
        max_iterations=settings.numerical_ik_iterations,
        orientation_mode="normal_only",
        jacobian_mode="pybullet",
    )
    checker = CollisionChecker(
        scene,
        support_link_index=support_link_index,
        moving_link_index=moving_link_index,
        support_point_m=support.xyz_m,
        target_point_m=target_candidate.xyz_m,
        collision_margin_m=settings.collision_margin_m,
        allowed_contact_radius_m=settings.allowed_contact_radius_m,
        support_suction_frame=support_frame,
        support_suction_pose=support_pose,
        moving_start_point_m=moving_start_xyz,
        support_position_tolerance_m=settings.position_tolerance_m,
        support_normal_tolerance_deg=settings.normal_tolerance_deg,
    )

    rows: list[dict[str, object]] = []
    counts: Counter = Counter()
    # best-effort严格按位置误差优先；法向误差只作为位置误差相同情况下的次级排序。
    best_score = (float("inf"), float("inf"))
    best_joints = np.asarray(start_joints, dtype=float).copy()
    best_position = float("inf")
    best_normal = float("inf")
    best_yaw = 0.0
    valid_goals: list[tuple[np.ndarray, float, float, float]] = []
    warm_starts: list[np.ndarray] = []
    yaw_values = _yaw_values(max(32, settings.yaw_samples))

    def record_result(yaw: float, seed_index: int, result, reason: str, collision_link: str = "") -> None:
        nonlocal best_score, best_joints, best_position, best_normal, best_yaw
        score = (float(result.position_error_m), float(result.normal_error_deg))
        if score < best_score:
            best_score = score
            best_joints = result.joints.copy()
            best_position = float(result.position_error_m)
            best_normal = float(result.normal_error_deg)
            best_yaw = float(yaw)
        counts[reason] += 1
        rows.append({
            "segment_id": target.segment_id,
            "s_m": target.s_m,
            "x_m": target.sample.xyz_m[0],
            "y_m": target.sample.xyz_m[1],
            "z_m": target.sample.xyz_m[2],
            "yaw_rad": yaw,
            "seed_index": seed_index,
            "position_error_m": result.position_error_m,
            "normal_error_deg": result.normal_error_deg,
            "ik_success": int(result.success),
            "failure_reason": reason,
            "collision_link": collision_link,
            "iterations": result.iterations,
        })

    for yaw in yaw_values:
        scene.reset_joints(start_joints)
        target_pose = planner.pose_builder.build(
            target_candidate,
            preferred_y_reference_world=preferred_y,
            yaw_rad=float(yaw),
        )
        target_link_pose = target_pose.multiply(moving_frame.transform_link_to_suction.inverse())
        seeds = planner._build_ik_seeds(
            scene=scene,
            moving_link_index=moving_link_index,
            target_link_pose=target_link_pose,
            start_joints=start_joints,
            warm_start_joints=warm_starts,
        )
        results = solver.solve_all(target_pose, seeds)
        successful_results = [item for item in results if item.success]
        for seed_index, result in enumerate(results):
            if not result.success:
                reason = "IK_POSITION_ERROR" if result.position_error_m > settings.position_tolerance_m else (
                    "IK_NORMAL_ERROR" if result.normal_error_deg > settings.normal_tolerance_deg else "IK_FAILED"
                )
                record_result(float(yaw), seed_index, result, reason)
                continue
            if not scene.within_joint_limits(result.joints):
                record_result(float(yaw), seed_index, result, "JOINT_LIMIT_FAILED")
                continue
            scene.reset_joints(result.joints)
            report = checker.check_state(allow_goal_contact=True)
            if not report.ok:
                reason, link = _classify_collision(report)
                record_result(float(yaw), seed_index, result, reason, link)
                continue
            record_result(float(yaw), seed_index, result, "GOAL_SUCCESS")
            valid_goals.append(
                (
                    result.joints.copy(),
                    float(result.position_error_m),
                    float(result.normal_error_deg),
                    float(yaw),
                )
            )
            if len(warm_starts) < settings.warm_start_seed_count:
                warm_starts.append(result.joints.copy())

        # 明确把这一yaw的最佳结果留下，不因后续seed状态污染GUI或best-effort。
        if successful_results:
            scene.reset_joints(min(successful_results, key=lambda item: item.position_error_m).joints)
        elif rows:
            # 即使当前yaw没有收敛，也把该yaw的最佳近似解作为下一个yaw
            # 的邻近历史seed，避免每个yaw都从完全相同的状态重新开始。
            best_row = min(rows, key=lambda row: float(row["position_error_m"]))
            best_row_joints = None
            # record_result只保存诊断标量；从当前solver结果中取该yaw最优解。
            if results:
                best_row_joints = min(results, key=lambda item: item.position_error_m).joints
            if best_row_joints is not None and not any(
                np.linalg.norm(best_row_joints - old) < 1e-10 for old in warm_starts
            ):
                warm_starts.append(best_row_joints.copy())

    # 对粗采样最优yaw进行局部细化，仍然只针对同一个target。
    if rows:
        best_yaw = min(rows, key=lambda row: float(row["position_error_m"]))["yaw_rad"]
        coarse_step = 2.0 * math.pi / max(32, settings.yaw_samples)
        refine_yaws = [float(best_yaw + factor * coarse_step / 8.0) % (2.0 * math.pi) for factor in range(-4, 5)]
        for yaw in refine_yaws:
            scene.reset_joints(start_joints)
            target_pose = planner.pose_builder.build(target_candidate, preferred_y_reference_world=preferred_y, yaw_rad=yaw)
            target_link_pose = target_pose.multiply(moving_frame.transform_link_to_suction.inverse())
            seeds = planner._build_ik_seeds(
                scene=scene, moving_link_index=moving_link_index, target_link_pose=target_link_pose,
                start_joints=start_joints, warm_start_joints=warm_starts,
            )
            results = solver.solve_all(target_pose, seeds)
            for seed_index, result in enumerate(results):
                if not result.success:
                    reason = "IK_POSITION_ERROR" if result.position_error_m > settings.position_tolerance_m else (
                        "IK_NORMAL_ERROR" if result.normal_error_deg > settings.normal_tolerance_deg else "IK_FAILED"
                    )
                    record_result(yaw, seed_index, result, reason)
                    continue
                if not scene.within_joint_limits(result.joints):
                    record_result(yaw, seed_index, result, "JOINT_LIMIT_FAILED")
                    continue
                scene.reset_joints(result.joints)
                report = checker.check_state(allow_goal_contact=True)
                if not report.ok:
                    reason, link = _classify_collision(report)
                    record_result(yaw, seed_index, result, reason, link)
                    continue
                record_result(yaw, seed_index, result, "GOAL_SUCCESS")
                valid_goals.append(
                    (
                        result.joints.copy(),
                        float(result.position_error_m),
                        float(result.normal_error_deg),
                        float(yaw),
                    )
                )

    # 同一IK分支可能因多个seed重复出现，保留关节状态不同的候选。
    unique_goals: list[tuple[np.ndarray, float, float, float]] = []
    for goal in valid_goals:
        if not any(np.linalg.norm(goal[0] - old[0]) < 1e-7 for old in unique_goals):
            unique_goals.append(goal)
    return IKSearchResult(best_joints, best_position, best_normal, best_yaw, unique_goals, counts, rows)


def _write_ik_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "segment_id", "s_m", "x_m", "y_m", "z_m", "yaw_rad", "seed_index",
        "position_error_m", "normal_error_deg", "ik_success", "failure_reason",
        "collision_link", "iterations",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_trajectory(
    scene: PyBulletScene,
    checker: CollisionChecker,
    trajectory: np.ndarray,
) -> tuple[bool, int | None, CollisionReport | None]:
    for index, state in enumerate(trajectory):
        scene.reset_joints(state)
        if not scene.within_joint_limits(state):
            return False, index, None
        report = checker.check_state(
            allow_goal_contact=index == len(trajectory) - 1,
            allow_start_contact=index == 0,
        )
        if not report.ok:
            return False, index, report
    return True, None, None


def _measure_support_drift(
    scene: PyBulletScene,
    trajectory: np.ndarray,
    support_frame,
    support_pose: RigidTransform,
) -> tuple[float, float]:
    """逐状态测量固定支撑吸盘的位置和法向漂移。"""

    position_errors: list[float] = []
    orientation_errors: list[float] = []
    for state in np.asarray(trajectory, dtype=float):
        scene.reset_joints(state)
        actual = scene.get_suction_pose(support_frame)
        position_errors.append(float(np.linalg.norm(actual.position - support_pose.position)))
        orientation_errors.append(angle_between_vectors_rad(actual.z_axis, support_pose.z_axis))
    return max(position_errors, default=0.0), max(orientation_errors, default=0.0)


def _write_rrt_csv(path: Path, attempts: list[RRTAttempt]) -> None:
    fields = [
        "seed", "success", "iterations", "tree_nodes", "planning_time_s",
        "nearest_goal_distance", "first_collision_kind", "first_collision_link",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for attempt in attempts:
            writer.writerow(attempt.__dict__)


def _save_success_trajectory(
    paths: ProjectPaths,
    scene: PyBulletScene,
    trajectory: np.ndarray,
    step2: SavedTrajectory,
    support: CandidatePoint,
    target: FixedTarget,
    method: str,
    target_yaw_rad: float,
) -> None:
    base_positions, base_orientations = scene.capture_base_poses_for_trajectory(trajectory)
    saved = SavedTrajectory(
        trajectory_rad=np.asarray(trajectory, dtype=float),
        joint_names=tuple(joint.name for joint in scene.joints),
        support_xyz_m=support.xyz_m,
        support_normal=support.normal,
        support_point_id=int(step2.target_segment_id),
        support_region_id=int(step2.target_segment_id),
        target_xyz_m=target.sample.xyz_m,
        target_normal=target.sample.normal,
        target_yaw_rad=float(target_yaw_rad),
        target_segment_id=target.segment_id,
        target_s_m=target.s_m,
        start_joints_rad=trajectory[0],
        goal_joints_rad=trajectory[-1],
        vertical_progress_m=float(target.sample.xyz_m[2] - step2.target_xyz_m[2]),
        trajectory_method=method,
        support_foot_name="foot1",
        moving_foot_name="foot2",
        support_frame_name="base_end",
        moving_frame_name="l8_end",
        base_position_m=base_positions,
        base_orientation_xyzw=base_orientations,
    )
    saved.save(paths.fixed_cross_obstacle_trajectory_npz, paths.fixed_cross_obstacle_trajectory_csv)
    print(f"轨迹NPZ: {paths.fixed_cross_obstacle_trajectory_npz}（固定跨障目标成功轨迹数据）", flush=True)
    print(f"轨迹CSV: {paths.fixed_cross_obstacle_trajectory_csv}（固定跨障目标逐状态关节角）", flush=True)


def _draw_experiment(scene: PyBulletScene, target: FixedTarget, target_pose: RigidTransform, support: CandidatePoint, best_joints: np.ndarray, goal_joints: np.ndarray | None) -> None:
    scene.highlight_robot()
    scene.draw_point(support.xyz_m, [1.0, 0.1, 0.1], size=14.0)
    scene.draw_point(target.sample.xyz_m, [0.1, 1.0, 0.1], size=16.0)
    scene.draw_frame(target_pose, "fixed target", axis_length=0.18)
    scene.focus_camera_on_robot(distance=1.5, yaw=45.0, pitch=-20.0)
    scene.reset_joints(goal_joints if goal_joints is not None else best_joints)


def _play_saved(paths: ProjectPaths, *, keep_open: bool, seconds_per_state: float) -> None:
    """只播放已经保存的固定目标轨迹，不重新运行IK或RRT。"""

    saved = SavedTrajectory.load(paths.fixed_cross_obstacle_trajectory_npz)
    frames = SuctionFrameSet.load(paths.suction_config)
    if saved.base_position_m is None or saved.base_orientation_xyzw is None:
        raise ValueError("固定跨障轨迹缺少base pose序列")
    with PyBulletScene(paths, gui=True) as scene:
        scene.load_tower()
        scene.load_robot(RigidTransform(saved.base_position_m[0], saved.base_orientation_xyzw[0]))
        scene.highlight_robot()
        scene.focus_camera_on_robot(distance=1.5)
        scene.draw_point(saved.support_xyz_m, [1.0, 0.1, 0.1], size=14.0)
        scene.draw_point(saved.target_xyz_m, [0.1, 1.0, 0.1], size=16.0)
        scene.reset_joints(saved.trajectory_rad[0])
        for index, joints in enumerate(saved.trajectory_rad):
            scene.set_base_pose(RigidTransform(saved.base_position_m[index], saved.base_orientation_xyzw[index]))
            scene.reset_joints(joints)
            time.sleep(max(seconds_per_state, 0.0))
        if keep_open:
            print("固定跨障轨迹已播放，GUI保持打开，按Ctrl+C退出。", flush=True)
            try:
                while p.isConnected():
                    scene.set_base_pose(RigidTransform(saved.base_position_m[-1], saved.base_orientation_xyzw[-1]))
                    scene.reset_joints(saved.goal_joints_rad)
                    time.sleep(1.0 / 60.0)
            except KeyboardInterrupt:
                pass


def _play_best_effort(
    paths: ProjectPaths,
    *,
    keep_open: bool,
    animate: bool = False,
    seconds_per_state: float = 0.08,
) -> None:
    """只显示固定目标实验保存的best-effort构型，不重新运行IK。"""

    best_path = paths.fixed_cross_obstacle_best_ik_npz
    if not best_path.exists():
        raise FileNotFoundError(f"找不到best-effort构型：{best_path}（固定目标近似IK结果）")

    data = np.load(best_path, allow_pickle=False)
    q_best = np.asarray(data["q_best"], dtype=float)
    target_xyz = np.asarray(data["target_xyz_m"], dtype=float)
    target_normal = normalize(data["target_normal"], name="best-effort目标法向")
    step2 = SavedTrajectory.load(paths.successful_step2_trajectory_npz)
    frames = SuctionFrameSet.load(paths.suction_config)
    q_start, rebase, support, support_pose, _, _ = _restore_step2_and_rebase(paths, step2, frames)
    target_candidate = CandidatePoint(
        foot_name="foot2",
        point_id=-1,
        region_id=int(data["segment_id"].item()),
        xyz_m=target_xyz,
        normal=target_normal,
        uv_m=np.zeros(2, dtype=float),
    )
    planner = OneStepPlanner(paths=paths, settings=PlannerSettings(), gui=False)

    with PyBulletScene(paths, gui=True) as scene:
        scene.load_tower()
        scene.load_robot(rebase.base_pose)
        scene.enable_support_anchor(frames.base_end, support_pose)
        scene.reset_joints(q_start)
        target_pose = planner.pose_builder.build(
            target_candidate,
            preferred_y_reference_world=scene.get_suction_pose(frames.l8_end).y_axis,
            yaw_rad=float(data["best_yaw_rad"].item()),
        )
        scene.highlight_robot()
        scene.draw_point(support.xyz_m, [1.0, 0.1, 0.1], size=14.0)
        scene.draw_point(target_xyz, [1.0, 0.85, 0.0], size=16.0)
        scene.draw_frame(target_pose, "fixed target", axis_length=0.18)
        scene.focus_camera_on_robot(distance=1.5, yaw=45.0, pitch=-20.0)
        if animate:
            # 这里只是诊断性关节插值。best-effort构型没有通过IK和碰撞
            # 验证，因此不能把这段插值当作可执行的跨障轨迹。
            print("播放best-effort诊断插值；这不是已验证的无碰撞轨迹。", flush=True)
            state_count = max(2, int(round(1.0 / max(seconds_per_state, 1e-3))) + 1)
            for joints in np.linspace(q_start, q_best, state_count):
                scene.reset_joints(joints)
                time.sleep(max(seconds_per_state, 0.0))
        else:
            scene.reset_joints(q_best)
        print(
            "best-effort构型已显示："
            f"position error={float(data['best_position_error_m'].item()):.6f} m, "
            f"normal error={float(data['best_normal_error_deg'].item()):.3f} deg",
            flush=True,
        )
        if keep_open:
            print("GUI保持best-effort构型，按Ctrl+C退出。", flush=True)
            try:
                while p.isConnected():
                    scene.reset_joints(q_best)
                    time.sleep(1.0 / 60.0)
            except KeyboardInterrupt:
                pass


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    paths.validate_required_files()
    # 失败实验不能留下上一次成功轨迹，避免回放脚本误读旧结果。
    for stale_path in (
        paths.fixed_cross_obstacle_trajectory_npz,
        paths.fixed_cross_obstacle_trajectory_csv,
    ):
        if stale_path.exists():
            stale_path.unlink()
    step2 = SavedTrajectory.load(paths.successful_step2_trajectory_npz)
    frames = SuctionFrameSet.load(paths.suction_config)
    q_start, rebase, support, support_pose, current_z, _ = _restore_step2_and_rebase(paths, step2, frames)

    lines = AttachLineSet.load_npz(paths.attach_lines_npz("foot2"), "foot2")
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_tower()
        scene.load_robot(rebase.base_pose)
        scene.enable_support_anchor(frames.base_end, support_pose)
        scene.reset_joints(q_start)
        current_moving_pose = scene.get_suction_pose(frames.l8_end)
        target, _, _, _, _ = _select_fixed_target(lines, current_moving_pose.position, current_z, args.target_s_m)

    settings = PlannerSettings(
        yaw_samples=max(32, int(args.yaw_samples)),
        ik_random_seeds=max(16, int(args.ik_random_seeds)),
        numerical_ik_iterations=max(300, int(args.numerical_ik_iterations)),
        ik_max_iterations=max(800, int(args.ik_max_iterations)),
        warm_start_seed_count=5,
        pybullet_ik_seed_count=8,
        use_pybullet_ik_seeds=True,
        position_tolerance_m=0.005,
        normal_tolerance_deg=3.0,
        collision_margin_m=0.0,
        allowed_contact_radius_m=0.09,
        ik_orientation_mode="normal_only",
        ik_jacobian_mode="pybullet",
    )
    planner = OneStepPlanner(paths=paths, settings=settings, gui=False)
    target_candidate = planner._candidate_from_line_sample(target.sample, "foot2", "surface2")

    with PyBulletScene(paths, gui=args.gui) as scene:
        scene.load_tower()
        scene.load_robot(rebase.base_pose)
        scene.enable_support_anchor(frames.base_end, support_pose)
        scene.reset_joints(q_start)
        moving_start_xyz = scene.get_suction_pose(frames.l8_end).position.copy()
        ik_result = _run_fixed_target_ik(
            scene=scene, planner=planner, target=target, support=support,
            support_pose=support_pose, moving_frame=frames.l8_end,
            support_frame=frames.base_end, start_joints=q_start,
            moving_start_xyz=moving_start_xyz, settings=settings,
        )
        paths.fixed_cross_obstacle_ik_diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_ik_csv(paths.fixed_cross_obstacle_ik_diagnostics_csv, ik_result.rows)
        np.savez(
            paths.fixed_cross_obstacle_best_ik_npz,
            q_best=ik_result.best_joints,
            best_position_error_m=ik_result.best_position_error_m,
            best_normal_error_deg=ik_result.best_normal_error_deg,
            best_yaw_rad=ik_result.best_yaw_rad,
            segment_id=target.segment_id,
            s_m=target.s_m,
            target_xyz_m=target.sample.xyz_m,
            target_normal=target.sample.normal,
        )

        valid_goal = bool(ik_result.valid_goals)
        best_goal: np.ndarray | None = None
        goal_position_error = float("nan")
        goal_normal_error = float("nan")
        goal_yaw_rad = float("nan")
        if valid_goal:
            valid_goals = sorted(
                ik_result.valid_goals,
                key=lambda item: (item[1], item[2], np.linalg.norm(item[0] - q_start)),
            )
            best_goal, goal_position_error, goal_normal_error, goal_yaw_rad = valid_goals[0]

        target_pose = planner.pose_builder.build(
            target_candidate,
            preferred_y_reference_world=scene.get_suction_pose(frames.l8_end).y_axis,
            yaw_rad=ik_result.best_yaw_rad,
        )
        if args.gui:
            _draw_experiment(scene, target, target_pose, support, ik_result.best_joints, best_goal)

        support_drift_position, support_drift_orientation = _measure_support_drift(
            scene,
            np.asarray([q_start, ik_result.best_joints]),
            frames.base_end,
            support_pose,
        )

        print("========== Fixed cross-obstacle target experiment ==========")
        print("Step 3 support: foot1/base_end")
        print("Step 3 moving foot: foot2/l8_end")
        print(f"当前foot2高度: {current_z:.6f} m")
        print(f"指定目标segment/s: {target.segment_id}/{target.s_m:.9f}")
        print(f"指定目标xyz: {target.sample.xyz_m.tolist()}")
        print(f"指定目标z: {target.sample.xyz_m[2]:.6f} m")
        print(f"需要上升: {target.sample.xyz_m[2] - current_z:.6f} m")
        print(f"IK best position error: {ik_result.best_position_error_m:.9f} m")
        print(f"IK best normal error: {ik_result.best_normal_error_deg:.6f} deg")
        print(f"IK best yaw: {ik_result.best_yaw_rad:.6f} rad")
        print(f"合法q_goal是否存在: {'是' if valid_goal else '否'}")
        print(f"Support max position drift: {support_drift_position:.12e} m")
        print(f"Support max orientation drift: {support_drift_orientation:.12e} rad")
        print(f"IK诊断CSV: {paths.fixed_cross_obstacle_ik_diagnostics_csv}（固定目标IK搜索诊断）")
        print(f"best-effort NPZ: {paths.fixed_cross_obstacle_best_ik_npz}（位置误差最小的近似构型）")
        if not valid_goal:
            print("最终结论: A（指定跨障目标本身没有找到合法终点构型）")
            for key, value in sorted(ik_result.counts.items()):
                print(f"  {key}: {value}")
            if args.keep_open and args.gui:
                try:
                    while p.isConnected():
                        scene.reset_joints(ik_result.best_joints)
                        time.sleep(1.0 / 60.0)
                except KeyboardInterrupt:
                    pass
            return

        support_link_index = scene.link_index(frames.base_end.link_name)
        moving_link_index = scene.link_index(frames.l8_end.link_name)
        checker = CollisionChecker(
            scene,
            support_link_index=support_link_index,
            moving_link_index=moving_link_index,
            support_point_m=support.xyz_m,
            target_point_m=target.sample.xyz_m,
            collision_margin_m=settings.collision_margin_m,
            allowed_contact_radius_m=settings.allowed_contact_radius_m,
            support_suction_frame=frames.base_end,
            support_suction_pose=support_pose,
            moving_start_point_m=moving_start_xyz,
            support_position_tolerance_m=settings.position_tolerance_m,
            support_normal_tolerance_deg=settings.normal_tolerance_deg,
        )
        straight = planner._interpolate_joints(q_start, best_goal)
        straight_ok, straight_index, straight_report = _validate_trajectory(scene, checker, straight)
        straight_link = straight_report.first_item.link_a if straight_report and straight_report.first_item else ""
        print(f"Straight: {'成功' if straight_ok else '失败'}", flush=True)
        print(f"Straight first collision link: {straight_link or '无'}", flush=True)
        if straight_ok:
            _save_success_trajectory(
                paths, scene, straight, step2, support, target, "STRAIGHT", goal_yaw_rad
            )
            if args.gui:
                scene.reset_joints(straight[0])
                scene.play_joint_trajectory(straight, seconds_per_state=args.seconds_per_state, repeats=1, step_simulation=False)
            print("最终结论: B（固定目标q_goal存在，Straight成功）")
            return

        attempts: list[RRTAttempt] = []
        rrt_path: list[np.ndarray] | None = None
        scale = np.maximum(scene.joint_upper_limits() - scene.joint_lower_limits(), 1e-6)
        for run_index in range(max(1, int(args.rrt_runs))):
            seed = int(args.rrt_seed_base) + run_index
            first_collision: list[tuple[str, str]] = []
            nearest_goal = [float("inf")]

            def state_valid(state: np.ndarray) -> bool:
                value = np.asarray(state, dtype=float)
                scene.reset_joints(value)
                is_goal = np.linalg.norm(value - best_goal) < 1e-8
                is_start = np.linalg.norm(value - q_start) < 1e-8
                report = checker.check_state(allow_goal_contact=is_goal, allow_start_contact=is_start)
                if report.ok:
                    nearest_goal[0] = min(
                        nearest_goal[0],
                        float(np.linalg.norm((value - best_goal) / scale)),
                    )
                if not report.ok and report.first_item is not None and not first_collision:
                    first_collision.append((report.first_item.kind, report.first_item.link_a))
                return report.ok

            def segment_valid(start: np.ndarray, goal: np.ndarray) -> bool:
                for state in planner._interpolate_segment(start, goal, 0.06):
                    if not state_valid(state):
                        return False
                return True

            started = time.perf_counter()
            rrt = RRTConnect(
                scene.joint_lower_limits(), scene.joint_upper_limits(),
                step_size_rad=0.18, max_iterations=max(1000, int(args.rrt_max_iterations)),
                goal_bias=0.2, edge_resolution_rad=0.06, random_seed=seed,
            )
            path = rrt.plan(q_start, best_goal, is_state_valid=state_valid, is_segment_valid=segment_valid)
            elapsed = time.perf_counter() - started
            stats = rrt.last_stats
            if path is not None:
                candidate = planner._densify_path(path, 0.04)
                ok, _, report = _validate_trajectory(scene, checker, candidate)
                if ok:
                    rrt_path = [state.copy() for state in candidate]
                    success = True
                else:
                    success = False
                    if report and report.first_item and not first_collision:
                        first_collision.append((report.first_item.kind, report.first_item.link_a))
            else:
                success = False
            kind, link = first_collision[0] if first_collision else ("", "")
            attempts.append(RRTAttempt(seed, success, stats.iterations, stats.tree_nodes, elapsed, nearest_goal[0], link, kind))
            print(f"RRT seed={seed}: {'成功' if success else '失败'}，iterations={stats.iterations}，nodes={stats.tree_nodes}，time={elapsed:.3f}s", flush=True)
            if success:
                break

        _write_rrt_csv(paths.fixed_cross_obstacle_rrt_diagnostics_csv, attempts)
        if rrt_path is not None:
            trajectory = np.asarray(rrt_path, dtype=float)
            _save_success_trajectory(
                paths, scene, trajectory, step2, support, target, "RRT_CONNECT", goal_yaw_rad
            )
            drift_position, drift_orientation = _measure_support_drift(
                scene, trajectory, frames.base_end, support_pose
            )
            if args.gui:
                scene.reset_joints(trajectory[0])
                scene.play_joint_trajectory(trajectory, seconds_per_state=args.seconds_per_state, repeats=1, step_simulation=False)
            print(f"RRT独立运行次数: {len(attempts)}")
            print(f"RRT成功次数: {sum(item.success for item in attempts)}")
            print(f"Support max position drift: {drift_position:.12e} m")
            print(f"Support max orientation drift: {drift_orientation:.12e} rad")
            print("最终结论: B（q_goal存在，Straight失败，RRT成功）")
        else:
            print(f"RRT独立运行次数: {len(attempts)}")
            print("RRT成功次数: 0")
            print("Support max position drift: 轨迹未生成")
            print("Support max orientation drift: 轨迹未生成")
            print("最终结论: C（q_goal存在，但所有RRT均失败）")
        _write_rrt_csv(paths.fixed_cross_obstacle_rrt_diagnostics_csv, attempts)
        if args.gui and args.keep_open:
            try:
                while p.isConnected():
                    scene.reset_joints((rrt_path[-1] if rrt_path else best_goal))
                    time.sleep(1.0 / 60.0)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
