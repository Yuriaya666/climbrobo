from __future__ import annotations

import math
import time
import time as time_module
from dataclasses import dataclass, field, replace

import numpy as np
import pybullet as p

from environment.attach_lines import AttachLineSample, AttachLineSet
from environment.candidates import CandidatePoint, CandidateSet
from environment.collision import CollisionReport, CollisionChecker
from environment.diagnostics import (
    IKDiagnostic,
    PlanningDiagnostic,
    append_ik_diagnostics,
    write_ik_diagnostics,
    write_planning_diagnostics,
)
from environment.ik import NumericalSuctionIKSolver
from environment.paths import ProjectPaths
from environment.rrt_connect import RRTConnect
from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrameSet
from environment.trajectory_io import SavedTrajectory
from environment.transforms import (
    RigidTransform,
    angle_between_vectors_rad,
    build_frame_from_z_and_y_reference,
    normalize,
    rotate_vector_about_axis,
)


@dataclass(frozen=True)
class PlannerSettings:
    """单步规划的集中参数，后续可以从命令行或配置文件覆盖。"""

    support_rank_from_bottom: int = 1
    trajectory_steps: int = 50
    position_tolerance_m: float = 0.005
    normal_tolerance_deg: float = 3.0
    collision_margin_m: float = 0.0
    allowed_contact_radius_m: float = 0.09
    yaw_samples: int = 16
    ik_max_iterations: int = 300
    ik_residual_threshold: float = 1e-5
    numerical_ik_iterations: int = 120
    reach_margin_m: float = 0.05
    attach_search_spacing_m: float = 0.25
    attach_refinement_spacing_m: float = 0.125
    attach_refinement_window_m: float = 0.5
    ik_random_seeds: int = 3
    ik_seed_jitter_rad: float = 0.35
    diagnostic_output_name: str = "ik_search_diagnostics.csv"
    planning_diagnostic_output_name: str = "planning_diagnostics.csv"
    endpoint_only: bool = False
    skip_trajectory_planning: bool = False
    min_vertical_progress_m: float = 1e-6
    target_scan_limit: int | None = None
    progress_interval: int = 50
    rrt_max_iterations: int = 1800
    rrt_seed_count: int = 1
    rrt_seed_base: int = 20260812
    pybullet_ik_seed_count: int | None = None
    warm_start_seed_count: int = 3
    # 仅在诊断实验中可关闭PyBullet rest-pose IK，避免每个yaw再重复调用
    # 一组昂贵的外部IK；数值IK仍保留当前构型、warm start和随机seed。
    use_pybullet_ik_seeds: bool = True
    ik_orientation_mode: str = "full"
    ik_jacobian_mode: str = "finite_difference"


@dataclass(frozen=True)
class TargetAttempt:
    """一次候选目标尝试的失败原因，用于最终报告。"""

    point_id: int
    distance_m: float
    yaw_rad: float
    reason: str


@dataclass(frozen=True)
class EndpointCandidate:
    """已经通过IK、限位和终点碰撞检查的终点构型。"""

    target: CandidatePoint
    segment_id: int
    s_m: float
    yaw_rad: float
    goal_joints: np.ndarray
    position_error_m: float
    normal_error_deg: float
    vertical_progress_m: float


@dataclass(frozen=True)
class PlanResult:
    """单步规划结果。"""

    success: bool
    support: CandidatePoint
    target: CandidatePoint | None
    target_distance_m: float | None
    target_yaw_rad: float | None
    start_joints: np.ndarray
    goal_joints: np.ndarray | None
    trajectory: np.ndarray | None
    position_error_m: float | None
    normal_error_deg: float | None
    checked_targets: int
    attempts: list[TargetAttempt] = field(default_factory=list)
    failure_step: int | None = None
    collision_report: CollisionReport | None = None
    target_segment_id: int | None = None
    target_s_m: float | None = None
    vertical_progress_m: float | None = None
    diagnostics_path: str | None = None
    endpoint_candidates: list[EndpointCandidate] = field(default_factory=list)
    trajectory_method: str | None = None
    planning_time_s: float | None = None
    trajectory_npz_path: str | None = None
    trajectory_csv_path: str | None = None
    support_foot_name: str = "foot1"
    moving_foot_name: str = "foot2"
    support_frame_name: str = "base_end"
    moving_frame_name: str = "l8_end"


class AttachmentPoseBuilder:
    """把候选点位置和表面法向转换为吸盘目标位姿。"""

    def __init__(self, fallback_y_references: list[np.ndarray] | None = None) -> None:
        if fallback_y_references is None:
            fallback_y_references = [
                np.array([0.0, 0.0, 1.0], dtype=float),
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
            ]
        self.fallback_y_references = fallback_y_references

    def build(
        self,
        candidate: CandidatePoint,
        *,
        preferred_y_reference_world: np.ndarray,
        yaw_rad: float = 0.0,
    ) -> RigidTransform:
        surface_normal = normalize(candidate.normal, name="candidate normal")

        # 候选点法向由表面向外，吸附时吸盘Z轴应朝向表面，所以方向取反。
        suction_z_world = -surface_normal

        y_reference = self._usable_y_reference(
            suction_z_world=suction_z_world,
            preferred_y_reference_world=preferred_y_reference_world,
        )
        if abs(yaw_rad) > 1e-12:
            y_reference = rotate_vector_about_axis(
                y_reference,
                suction_z_world,
                yaw_rad,
            )

        rotation = build_frame_from_z_and_y_reference(
            z_axis=suction_z_world,
            y_reference=y_reference,
        )
        pose = RigidTransform.from_rotation_matrix(candidate.xyz_m, rotation)

        # 方向校验写在这里，避免调用者把表面外法向和吸盘法向弄反。
        alignment = float(np.dot(pose.z_axis, surface_normal))
        if alignment > -0.999:
            raise ValueError("吸盘Z轴没有与候选点外法向反向对齐")
        return pose

    def _usable_y_reference(
        self,
        *,
        suction_z_world: np.ndarray,
        preferred_y_reference_world: np.ndarray,
    ) -> np.ndarray:
        references = [np.asarray(preferred_y_reference_world, dtype=float)]
        references.extend(self.fallback_y_references)
        for reference in references:
            projected_norm = np.linalg.norm(
                reference - float(np.dot(reference, suction_z_world)) * suction_z_world
            )
            if projected_norm > 1e-8:
                return reference
        raise ValueError("无法找到可用的目标面内Y参考方向")


class OneStepPlanner:
    """base端固定，搜索L8端最远可达候选点的第一版规划器。"""

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        settings: PlannerSettings | None = None,
        *,
        gui: bool = False,
    ) -> None:
        self.paths = paths or ProjectPaths.from_repo_root()
        self.settings = settings or PlannerSettings()
        self.gui = gui
        self.pose_builder = AttachmentPoseBuilder()

    @staticmethod
    def _frame_by_name(suction_frames: SuctionFrameSet, name: str):
        """按配置名取得吸盘坐标系，避免规划逻辑写死某一只脚。"""

        if name == "base_end":
            return suction_frames.base_end
        if name == "l8_end":
            return suction_frames.l8_end
        raise ValueError(f"未知吸盘坐标系：{name!r}，当前仅支持base_end或l8_end")

    def plan(
        self,
        *,
        support_foot_name: str = "foot1",
        moving_foot_name: str = "foot2",
        support_frame_name: str = "base_end",
        moving_frame_name: str = "l8_end",
        support_candidate: CandidatePoint | None = None,
        start_joints: np.ndarray | None = None,
        initial_base_pose: RigidTransform | None = None,
        support_suction_pose_override: RigidTransform | None = None,
        trajectory_output_npz=None,
        trajectory_output_csv=None,
        playback: bool = False,
        keep_open: bool = False,
        playback_repeats: int = 1,
        playback_seconds_per_state: float = 0.05,
    ) -> PlanResult:
        self.paths.validate_required_files()
        # 诊断文件属于本次运行产物，开始新规划时清空旧内容。
        write_ik_diagnostics(self.paths.candidate_dir / self.settings.diagnostic_output_name, [])
        write_planning_diagnostics(self.paths.candidate_dir / self.settings.planning_diagnostic_output_name, [])
        suction_frames = SuctionFrameSet.load(self.paths.suction_config)
        support_set = CandidateSet.load_npz(
            self.paths.candidate_npz(support_foot_name), support_foot_name
        )
        moving_set = CandidateSet.load_npz(
            self.paths.candidate_npz(moving_foot_name), moving_foot_name
        )
        moving_lines = AttachLineSet.load_npz(
            self.paths.attach_lines_npz(moving_foot_name), moving_foot_name
        )

        support = support_candidate or support_set.select_from_bottom(
            self.settings.support_rank_from_bottom
        )
        support_frame = self._frame_by_name(suction_frames, support_frame_name)
        moving_frame = self._frame_by_name(suction_frames, moving_frame_name)

        # 初始支撑姿态：优先让吸盘面内Y轴接近世界Z轴，适合竖直塔面。
        support_suction_pose = support_suction_pose_override or self.pose_builder.build(
            support,
            preferred_y_reference_world=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        base_pose = initial_base_pose or support_suction_pose.multiply(
            support_frame.transform_link_to_suction.inverse()
        )

        with PyBulletScene(self.paths, gui=self.gui) as scene:
            scene.load_tower()
            scene.load_robot(base_pose)
            # 统一使用支撑吸盘锚定。第一步中它等价于固定base，第二步中
            # 它把新的支撑脚变成真正的固定运动学参考端。
            scene.enable_support_anchor(support_frame, support_suction_pose)

            actual_start_joints = (
                np.zeros(len(scene.joints), dtype=float)
                if start_joints is None
                else np.asarray(start_joints, dtype=float)
            )
            scene.reset_joints(actual_start_joints)
            p.performCollisionDetection()

            if self.gui:
                scene.highlight_robot()
                scene.focus_camera_on_robot()
                self._draw_initial_debug(
                    scene, support_frame, moving_frame, support_suction_pose, support
                )

            if self.gui:
                print("开始搜索单步规划目标，GUI中机器人会先保持初始姿态。")
            result = self._search_target(
                scene=scene,
                suction_frames=suction_frames,
                moving_set=moving_set,
                moving_lines=moving_lines,
                support=support,
                support_frame=support_frame,
                moving_frame=moving_frame,
                start_joints=actual_start_joints,
            )

            # 先用粗弧长样本定位最高可行区，再只在最高终点附近细化。
            if result.endpoint_candidates and (
                not self.settings.endpoint_only or self.settings.skip_trajectory_planning
            ):
                refined_samples = self._refinement_samples(
                    moving_lines,
                    result.endpoint_candidates[0],
                )
                if refined_samples:
                    refined_result = self._search_target(
                        scene=scene,
                        suction_frames=suction_frames,
                        moving_set=moving_set,
                        moving_lines=moving_lines,
                        support=support,
                        support_frame=support_frame,
                        moving_frame=moving_frame,
                        start_joints=actual_start_joints,
                        target_samples_override=refined_samples,
                    )
                    if refined_result.endpoint_candidates:
                        result = refined_result

            # 阶段2只负责筛选合法终点；阶段3再对这些终点做轨迹规划。
            if (
                result.endpoint_candidates
                and not self.settings.endpoint_only
                and not self.settings.skip_trajectory_planning
            ):
                # 搜索过程会留下最后一次IK试探状态，轨迹层必须从真实单步
                # 起点重新读取运动脚位置，不能使用试探后的PyBullet状态。
                scene.reset_joints(actual_start_joints)
                result = self._plan_endpoint_candidates(
                    scene=scene,
                    suction_frames=suction_frames,
                    support_frame=support_frame,
                    moving_frame=moving_frame,
                    support_suction_pose=support_suction_pose,
                    moving_start_point_m=scene.get_suction_pose(moving_frame).position.copy(),
                    result=result,
                )

            # 规划成功后立即保存轨迹，后续播放不再重新执行搜索和IK。
            if result.success and result.trajectory is not None and result.target is not None:
                base_positions_m, base_orientations_xyzw = scene.capture_base_poses_for_trajectory(
                    result.trajectory
                )
                npz_path, csv_path = self._save_successful_trajectory(
                    scene, result, support_foot_name, moving_foot_name,
                    support_frame_name, moving_frame_name,
                    base_positions_m=base_positions_m,
                    base_orientations_xyzw=base_orientations_xyzw,
                    npz_path=trajectory_output_npz or self.paths.successful_trajectory_npz,
                    csv_path=trajectory_output_csv or self.paths.successful_trajectory_csv,
                )
                result = replace(
                    result,
                    trajectory_npz_path=str(npz_path),
                    trajectory_csv_path=str(csv_path),
                )

            if self.gui and result.target is not None:
                scene.reset_joints(result.start_joints)
                scene.focus_camera_on_robot()
                self._draw_result_debug(scene, moving_frame, result)
            if self.gui and playback and result.success and result.trajectory is not None:
                print("规划成功，开始慢速播放轨迹。")
                scene.play_joint_trajectory(
                    result.trajectory,
                    seconds_per_state=playback_seconds_per_state,
                    repeats=playback_repeats,
                )
            if self.gui and keep_open:
                hold_joints = result.goal_joints if result.success else result.start_joints
                if hold_joints is not None:
                    scene.reset_joints(hold_joints)
                print("GUI保持打开并固定当前姿态，按Ctrl+C退出。")
                try:
                    while p.isConnected():
                        if hold_joints is not None:
                            scene.reset_joints(hold_joints)
                        time.sleep(1.0 / 60.0)
                except KeyboardInterrupt:
                    pass

            return replace(
                result,
                support_foot_name=support_foot_name,
                moving_foot_name=moving_foot_name,
                support_frame_name=support_frame_name,
                moving_frame_name=moving_frame_name,
            )

    def _save_successful_trajectory(
        self,
        scene: PyBulletScene,
        result: PlanResult,
        support_foot_name: str,
        moving_foot_name: str,
        support_frame_name: str,
        moving_frame_name: str,
        *,
        base_positions_m: np.ndarray | None = None,
        base_orientations_xyzw: np.ndarray | None = None,
        npz_path=None,
        csv_path=None,
    ) -> tuple[object, object]:
        """保存已经通过碰撞验证的完整关节角轨迹。"""

        if result.target is None or result.goal_joints is None or result.trajectory is None:
            raise ValueError("成功轨迹缺少目标点、终点关节或轨迹数组")
        saved = SavedTrajectory(
            trajectory_rad=np.asarray(result.trajectory, dtype=float),
            joint_names=tuple(joint.name for joint in scene.joints),
            support_xyz_m=result.support.xyz_m,
            support_normal=result.support.normal,
            support_point_id=result.support.point_id,
            support_region_id=result.support.region_id,
            target_xyz_m=result.target.xyz_m,
            target_normal=result.target.normal,
            target_yaw_rad=float(result.target_yaw_rad or 0.0),
            target_segment_id=int(result.target_segment_id or 0),
            target_s_m=float(result.target_s_m or 0.0),
            start_joints_rad=result.start_joints,
            goal_joints_rad=result.goal_joints,
            vertical_progress_m=float(result.vertical_progress_m or 0.0),
            trajectory_method=result.trajectory_method or "UNKNOWN",
            support_foot_name=support_foot_name,
            moving_foot_name=moving_foot_name,
            support_frame_name=support_frame_name,
            moving_frame_name=moving_frame_name,
            base_position_m=base_positions_m,
            base_orientation_xyzw=base_orientations_xyzw,
        )
        npz_path = npz_path or self.paths.successful_trajectory_npz
        csv_path = csv_path or self.paths.successful_trajectory_csv
        saved.save(npz_path, csv_path)
        print(
            f"成功轨迹已保存：{npz_path}（可直接播放的NPZ轨迹）",
            flush=True,
        )
        print(
            f"成功轨迹表已保存：{csv_path}（逐步关节角CSV）",
            flush=True,
        )
        return npz_path, csv_path

    def visualize_result(
        self,
        result: PlanResult,
        *,
        playback: bool = True,
        keep_open: bool = True,
        playback_repeats: int = 1,
        playback_seconds_per_state: float = 0.05,
    ) -> None:
        """
        只可视化已经规划好的确定轨迹。

        这里不做目标搜索、不做IK、不做碰撞检测，避免GUI里出现规划过程
        的中间关节状态。
        """

        if not result.success or result.trajectory is None:
            print("没有成功轨迹，跳过GUI播放。")
            return

        suction_frames = SuctionFrameSet.load(self.paths.suction_config)
        support_suction_pose = self.pose_builder.build(
            result.support,
            preferred_y_reference_world=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        base_pose = support_suction_pose.multiply(
            suction_frames.base_end.transform_link_to_suction.inverse()
        )

        with PyBulletScene(self.paths, gui=True) as scene:
            scene.load_tower()
            scene.load_robot(base_pose)
            scene.reset_joints(result.start_joints)
            scene.highlight_robot()
            scene.focus_camera_on_robot()
            self._draw_initial_debug(
                scene,
                self._frame_by_name(suction_frames, result.support_frame_name),
                self._frame_by_name(suction_frames, result.moving_frame_name),
                support_suction_pose,
                result.support,
            )
            self._draw_result_debug(
                scene,
                self._frame_by_name(suction_frames, result.moving_frame_name),
                result,
            )

            if playback:
                print("开始播放已确定的单步轨迹。")
                scene.play_joint_trajectory(
                    result.trajectory,
                    seconds_per_state=playback_seconds_per_state,
                    repeats=playback_repeats,
                    step_simulation=False,
                )

            scene.reset_joints(result.goal_joints)
            scene.focus_camera_on_robot()
            if keep_open:
                print("GUI保持终点姿态，按Ctrl+C退出。")
                try:
                    while p.isConnected():
                        scene.reset_joints(result.goal_joints)
                        time.sleep(1.0 / 60.0)
                except KeyboardInterrupt:
                    pass

    def _search_target(
        self,
        *,
        scene: PyBulletScene,
        suction_frames: SuctionFrameSet,
        moving_set: CandidateSet,
        moving_lines: AttachLineSet,
        support: CandidatePoint,
        support_frame,
        moving_frame,
        start_joints: np.ndarray,
        target_samples_override: list[AttachLineSample] | None = None,
    ) -> PlanResult:
        # 每次粗搜或细化都从同一个起始构型计算运动脚初始高度，避免
        # 上一次IK试探留下的PyBullet状态污染vertical_progress。
        scene.reset_joints(start_joints)
        moving_link_index = scene.link_index(moving_frame.link_name)
        support_link_index = scene.link_index(support_frame.link_name)
        current_moving_suction_pose = scene.get_suction_pose(moving_frame)
        preferred_target_y = current_moving_suction_pose.y_axis
        chain_upper_bound = scene.estimate_serial_chain_upper_bound(
            end_link_index=moving_link_index,
            base_suction_frame=support_frame,
            end_suction_frame=moving_frame,
        )
        max_target_distance = chain_upper_bound + self.settings.reach_margin_m
        ik_solver = NumericalSuctionIKSolver(
            scene,
            moving_frame,
            position_tolerance_m=self.settings.position_tolerance_m,
            normal_tolerance_deg=self.settings.normal_tolerance_deg,
            max_iterations=self.settings.numerical_ik_iterations,
            orientation_mode=self.settings.ik_orientation_mode,
            jacobian_mode=self.settings.ik_jacobian_mode,
        )

        attempts: list[TargetAttempt] = []
        diagnostics: list[IKDiagnostic] = []
        endpoint_candidates: list[EndpointCandidate] = []
        warm_start_joints: list[np.ndarray] = []
        checked_targets = 0
        yaw_values = self._yaw_values()
        current_moving_z = float(current_moving_suction_pose.position[2])
        target_samples = target_samples_override or self._build_target_samples(moving_lines)
        if self.settings.target_scan_limit is not None:
            target_samples = target_samples[: self.settings.target_scan_limit]
        total_targets = len(target_samples)

        # 先做一次纯几何筛选。高处诊断时附着线可能包含数百个样本，
        # 其中绝大多数距离已经超过串联链长度上界；这些样本仍会写入
        # GEOMETRICALLY_UNREACHABLE诊断，但不能进入昂贵的yaw/IK循环。
        ik_target_samples: list[AttachLineSample] = []
        for sample in target_samples:
            target_xyz = np.asarray(sample.xyz_m, dtype=float)
            vertical_progress = float(target_xyz[2] - current_moving_z)
            target_distance = float(np.linalg.norm(target_xyz - support.xyz_m))
            if (
                vertical_progress > self.settings.min_vertical_progress_m
                and target_distance <= max_target_distance
            ):
                ik_target_samples.append(sample)
        ik_target_keys = {
            (int(sample.segment_id), round(float(sample.s_m), 9))
            for sample in ik_target_samples
        }

        if self.settings.progress_interval > 0:
            print(
                f"几何筛选后进入IK的目标数：{len(ik_target_samples)}/{total_targets}，"
                f"目标z范围="
                f"[{min((float(s.xyz_m[2]) for s in ik_target_samples), default=float('nan')):.3f},"
                f" {max((float(s.xyz_m[2]) for s in ik_target_samples), default=float('nan')):.3f}] m。",
                flush=True,
            )

        if self.settings.progress_interval > 0:
            print(
                f"开始按连续附着线搜索{moving_set.foot_name}目标，共{total_targets}个(s)样本。",
                flush=True,
            )

        for sample in target_samples:
            checked_targets += 1
            target = self._candidate_from_line_sample(sample, moving_set.foot_name)
            target_distance = float(np.linalg.norm(target.xyz_m - support.xyz_m))
            vertical_progress = float(target.xyz_m[2] - current_moving_z)

            # 只接受满足本次实验最低向上进度的目标，避免搜索失败后
            # 退回当前高度或更低位置并误报成功。
            if vertical_progress <= self.settings.min_vertical_progress_m:
                diagnostics.append(
                    IKDiagnostic(
                        segment_id=sample.segment_id,
                        s_m=sample.s_m,
                        xyz_m=sample.xyz_m,
                        yaw_rad=0.0,
                        vertical_progress_m=vertical_progress,
                        position_error_m=float("inf"),
                        normal_error_deg=float("inf"),
                        failure_reason="NON_POSITIVE_VERTICAL_PROGRESS",
                    )
                )
                attempts.append(
                    TargetAttempt(
                        point_id=target.point_id,
                        distance_m=target_distance,
                        yaw_rad=0.0,
                        reason=(
                            "目标没有超过当前运动脚高度加最低向上进度："
                            f"{self.settings.min_vertical_progress_m:.3f} m"
                        ),
                    )
                )
                continue

            if (
                self.settings.progress_interval > 0
                and checked_targets % self.settings.progress_interval == 0
            ):
                print(
                    f"规划进度：已检查{checked_targets}/{total_targets}个目标点，"
                    f"当前z={target.xyz_m[2]:.3f} m，提升={vertical_progress:.3f} m。",
                    flush=True,
                )

            if target_distance > max_target_distance:
                attempts.append(
                    TargetAttempt(
                        point_id=target.point_id,
                        distance_m=target_distance,
                        yaw_rad=0.0,
                        reason=(
                            f"超过机器人串联链长度上界：{target_distance:.3f} m "
                            f"> {max_target_distance:.3f} m"
                        ),
                    )
                )
                diagnostics.append(
                    IKDiagnostic(
                        segment_id=sample.segment_id,
                        s_m=sample.s_m,
                        xyz_m=sample.xyz_m,
                        yaw_rad=0.0,
                        vertical_progress_m=vertical_progress,
                        position_error_m=float("inf"),
                        normal_error_deg=float("inf"),
                        failure_reason="GEOMETRICALLY_UNREACHABLE",
                    )
                )
                continue

            # 几何上界已在上面的预筛选阶段判断；这里保留一次诊断，
            # 但不再进入yaw和IK。
            sample_key = (int(sample.segment_id), round(float(sample.s_m), 9))
            if sample_key not in ik_target_keys:
                continue

            for yaw_rad in yaw_values:
                scene.reset_joints(start_joints)
                try:
                    target_suction_pose = self.pose_builder.build(
                        target,
                        preferred_y_reference_world=preferred_target_y,
                        yaw_rad=float(yaw_rad),
                    )
                except ValueError as exc:
                    attempts.append(
                        TargetAttempt(
                            point_id=target.point_id,
                            distance_m=target_distance,
                            yaw_rad=float(yaw_rad),
                            reason=f"目标姿态构造失败：{exc}",
                        )
                    )
                    continue

                target_link_pose = target_suction_pose.multiply(
                    moving_frame.transform_link_to_suction.inverse()
                )
                seed_joints = self._build_ik_seeds(
                    scene=scene,
                    moving_link_index=moving_link_index,
                    target_link_pose=target_link_pose,
                    start_joints=start_joints,
                    warm_start_joints=warm_start_joints,
                )
                ik_results = ik_solver.solve_all(target_suction_pose, seed_joints)
                successful_results = [item for item in ik_results if item.success]
                ik_result = min(successful_results or ik_results, key=ik_solver._score)
                goal_joints = ik_result.joints

                if not ik_result.success:
                    reason = self._ik_failure_reason(ik_result)
                    diagnostics.append(
                        IKDiagnostic(
                            segment_id=sample.segment_id,
                            s_m=sample.s_m,
                            xyz_m=sample.xyz_m,
                            yaw_rad=float(yaw_rad),
                            vertical_progress_m=vertical_progress,
                            position_error_m=ik_result.position_error_m,
                            normal_error_deg=ik_result.normal_error_deg,
                            failure_reason=reason,
                        )
                    )
                    attempts.append(
                        TargetAttempt(
                            point_id=target.point_id,
                            distance_m=target_distance,
                            yaw_rad=float(yaw_rad),
                            reason=f"{reason}：{ik_result.reason}",
                        )
                    )
                    continue

                if not scene.within_joint_limits(goal_joints):
                    reason = "JOINT_LIMIT_FAILED"
                    diagnostics.append(
                        IKDiagnostic(sample.segment_id, sample.s_m, sample.xyz_m, float(yaw_rad), vertical_progress,
                                     ik_result.position_error_m, ik_result.normal_error_deg, reason)
                    )
                    attempts.append(TargetAttempt(target.point_id, target_distance, float(yaw_rad), reason))
                    continue

                endpoint_checker = CollisionChecker(
                    scene,
                    support_link_index=support_link_index,
                    moving_link_index=moving_link_index,
                    support_point_m=support.xyz_m,
                    target_point_m=target.xyz_m,
                    collision_margin_m=self.settings.collision_margin_m,
                    allowed_contact_radius_m=self.settings.allowed_contact_radius_m,
                    support_suction_frame=support_frame,
                    support_suction_pose=scene.support_anchor.target_pose
                    if scene.support_anchor is not None
                    else self.pose_builder.build(
                        support,
                        preferred_y_reference_world=np.array([0.0, 0.0, 1.0], dtype=float),
                    ),
                    support_position_tolerance_m=self.settings.position_tolerance_m,
                    support_normal_tolerance_deg=self.settings.normal_tolerance_deg,
                )
                endpoint_success_count = 0
                endpoint_failure_reason = "GOAL_TOWER_COLLISION"
                # 同一目标保留所有通过终点碰撞的IK分支，轨迹层逐一测试。
                for successful in successful_results:
                    scene.reset_joints(successful.joints)
                    endpoint_report = endpoint_checker.check_state(allow_goal_contact=True)
                    if not endpoint_report.ok:
                        item = endpoint_report.first_item
                        endpoint_failure_reason = "SELF_COLLISION" if item and item.kind == "self" else "GOAL_TOWER_COLLISION"
                        continue
                    endpoint_success_count += 1
                    endpoint_candidates.append(
                        EndpointCandidate(
                            target=target,
                            segment_id=sample.segment_id,
                            s_m=sample.s_m,
                            yaw_rad=float(yaw_rad),
                            goal_joints=successful.joints.copy(),
                            position_error_m=successful.position_error_m,
                            normal_error_deg=successful.normal_error_deg,
                            vertical_progress_m=vertical_progress,
                        )
                    )
                    if len(warm_start_joints) < self.settings.warm_start_seed_count:
                        warm_start_joints.append(successful.joints.copy())
                if endpoint_success_count == 0:
                    diagnostics.append(
                        IKDiagnostic(sample.segment_id, sample.s_m, sample.xyz_m, float(yaw_rad), vertical_progress,
                                     ik_result.position_error_m, ik_result.normal_error_deg, endpoint_failure_reason)
                    )
                    attempts.append(TargetAttempt(target.point_id, target_distance, float(yaw_rad), endpoint_failure_reason))
                    continue
                diagnostics.append(
                    IKDiagnostic(sample.segment_id, sample.s_m, sample.xyz_m, float(yaw_rad), vertical_progress,
                                 ik_result.position_error_m, ik_result.normal_error_deg, "SUCCESS")
                )

        diagnostics_path = self.paths.candidate_dir / self.settings.diagnostic_output_name
        append_ik_diagnostics(diagnostics_path, diagnostics)
        print(f"IK搜索诊断已写入：{diagnostics_path}（连续目标搜索诊断表）", flush=True)
        endpoint_candidates.sort(key=lambda item: item.vertical_progress_m, reverse=True)
        best_endpoint = endpoint_candidates[0] if endpoint_candidates else None
        return PlanResult(
            success=bool(best_endpoint is not None and self.settings.endpoint_only),
            support=support,
            target=best_endpoint.target if best_endpoint else None,
            target_distance_m=(float(np.linalg.norm(best_endpoint.target.xyz_m - support.xyz_m)) if best_endpoint else None),
            target_yaw_rad=best_endpoint.yaw_rad if best_endpoint else None,
            start_joints=start_joints,
            goal_joints=best_endpoint.goal_joints.copy() if best_endpoint else None,
            trajectory=None,
            position_error_m=best_endpoint.position_error_m if best_endpoint else None,
            normal_error_deg=best_endpoint.normal_error_deg if best_endpoint else None,
            checked_targets=checked_targets,
            attempts=attempts,
            diagnostics_path=str(diagnostics_path),
            target_segment_id=best_endpoint.segment_id if best_endpoint else None,
            target_s_m=best_endpoint.s_m if best_endpoint else None,
            vertical_progress_m=best_endpoint.vertical_progress_m if best_endpoint else None,
            endpoint_candidates=endpoint_candidates,
        )

    def _plan_endpoint_candidates(
        self,
        *,
        scene: PyBulletScene,
        suction_frames: SuctionFrameSet,
        support_frame,
        moving_frame,
        support_suction_pose: RigidTransform,
        moving_start_point_m: np.ndarray,
        result: PlanResult,
    ) -> PlanResult:
        """阶段3：按高度从高到低，对合法终点尝试直线和RRT。"""

        support_link_index = scene.link_index(support_frame.link_name)
        moving_link_index = scene.link_index(moving_frame.link_name)
        planning_rows: list[PlanningDiagnostic] = []
        for endpoint in result.endpoint_candidates:
            planning_start = time_module.perf_counter()
            checker = self._make_collision_checker(
                scene=scene,
                support_frame=support_frame,
                support_suction_pose=support_suction_pose,
                support_link_index=support_link_index,
                moving_link_index=moving_link_index,
                support=result.support,
                target=endpoint.target,
                moving_start_point_m=moving_start_point_m,
            )
            straight = self._interpolate_joints(result.start_joints, endpoint.goal_joints)
            if self._trajectory_is_valid(scene, checker, straight):
                planning_rows.append(self._planning_row(endpoint, "SUCCESS", "STRAIGHT", planning_start))
                write_planning_diagnostics(
                    self.paths.candidate_dir / self.settings.planning_diagnostic_output_name,
                    planning_rows,
                )
                print(
                    f"最高轨迹目标使用关节直线：z={endpoint.target.xyz_m[2]:.6f} m，"
                    f"提升={endpoint.vertical_progress_m:.6f} m",
                    flush=True,
                )
                return self._result_with_trajectory(result, endpoint, straight, "STRAIGHT")

            planning_rows.append(self._planning_row(endpoint, "STRAIGHT_PATH_COLLISION", "STRAIGHT", planning_start))

            print(
                f"直线轨迹失败，尝试RRT：segment={endpoint.segment_id}, s={endpoint.s_m:.3f} m，"
                f"z={endpoint.target.xyz_m[2]:.6f} m",
                flush=True,
            )
            def state_valid(state: np.ndarray) -> bool:
                scene.reset_joints(state)
                is_goal = np.linalg.norm(np.asarray(state) - endpoint.goal_joints) < 1e-8
                is_start = np.linalg.norm(np.asarray(state) - result.start_joints) < 1e-8
                return checker.check_state(
                    allow_goal_contact=is_goal,
                    allow_start_contact=is_start,
                ).ok

            def segment_valid(start: np.ndarray, goal: np.ndarray) -> bool:
                for state in self._interpolate_segment(start, goal, 0.06):
                    if not state_valid(state):
                        return False
                return True

            for seed_offset in range(max(1, self.settings.rrt_seed_count)):
                seed = self.settings.rrt_seed_base + seed_offset
                rrt = RRTConnect(
                    scene.joint_lower_limits(),
                    scene.joint_upper_limits(),
                    step_size_rad=0.18,
                    max_iterations=self.settings.rrt_max_iterations,
                    goal_bias=0.2,
                    edge_resolution_rad=0.06,
                    random_seed=seed,
                )
                path = rrt.plan(
                    result.start_joints,
                    endpoint.goal_joints,
                    is_state_valid=state_valid,
                    is_segment_valid=segment_valid,
                )
                if path is None:
                    planning_rows.append(self._planning_row(endpoint, "RRT_FAILED", "RRT_CONNECT", planning_start))
                    print(
                        f"RRT失败：seed={seed}, segment={endpoint.segment_id}, "
                        f"s={endpoint.s_m:.3f} m，z={endpoint.target.xyz_m[2]:.6f} m",
                        flush=True,
                    )
                    continue
                trajectory = self._densify_path(path, 0.04)
                if len(trajectory) < 2:
                    planning_rows.append(self._planning_row(endpoint, "RRT_DEGENERATE_PATH", "RRT_CONNECT", planning_start))
                    continue
                if self._trajectory_is_valid(scene, checker, trajectory):
                    planning_rows.append(self._planning_row(endpoint, "SUCCESS", "RRT_CONNECT", planning_start))
                    write_planning_diagnostics(
                        self.paths.candidate_dir / self.settings.planning_diagnostic_output_name,
                        planning_rows,
                    )
                    print(
                        f"最高轨迹目标使用RRT：seed={seed}, z={endpoint.target.xyz_m[2]:.6f} m，"
                        f"提升={endpoint.vertical_progress_m:.6f} m",
                        flush=True,
                    )
                    return self._result_with_trajectory(result, endpoint, trajectory, "RRT_CONNECT")

        print("所有合法终点均未找到无碰撞轨迹。", flush=True)
        write_planning_diagnostics(
            self.paths.candidate_dir / self.settings.planning_diagnostic_output_name,
            planning_rows,
        )
        return PlanResult(
            success=False,
            support=result.support,
            target=None,
            target_distance_m=None,
            target_yaw_rad=None,
            start_joints=result.start_joints,
            goal_joints=None,
            trajectory=None,
            position_error_m=None,
            normal_error_deg=None,
            checked_targets=result.checked_targets,
            attempts=result.attempts,
            diagnostics_path=result.diagnostics_path,
            endpoint_candidates=result.endpoint_candidates,
            trajectory_method=None,
        )

    def _planning_row(self, endpoint: EndpointCandidate, reason: str, method: str, started: float) -> PlanningDiagnostic:
        return PlanningDiagnostic(
            segment_id=endpoint.segment_id,
            s_m=endpoint.s_m,
            xyz_m=endpoint.target.xyz_m,
            yaw_rad=endpoint.yaw_rad,
            vertical_progress_m=endpoint.vertical_progress_m,
            position_error_m=endpoint.position_error_m,
            normal_error_deg=endpoint.normal_error_deg,
            failure_reason=reason,
            trajectory_method=method,
            planning_time_s=time_module.perf_counter() - started,
        )

    def _result_with_trajectory(
        self,
        result: PlanResult,
        endpoint: EndpointCandidate,
        trajectory: np.ndarray,
        method: str,
    ) -> PlanResult:
        return PlanResult(
            success=True,
            support=result.support,
            target=endpoint.target,
            target_distance_m=float(np.linalg.norm(endpoint.target.xyz_m - result.support.xyz_m)),
            target_yaw_rad=endpoint.yaw_rad,
            start_joints=result.start_joints,
            goal_joints=endpoint.goal_joints.copy(),
            trajectory=trajectory,
            position_error_m=endpoint.position_error_m,
            normal_error_deg=endpoint.normal_error_deg,
            checked_targets=result.checked_targets,
            attempts=result.attempts,
            target_segment_id=endpoint.segment_id,
            target_s_m=endpoint.s_m,
            vertical_progress_m=endpoint.vertical_progress_m,
            diagnostics_path=result.diagnostics_path,
            endpoint_candidates=result.endpoint_candidates,
            trajectory_method=method,
        )

    def _make_collision_checker(
        self,
        *,
        scene: PyBulletScene,
        support_frame,
        support_suction_pose: RigidTransform,
        support_link_index: int,
        moving_link_index: int,
        support: CandidatePoint,
        target: CandidatePoint,
        moving_start_point_m: np.ndarray | None = None,
    ) -> CollisionChecker:
        return CollisionChecker(
            scene,
            support_link_index=support_link_index,
            moving_link_index=moving_link_index,
            support_point_m=support.xyz_m,
            target_point_m=target.xyz_m,
            collision_margin_m=self.settings.collision_margin_m,
            allowed_contact_radius_m=self.settings.allowed_contact_radius_m,
            support_suction_frame=support_frame,
            support_suction_pose=support_suction_pose,
            moving_start_point_m=moving_start_point_m,
            support_position_tolerance_m=self.settings.position_tolerance_m,
            support_normal_tolerance_deg=self.settings.normal_tolerance_deg,
        )

    @staticmethod
    def _interpolate_segment(start: np.ndarray, goal: np.ndarray, resolution: float) -> np.ndarray:
        distance = float(np.max(np.abs(np.asarray(goal) - np.asarray(start))))
        count = max(2, int(np.ceil(distance / resolution)) + 1)
        return np.linspace(start, goal, count)

    def _trajectory_is_valid(
        self,
        scene: PyBulletScene,
        checker: CollisionChecker,
        trajectory: np.ndarray,
    ) -> bool:
        for index, state in enumerate(trajectory):
            if not scene.within_joint_limits(state):
                return False
            scene.reset_joints(state)
            if not checker.check_state(
                allow_goal_contact=(index == len(trajectory) - 1),
                allow_start_contact=(index == 0),
            ).ok:
                return False
        return True

    @staticmethod
    def _densify_path(path: list[np.ndarray], resolution: float) -> np.ndarray:
        pieces = []
        for start, goal in zip(path[:-1], path[1:]):
            segment = OneStepPlanner._interpolate_segment(start, goal, resolution)
            if pieces:
                segment = segment[1:]
            pieces.append(segment)
        if not pieces:
            return np.asarray(path, dtype=float)
        return np.concatenate(pieces, axis=0)

    @staticmethod
    def _candidate_from_line_sample(sample: AttachLineSample, foot_name: str) -> CandidatePoint:
        """将连续样本包装成兼容旧姿态和碰撞接口的候选点。"""

        synthetic_id = -((int(sample.segment_id) + 1) * 1_000_000 + int(round(sample.s_m * 1000.0)))
        return CandidatePoint(
            foot_name=foot_name,
            point_id=synthetic_id,
            region_id=int(sample.segment_id),
            xyz_m=np.asarray(sample.xyz_m, dtype=float),
            normal=np.asarray(sample.normal, dtype=float),
            uv_m=np.asarray(sample.uv_m, dtype=float),
        )

    def _build_target_samples(self, lines: AttachLineSet) -> list[AttachLineSample]:
        samples: list[AttachLineSample] = []
        for segment_id in lines.segment_ids:
            samples.extend(lines.sample_uniform(int(segment_id), self.settings.attach_search_spacing_m))
        samples.sort(key=lambda sample: (float(sample.xyz_m[2]), float(sample.s_m)), reverse=True)
        return samples

    def _refinement_samples(
        self,
        lines: AttachLineSet,
        endpoint: EndpointCandidate,
    ) -> list[AttachLineSample]:
        """围绕当前最高终点建立局部细化样本，保持segment边界不跨越。"""

        spacing = float(self.settings.attach_refinement_spacing_m)
        window = max(float(self.settings.attach_refinement_window_m), spacing)
        length = lines.segment_length_m(endpoint.segment_id)
        start = max(0.0, endpoint.s_m - window)
        end = min(length, endpoint.s_m + window)
        values = np.arange(start, end + 0.5 * spacing, spacing, dtype=float)
        if len(values) == 0 or values[-1] < end:
            values = np.append(values, end)
        samples = [lines.evaluate(endpoint.segment_id, float(value)) for value in values]
        samples.sort(key=lambda sample: (float(sample.xyz_m[2]), float(sample.s_m)), reverse=True)
        print(
            f"局部细化：segment={endpoint.segment_id}, s范围=[{start:.3f}, {end:.3f}] m，"
            f"间隔={spacing:.3f} m，共{len(samples)}个样本。",
            flush=True,
        )
        return samples

    def _build_ik_seeds(
        self,
        *,
        scene: PyBulletScene,
        moving_link_index: int,
        target_link_pose: RigidTransform,
        start_joints: np.ndarray,
        warm_start_joints: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """生成真正不同的数值IK初值，随机部分使用固定种子保证复现。"""

        lower = scene.joint_lower_limits()
        upper = scene.joint_upper_limits()
        rng = np.random.default_rng(20260812)
        seeds = [np.asarray(start_joints, dtype=float).copy()]
        if warm_start_joints:
            seeds.extend(
                np.asarray(value, dtype=float).copy()
                for value in warm_start_joints[: self.settings.warm_start_seed_count]
            )
        for scale in (0.12, 0.35):
            seeds.append(np.clip(start_joints + rng.normal(0.0, scale, size=len(start_joints)), lower, upper))
        for _ in range(max(0, self.settings.ik_random_seeds)):
            seeds.append(rng.uniform(lower, upper))

        if not self.settings.use_pybullet_ik_seeds:
            return seeds

        # PyBullet rest pose每次变化，返回的初值也可能落入不同吸附姿态分支。
        pybullet_seed_count = self.settings.pybullet_ik_seed_count
        if pybullet_seed_count is None:
            pybullet_seed_count = 2 + max(0, self.settings.ik_random_seeds)
        for rest in seeds[: max(0, int(pybullet_seed_count))]:
            solution = scene.calculate_ik(
                moving_link_index,
                target_link_pose,
                rest_positions=rest,
                max_iterations=self.settings.ik_max_iterations,
                residual_threshold=self.settings.ik_residual_threshold,
            )
            solution = scene.normalize_revolute_solutions(solution)
            if scene.within_joint_limits(solution):
                seeds.append(solution)
        return seeds

    @staticmethod
    def _ik_failure_reason(result) -> str:
        if result.position_error_m > 0.005:
            return "IK_POSITION_ERROR"
        if result.normal_error_deg > 3.0:
            return "IK_NORMAL_ERROR"
        return "IK_FAILED"

    def _yaw_values(self) -> np.ndarray:
        if self.settings.yaw_samples <= 1:
            return np.array([0.0], dtype=float)
        return np.linspace(0.0, 2.0 * math.pi, self.settings.yaw_samples, endpoint=False)

    def _interpolate_joints(self, start_joints: np.ndarray, goal_joints: np.ndarray) -> np.ndarray:
        if self.settings.trajectory_steps < 2:
            raise ValueError("trajectory_steps至少为2")
        alphas = np.linspace(0.0, 1.0, self.settings.trajectory_steps)
        return (
            (1.0 - alphas[:, None]) * np.asarray(start_joints, dtype=float)[None, :]
            + alphas[:, None] * np.asarray(goal_joints, dtype=float)[None, :]
        )

    def _validate_trajectory(
        self,
        *,
        scene: PyBulletScene,
        trajectory: np.ndarray,
        support_link_index: int,
        moving_link_index: int,
        support: CandidatePoint,
        target: CandidatePoint,
    ) -> tuple[bool, int | None, CollisionReport | None]:
        checker = CollisionChecker(
            scene,
            support_link_index=support_link_index,
            moving_link_index=moving_link_index,
            support_point_m=support.xyz_m,
            target_point_m=target.xyz_m,
            collision_margin_m=self.settings.collision_margin_m,
            allowed_contact_radius_m=self.settings.allowed_contact_radius_m,
        )

        for step_index, positions in enumerate(trajectory):
            if not scene.within_joint_limits(positions):
                return False, step_index, None
            scene.reset_joints(positions)
            report = checker.check_state(allow_goal_contact=(step_index == len(trajectory) - 1))
            if not report.ok:
                return False, step_index, report
        return True, None, None

    def _draw_initial_debug(
        self,
        scene: PyBulletScene,
        support_frame,
        moving_frame,
        support_suction_pose: RigidTransform,
        support: CandidatePoint,
    ) -> None:
        scene.draw_point(support.xyz_m, [1.0, 0.0, 0.0], size=12.0)
        scene.draw_frame(support_suction_pose, "base target")
        scene.draw_frame(scene.get_suction_pose(support_frame), "support suction")
        scene.draw_frame(scene.get_suction_pose(moving_frame), "moving suction")

    def _draw_result_debug(
        self,
        scene: PyBulletScene,
        moving_frame,
        result: PlanResult,
    ) -> None:
        if result.target is None:
            return
        scene.draw_point(result.target.xyz_m, [0.0, 0.3, 1.0], size=12.0)
        target_pose = self.pose_builder.build(
            result.target,
            preferred_y_reference_world=np.array([0.0, 0.0, 1.0], dtype=float),
            yaw_rad=result.target_yaw_rad or 0.0,
        )
        scene.draw_frame(target_pose, f"{moving_frame.name} target")


def format_plan_result(result: PlanResult) -> str:
    """把规划结果整理成命令行友好的中文文本。"""

    lines = [
        "========== 单步规划结果 ==========",
        f"支撑点：foot1 point_id={result.support.point_id}, "
        f"region_id={result.support.region_id}, xyz_m={result.support.xyz_m.tolist()}",
        f"已检查目标点数量：{result.checked_targets}",
    ]

    if not result.success:
        lines.append("规划结果：失败")
        if result.attempts:
            lines.append("最近几次失败原因：")
            for attempt in result.attempts[-8:]:
                lines.append(
                    f"  point_id={attempt.point_id}, distance={attempt.distance_m:.4f} m, "
                    f"yaw={math.degrees(attempt.yaw_rad):.1f} deg, {attempt.reason}"
                )
        return "\n".join(lines)

    assert result.target is not None
    assert result.goal_joints is not None
    lines.extend(
        [
            "规划结果：成功",
            f"目标点：foot2 point_id={result.target.point_id}, "
            f"region_id={result.target.region_id}, xyz_m={result.target.xyz_m.tolist()}",
            f"支撑点到目标点距离：{result.target_distance_m:.4f} m",
            f"目标yaw采样角：{math.degrees(result.target_yaw_rad or 0.0):.1f} deg",
            f"末端吸盘位置误差：{result.position_error_m:.6f} m",
            f"末端吸盘法向误差：{result.normal_error_deg:.3f} deg",
            f"实际向上位移：{result.vertical_progress_m:.6f} m",
            f"目标关节角rad：{result.goal_joints.tolist()}",
            f"轨迹状态数量：{len(result.trajectory) if result.trajectory is not None else 0}",
            f"轨迹方法：{result.trajectory_method or '未执行'}",
            f"IK诊断：{result.diagnostics_path or '未生成'}",
            f"轨迹诊断：{str(result.diagnostics_path).replace('ik_search_diagnostics.csv', 'planning_diagnostics.csv') if result.diagnostics_path else '未生成'}",
            f"轨迹NPZ：{result.trajectory_npz_path or '未生成'}",
            f"轨迹CSV：{result.trajectory_csv_path or '未生成'}",
        ]
    )
    return "\n".join(lines)
