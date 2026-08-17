"""两个对称的同面附着诊断实验。

本入口只执行Test A和Test B，不会自动进入第三步或后续多步规划。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environment.attach_lines import AttachLineSample, AttachLineSet
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


MINIMUM_SAME_SURFACE_CENTER_DISTANCE_M = 2.0 * 0.062


@dataclass(frozen=True)
class SameSurfaceCase:
    """一个同面测试的物理端和表面配置。"""

    name: str
    support_frame_name: str
    support_surface_name: str
    moving_frame_name: str
    target_surface_name: str
    output_npz: Path
    output_csv: Path
    diagnostics_csv: Path
    best_ik_npz: Path


def build_cases(paths: ProjectPaths) -> tuple[SameSurfaceCase, SameSurfaceCase]:
    return (
        SameSurfaceCase(
            name="Test A",
            support_frame_name="base_end",
            support_surface_name="surface1",
            moving_frame_name="l8_end",
            target_surface_name="surface1",
            output_npz=paths.same_surface_test_a_npz,
            output_csv=paths.same_surface_test_a_csv,
            diagnostics_csv=paths.same_surface_test_a_diagnostics_csv,
            best_ik_npz=paths.same_surface_test_a_best_ik_npz,
        ),
        SameSurfaceCase(
            name="Test B",
            support_frame_name="l8_end",
            support_surface_name="surface2",
            moving_frame_name="base_end",
            target_surface_name="surface2",
            output_npz=paths.same_surface_test_b_npz,
            output_csv=paths.same_surface_test_b_csv,
            diagnostics_csv=paths.same_surface_test_b_diagnostics_csv,
            best_ik_npz=paths.same_surface_test_b_best_ik_npz,
        ),
    )


def _candidate_from_saved(
    saved: SavedTrajectory,
    *,
    xyz: np.ndarray,
    normal: np.ndarray,
    surface_name: str,
) -> CandidatePoint:
    """把第二步保存的支撑或目标信息转换成显式surface语义候选点。"""

    return CandidatePoint(
        # 旧候选接口仍需要一个合法的历史标签；真正的语义由surface_name提供。
        foot_name="foot1" if surface_name == "surface1" else "foot2",
        point_id=int(saved.target_segment_id),
        region_id=int(saved.target_segment_id),
        xyz_m=np.asarray(xyz, dtype=float),
        normal=normalize(normal, name="保存轨迹中的附着面法向"),
        uv_m=np.zeros(2, dtype=float),
        surface_name=surface_name,
    )


def _restore_step2_and_rebase(
    paths: ProjectPaths,
    step2: SavedTrajectory,
    frames: SuctionFrameSet,
    case: SameSurfaceCase,
) -> tuple[np.ndarray, RigidTransform, RigidTransform, CandidatePoint, float, float]:
    """恢复同一个Step 2末状态，并把本测试支撑端设为严格锚定端。"""

    if step2.base_position_m is None or step2.base_orientation_xyzw is None:
        raise ValueError("Step 2轨迹缺少base pose序列，无法独立恢复末状态")
    q_start = np.asarray(step2.trajectory_rad[-1], dtype=float)
    base_pose = RigidTransform(
        position=step2.base_position_m[-1],
        quaternion_xyzw=step2.base_orientation_xyzw[-1],
    )

    if case.support_frame_name == "base_end":
        support_xyz = step2.target_xyz_m
        support_normal = step2.target_normal
    else:
        support_xyz = step2.support_xyz_m
        support_normal = step2.support_normal
    support = _candidate_from_saved(
        step2,
        xyz=support_xyz,
        normal=support_normal,
        surface_name=case.support_surface_name,
    )

    with PyBulletScene(paths, gui=False) as scene:
        scene.load_tower()
        scene.load_robot(base_pose)
        scene.reset_joints(q_start)
        new_support_frame = (
            frames.base_end if case.support_frame_name == "base_end" else frames.l8_end
        )
        support_before = scene.get_suction_pose(new_support_frame)
        result = rebase_to_support(
            scene,
            joints=q_start,
            old_support_name="Step 2 existing support",
            new_support_name=case.support_frame_name,
            new_support_frame=new_support_frame,
            target_support_pose=support_before,
        )
        moving_frame = (
            frames.base_end if case.moving_frame_name == "base_end" else frames.l8_end
        )
        moving_z = float(scene.get_suction_pose(moving_frame).position[2])
        return (
            q_start,
            result.base_pose,
            result.target_support_pose,
            support,
            moving_z,
            float(result.max_link_position_change_m),
        )


def _select_nearest_upward_target(
    paths: ProjectPaths,
    case: SameSurfaceCase,
    *,
    current_moving_z: float,
    support_xyz: np.ndarray,
) -> tuple[AttachLineSample, float]:
    """在指定surface中选取最近的向上目标，不比较更高目标。"""

    lines = AttachLineSet.load_npz(
        paths.attach_lines_for_surface_npz(case.target_surface_name),
        case.target_surface_name,
    )
    samples: list[AttachLineSample] = []
    for segment_id in lines.segment_ids:
        # 10 mm是现有中心线几何表示的采样间隔，用于找最近区域；
        # 不改变NPZ，也不把规划变量改成离散point_id。
        samples.extend(lines.sample_uniform(int(segment_id), 0.01))
    candidates = []
    for sample in samples:
        progress = float(sample.xyz_m[2] - current_moving_z)
        center_distance = float(np.linalg.norm(sample.xyz_m - support_xyz))
        if progress <= 1e-6:
            continue
        if center_distance < MINIMUM_SAME_SURFACE_CENTER_DISTANCE_M:
            continue
        candidates.append((progress, center_distance, sample))
    if not candidates:
        raise RuntimeError(f"{case.name}没有找到高于当前高度且不重叠的同面目标")
    candidates.sort(key=lambda item: (item[0], item[1], item[2].segment_id, item[2].s_m))
    progress, distance, sample = candidates[0]
    print(
        f"{case.name} target surface={case.target_surface_name}, "
        f"segment={sample.segment_id}, s={sample.s_m:.6f} m, "
        f"xyz={sample.xyz_m.tolist()}, z={sample.xyz_m[2]:.6f} m, "
        f"vertical progress={progress:.6f} m, center distance={distance:.6f} m",
        flush=True,
    )
    return sample, distance


def _run_case(paths: ProjectPaths, step2: SavedTrajectory, case: SameSurfaceCase) -> None:
    frames = SuctionFrameSet.load(paths.suction_config)
    q_start, base_pose, support_pose, support, moving_z, rebase_error = _restore_step2_and_rebase(
        paths, step2, frames, case
    )
    target_sample, center_distance = _select_nearest_upward_target(
        paths,
        case,
        current_moving_z=moving_z,
        support_xyz=support.xyz_m,
    )
    settings = PlannerSettings(
        yaw_samples=32,
        ik_random_seeds=10,
        numerical_ik_iterations=180,
        rrt_max_iterations=2500,
        rrt_seed_count=3,
        progress_interval=1,
        diagnostic_output_name=case.diagnostics_csv.name,
        planning_diagnostic_output_name=case.diagnostics_csv.name.replace(
            "_diagnostics.csv", "_planning_diagnostics.csv"
        ),
        same_surface_center_distance_m=MINIMUM_SAME_SURFACE_CENTER_DISTANCE_M,
    )
    planner = OneStepPlanner(paths=paths, settings=settings, gui=False)
    result = planner.plan(
        support_foot_name=case.support_frame_name,
        moving_foot_name=case.moving_frame_name,
        support_frame_name=case.support_frame_name,
        moving_frame_name=case.moving_frame_name,
        support_surface_name=case.support_surface_name,
        target_surface_name=case.target_surface_name,
        support_candidate=support,
        start_joints=q_start,
        initial_base_pose=base_pose,
        support_suction_pose_override=support_pose,
        target_samples_override=[target_sample],
        trajectory_output_npz=case.output_npz,
        trajectory_output_csv=case.output_csv,
    )
    print(format_plan_result(result), flush=True)
    if not result.success and result.best_ik_joints is not None:
        np.savez(
            case.best_ik_npz,
            q_best=np.asarray(result.best_ik_joints, dtype=float),
            xyz_best=np.asarray(result.best_ik_xyz, dtype=float),
            normal_best=np.asarray(result.best_ik_normal, dtype=float),
            yaw_rad=np.asarray(result.best_ik_yaw_rad, dtype=float),
            target_xyz_m=np.asarray(target_sample.xyz_m, dtype=float),
            target_normal=np.asarray(target_sample.normal, dtype=float),
            surface_name=np.asarray(case.target_surface_name),
            moving_frame_name=np.asarray(case.moving_frame_name),
        )
        best_error = float(result.best_ik_position_error_m)
        best_normal_error = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(result.best_ik_normal, -target_sample.normal)
                        / (
                            np.linalg.norm(result.best_ik_normal)
                            * np.linalg.norm(target_sample.normal)
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        print(
            f"{case.name} best-effort IK: position error={best_error:.6f} m, "
            f"normal error={best_normal_error:.3f} deg, yaw={result.best_ik_yaw_rad:.6f} rad, "
            f"xyz={result.best_ik_xyz.tolist()}, q={result.best_ik_joints.tolist()}",
            flush=True,
        )
    print(
        f"{case.name} summary: support={case.support_frame_name} on {case.support_surface_name}, "
        f"moving={case.moving_frame_name} to {case.target_surface_name}, "
        f"current moving z={moving_z:.6f} m, target z={target_sample.xyz_m[2]:.6f} m, "
        f"vertical progress={target_sample.xyz_m[2] - moving_z:.6f} m, "
        f"center distance={center_distance:.6f} m, rebase position discontinuity={rebase_error:.3e} m, "
        f"success={result.success}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test A/B同面附着诊断，不规划Step 4")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--case", choices=("A", "B", "both"), default="both")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    paths.validate_required_files()
    step2 = SavedTrajectory.load(paths.successful_step2_trajectory_npz)
    cases = build_cases(paths)
    selected = cases if args.case == "both" else (cases[0] if args.case == "A" else cases[1],)
    for case in selected:
        _run_case(paths, step2, case)
    print("同面Test完成；未规划Step 4。", flush=True)


if __name__ == "__main__":
    main()
