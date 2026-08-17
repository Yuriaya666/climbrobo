"""回放同面附着实验中位置误差最小的失败IK构型。

这里显示的是从Step 2终态到best-effort构型的诊断性关节插值，
不是经过碰撞和轨迹验证的可执行轨迹。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pybullet as p

from environment.attachment_semantics import canonical_frame_name, canonical_surface_name
from environment.candidates import CandidatePoint
from environment.one_step_planner import AttachmentPoseBuilder
from environment.paths import ProjectPaths
from environment.same_surface_tests import (
    SameSurfaceCase,
    _restore_step2_and_rebase,
    build_cases,
)
from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrameSet
from environment.trajectory_io import SavedTrajectory
from environment.transforms import RigidTransform, angle_between_vectors_rad, normalize


def build_parser(description: str) -> argparse.ArgumentParser:
    """构造两个回放入口共用的命令行参数。"""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo-root", type=Path, default=None, help="仓库根目录")
    parser.add_argument(
        "--seconds-per-state",
        type=float,
        default=0.08,
        help="诊断插值每个状态的停留时间",
    )
    parser.add_argument(
        "--states",
        type=int,
        default=32,
        help="起点到best-effort构型之间的插值状态数量",
    )
    parser.add_argument(
        "--close-after",
        action="store_true",
        help="显示完成后关闭GUI；默认保持窗口打开",
    )
    return parser


def _load_best_effort(path: Path) -> dict[str, np.ndarray]:
    """读取并校验同面实验保存的最佳失败构型。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到best-effort构型：{path}（同面附着失败IK构型）")
    data = np.load(path, allow_pickle=False)
    required = {
        "q_best",
        "xyz_best",
        "normal_best",
        "yaw_rad",
        "target_xyz_m",
        "target_normal",
        "surface_name",
        "moving_frame_name",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"{path}缺少best-effort字段：{missing}")

    result = {name: data[name] for name in required}
    result["q_best"] = np.asarray(result["q_best"], dtype=float)
    result["xyz_best"] = np.asarray(result["xyz_best"], dtype=float)
    result["normal_best"] = normalize(result["normal_best"], name="best-effort实际法向")
    result["target_xyz_m"] = np.asarray(result["target_xyz_m"], dtype=float)
    result["target_normal"] = normalize(result["target_normal"], name="目标表面法向")
    for name in ("xyz_best", "target_xyz_m"):
        if result[name].shape != (3,):
            raise ValueError(f"{path}中的{name}必须是(3,)，实际为{result[name].shape}")
    if result["q_best"].shape != (8,):
        raise ValueError(f"{path}中的q_best必须是(8,)，实际为{result['q_best'].shape}")
    return result


def _target_candidate(case: SameSurfaceCase, data: dict[str, np.ndarray]) -> CandidatePoint:
    """将保存的目标恢复成带显式surface语义的候选点。"""

    surface_name = canonical_surface_name(str(data["surface_name"].item()))
    if surface_name != case.target_surface_name:
        raise ValueError(
            f"best-effort文件的surface={surface_name}与测试配置的"
            f"target_surface={case.target_surface_name}不一致"
        )
    legacy_name = "foot1" if surface_name == "surface1" else "foot2"
    return CandidatePoint(
        foot_name=legacy_name,
        point_id=-1,
        region_id=-1,
        xyz_m=data["target_xyz_m"],
        normal=data["target_normal"],
        uv_m=np.zeros(2, dtype=float),
        surface_name=surface_name,
    )


def _case_by_name(paths: ProjectPaths, case_name: str) -> SameSurfaceCase:
    cases = build_cases(paths)
    if case_name == "A":
        return cases[0]
    if case_name == "B":
        return cases[1]
    raise ValueError(f"未知同面测试：{case_name!r}")


def run_best_effort_playback(
    case_name: str,
    *,
    repo_root: Path | None,
    seconds_per_state: float,
    state_count: int,
    close_after: bool,
) -> None:
    """显示指定Test的最佳失败构型和诊断性插值。"""

    if not os.environ.get("DISPLAY"):
        raise RuntimeError("启动GUI前请先执行：export DISPLAY=:1")
    if seconds_per_state < 0.0:
        raise ValueError("seconds_per_state不能小于0")
    if state_count < 2:
        raise ValueError("states至少需要2个状态")

    paths = ProjectPaths.from_repo_root(repo_root)
    paths.validate_required_files()
    case = _case_by_name(paths, case_name)
    best = _load_best_effort(case.best_ik_npz)
    step2 = SavedTrajectory.load(paths.successful_step2_trajectory_npz)
    frames = SuctionFrameSet.load(paths.suction_config)

    expected_moving = canonical_frame_name(case.moving_frame_name)
    saved_moving = canonical_frame_name(str(best["moving_frame_name"].item()))
    if expected_moving != saved_moving:
        raise ValueError(
            f"best-effort文件的moving frame={saved_moving}，"
            f"但测试配置要求{expected_moving}"
        )

    q_start, base_pose, support_pose, support, _, rebase_position_error = (
        _restore_step2_and_rebase(paths, step2, frames, case)
    )
    moving_frame = frames.base_end if case.moving_frame_name == "base_end" else frames.l8_end
    support_frame = frames.base_end if case.support_frame_name == "base_end" else frames.l8_end
    target = _target_candidate(case, best)

    with PyBulletScene(paths, gui=True) as scene:
        scene.load_tower()
        scene.load_robot(base_pose)
        scene.enable_support_anchor(support_frame, support_pose)
        scene.reset_joints(q_start)

        moving_start_pose = scene.get_suction_pose(moving_frame)
        target_pose = AttachmentPoseBuilder().build(
            target,
            preferred_y_reference_world=moving_start_pose.y_axis,
            yaw_rad=float(best["yaw_rad"].item()),
        )

        # 先设置best-effort关节并重新读取真实FK结果，避免只相信NPZ中的缓存坐标。
        scene.reset_joints(best["q_best"])
        actual_best_pose = scene.get_suction_pose(moving_frame)
        position_error = float(np.linalg.norm(actual_best_pose.position - target.xyz_m))
        normal_error_deg = float(
            np.degrees(
                angle_between_vectors_rad(actual_best_pose.z_axis, -target.normal)
            )
        )
        support_error = scene.support_anchor_errors()

        scene.highlight_robot()
        scene.focus_camera_on_robot(distance=1.5, yaw=45.0, pitch=-20.0)
        scene.draw_point(support.xyz_m, [1.0, 0.1, 0.1], size=14.0)
        scene.draw_frame(support_pose, "support", axis_length=0.12)
        scene.draw_point(target.xyz_m, [1.0, 0.75, 0.0], size=16.0)
        scene.draw_frame(target_pose, "target", axis_length=0.18)
        scene.draw_point(actual_best_pose.position, [0.1, 1.0, 0.2], size=14.0)
        scene.draw_frame(actual_best_pose, "best actual", axis_length=0.14)
        scene.draw_polyline(
            np.vstack((target.xyz_m, actual_best_pose.position)),
            [1.0, 0.05, 0.05],
            radius=0.002,
        )

        print(f"{case.name} best-effort诊断回放", flush=True)
        print(
            f"support={case.support_frame_name} on {case.support_surface_name}; "
            f"moving={case.moving_frame_name} to {case.target_surface_name}",
            flush=True,
        )
        print(f"rebase position discontinuity: {rebase_position_error:.3e} m", flush=True)
        print(f"target xyz: {target.xyz_m.tolist()}", flush=True)
        print(f"best actual xyz: {actual_best_pose.position.tolist()}", flush=True)
        print(f"position error: {position_error:.6f} m", flush=True)
        print(f"normal error: {normal_error_deg:.3f} deg", flush=True)
        if support_error is not None:
            print(
                f"support drift at best state: position={support_error[0]:.3e} m, "
                f"normal={np.degrees(support_error[1]):.3e} deg",
                flush=True,
            )
        print("注意：下面播放的是诊断性关节插值，不是经过碰撞验证的可行轨迹。", flush=True)

        scene.reset_joints(q_start)
        for alpha in np.linspace(0.0, 1.0, state_count):
            joints = (1.0 - alpha) * q_start + alpha * best["q_best"]
            scene.reset_joints(joints)
            time.sleep(seconds_per_state)
        scene.reset_joints(best["q_best"])

        if close_after:
            return
        print("GUI保持best-effort失败构型，按Ctrl+C退出。", flush=True)
        try:
            while p.isConnected():
                scene.reset_joints(best["q_best"])
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            pass


def run_from_args(case_name: str, args: argparse.Namespace) -> None:
    """供Test A/B两个独立命令入口调用。"""

    run_best_effort_playback(
        case_name,
        repo_root=args.repo_root,
        seconds_per_state=args.seconds_per_state,
        state_count=args.states,
        close_after=args.close_after,
    )
