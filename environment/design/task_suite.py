"""从真实Step 2状态和连续attach lines生成机构设计任务集。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.attach_lines import AttachLineSample, AttachLineSet
from environment.design.morphology import BaselineGeometry
from environment.paths import ProjectPaths
from environment.suction_frames import SuctionFrameSet
from environment.scene import PyBulletScene
from environment.trajectory_io import SavedTrajectory
from environment.transforms import RigidTransform, normalize


@dataclass(frozen=True)
class TaskSpec:
    """一条固定支撑、移动另一吸盘的局部任务。"""

    task_id: str
    label: str
    support_endpoint: str
    support_surface: str
    moving_endpoint: str
    target_surface: str
    support_pose: RigidTransform
    support_xyz: np.ndarray
    support_normal: np.ndarray
    start_q: np.ndarray
    moving_start_xyz: np.ndarray
    moving_start_z: float
    targets: tuple[AttachLineSample, ...]
    critical: bool = False

    @property
    def same_surface(self) -> bool:
        return self.support_surface == self.target_surface


def _line_samples(lines: AttachLineSet, *, max_samples: int) -> list[AttachLineSample]:
    """按高度分层抽取真实中心线样本，保留最低点和最高点。"""

    raw: list[AttachLineSample] = []
    for segment_id in lines.segment_ids:
        length = lines.segment_length_m(int(segment_id))
        values = np.linspace(0.0, length, max(2, max_samples), dtype=float)
        raw.extend(lines.evaluate(int(segment_id), float(value)) for value in values)
    unique: dict[tuple[int, int], AttachLineSample] = {}
    for sample in raw:
        unique[(sample.segment_id, int(round(sample.s_m * 1e6)))] = sample
    return list(unique.values())


def _select_targets(
    lines: AttachLineSet,
    *,
    moving_start_z: float,
    support_xyz: np.ndarray,
    same_surface: bool,
    minimum_same_surface_distance_m: float,
    minimum_vertical_progress_m: float,
    max_targets: int,
) -> tuple[AttachLineSample, ...]:
    samples = _line_samples(lines, max_samples=3)
    valid = []
    for sample in samples:
        progress = float(sample.xyz_m[2] - moving_start_z)
        if progress <= minimum_vertical_progress_m:
            continue
        distance = float(np.linalg.norm(sample.xyz_m - support_xyz))
        if same_surface and distance < minimum_same_surface_distance_m:
            continue
        valid.append((progress, distance, sample))
    if not valid:
        return ()

    # 保留最近向上目标、临界障碍后首段以及按高度分层的代表点。
    valid.sort(key=lambda item: (item[0], item[1], item[2].segment_id, item[2].s_m))
    selected: list[AttachLineSample] = [valid[0][2]]
    critical = min(valid, key=lambda item: (item[2].segment_id, item[0]))
    selected.append(critical[2])
    if len(valid) > 2:
        indices = np.linspace(0, len(valid) - 1, max(2, max_targets), dtype=int)
        selected.extend(valid[int(index)][2] for index in indices)

    result: list[AttachLineSample] = []
    seen: set[tuple[int, int]] = set()
    for sample in selected:
        key = (sample.segment_id, int(round(sample.s_m * 1e6)))
        if key in seen:
            continue
        seen.add(key)
        result.append(sample)
    return tuple(result[:max_targets])


def build_task_suite(
    paths: ProjectPaths | None = None,
    *,
    max_targets_per_task: int = 8,
    minimum_same_surface_distance_m: float = 0.124,
    minimum_vertical_progress_m: float = 0.02,
) -> tuple[TaskSpec, ...]:
    """建立真实surface1/surface2的对称局部任务集。"""

    paths = paths or ProjectPaths.from_repo_root()
    saved = SavedTrajectory.load(paths.successful_step2_trajectory_npz)
    if saved.base_position_m is None or saved.base_orientation_xyzw is None:
        raise ValueError("Step 2轨迹缺少base位姿，无法生成设计任务集")
    frames = SuctionFrameSet.load(paths.suction_config)
    geometry = BaselineGeometry.from_project(paths)

    q_urdf = np.asarray(saved.trajectory_rad[-1], dtype=float)
    base_pose = RigidTransform(
        position=saved.base_position_m[-1],
        quaternion_xyzw=saved.base_orientation_xyzw[-1],
    )
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_robot(base_pose)
        scene.reset_joints(q_urdf)
        endpoint_poses = {
            "base_end": scene.get_suction_pose(frames.base_end),
            "l8_end": scene.get_suction_pose(frames.l8_end),
        }

    endpoint_surface = {"base_end": "surface1", "l8_end": "surface2"}
    support_normal = {
        name: pose.z_axis.copy() for name, pose in endpoint_poses.items()
    }
    start_q = geometry.baseline_joint_vector(q_urdf)
    tasks: list[TaskSpec] = []

    for support_endpoint, moving_endpoint in (
        ("base_end", "l8_end"),
        ("l8_end", "base_end"),
    ):
        support_surface = endpoint_surface[support_endpoint]
        moving_surface = endpoint_surface[moving_endpoint]
        support_pose = endpoint_poses[support_endpoint]
        moving_pose = endpoint_poses[moving_endpoint]
        for target_surface in ("surface1", "surface2"):
            lines = AttachLineSet.load_npz(
                paths.attach_lines_for_surface_npz(target_surface),
                expected_surface_name=target_surface,
            )
            targets = _select_targets(
                lines,
                moving_start_z=float(moving_pose.position[2]),
                support_xyz=support_pose.position,
                same_surface=support_surface == target_surface,
                minimum_same_surface_distance_m=minimum_same_surface_distance_m,
                minimum_vertical_progress_m=minimum_vertical_progress_m,
                max_targets=max_targets_per_task,
            )
            if not targets:
                continue
            critical = target_surface == "surface2" and abs(targets[0].xyz_m[2] - 1.1219999) < 0.01
            task_id = f"{support_surface}_to_{target_surface}_{support_endpoint}_to_{moving_endpoint}"
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    label=f"{support_surface}支撑，{moving_endpoint}到{target_surface}",
                    support_endpoint=support_endpoint,
                    support_surface=support_surface,
                    moving_endpoint=moving_endpoint,
                    target_surface=target_surface,
                    support_pose=support_pose,
                    support_xyz=support_pose.position.copy(),
                    support_normal=normalize(support_normal[support_endpoint], name="support normal"),
                    start_q=start_q.copy(),
                    moving_start_xyz=moving_pose.position.copy(),
                    moving_start_z=float(moving_pose.position[2]),
                    targets=targets,
                    critical=critical,
                )
            )

    if not tasks:
        raise RuntimeError("真实attach lines没有生成任何上行任务")
    return tuple(tasks)


def summarize_task_suite(tasks: tuple[TaskSpec, ...]) -> list[dict[str, object]]:
    """转换为CSV/JSON友好的任务摘要。"""

    rows = []
    for task in tasks:
        rows.append(
            {
                "task_id": task.task_id,
                "label": task.label,
                "support_endpoint": task.support_endpoint,
                "support_surface": task.support_surface,
                "moving_endpoint": task.moving_endpoint,
                "target_surface": task.target_surface,
                "support_z_m": float(task.support_xyz[2]),
                "moving_start_z_m": task.moving_start_z,
                "target_count": len(task.targets),
                "critical": task.critical,
                "target_min_z_m": float(min(sample.xyz_m[2] for sample in task.targets)),
                "target_max_z_m": float(max(sample.xyz_m[2] for sample in task.targets)),
            }
        )
    return rows
