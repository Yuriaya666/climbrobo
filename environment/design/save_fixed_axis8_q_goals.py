"""Recreate and persist the four fixed finite-axis 8R q-goals.

This is intentionally a fixed-candidate diagnostic.  It does not search
link lengths or axis architectures and does not run collision or planning.
Each successful task is written immediately so a later audit can use the
exact solved configuration instead of solving IK again.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from environment.design.collision_audit import fixed_axis8_spec
from environment.design.evaluator import DesignEvaluationSettings, MorphologyTaskEvaluator
from environment.design.morphology import BaselineGeometry, MorphologyModel
from environment.design.task_suite import build_task_suite
from environment.paths import ProjectPaths


def _pose_matrix(pose) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = pose.rotation_matrix
    matrix[:3, 3] = pose.position
    return matrix


def _save_goal(
    path: Path,
    *,
    task_index: int,
    task,
    target,
    spec,
    model,
    timestamp: str,
) -> None:
    state = model.world_state_for_support(
        target.normal_best.q,
        task.support_endpoint,
        _pose_matrix(task.support_pose),
    )
    moving_pose = state.suction_pose(task.moving_endpoint)
    achieved_normal = moving_pose[:3, 2].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        task_index=np.asarray(task_index, dtype=np.int64),
        task_id=np.asarray(task.task_id),
        q=np.asarray(target.normal_best.q, dtype=float),
        target_xyz=np.asarray(target.sample.xyz_m, dtype=float),
        target_surface_normal=np.asarray(target.sample.normal, dtype=float),
        target_suction_normal=-np.asarray(target.sample.normal, dtype=float),
        achieved_xyz=np.asarray(moving_pose[:3, 3], dtype=float),
        achieved_suction_normal=achieved_normal,
        target_segment_id=np.asarray(target.sample.segment_id, dtype=np.int64),
        target_s_m=np.asarray(target.sample.s_m, dtype=float),
        yaw_rad=np.asarray(target.normal_best.yaw_rad, dtype=float),
        position_error_m=np.asarray(target.normal_best.position_error_m, dtype=float),
        normal_error_deg=np.asarray(target.normal_best.normal_error_deg, dtype=float),
        support_endpoint=np.asarray(task.support_endpoint),
        support_surface=np.asarray(task.support_surface),
        moving_endpoint=np.asarray(task.moving_endpoint),
        target_surface=np.asarray(task.target_surface),
        support_xyz=np.asarray(task.support_xyz, dtype=float),
        support_normal=np.asarray(task.support_normal, dtype=float),
        moving_start_xyz=np.asarray(task.moving_start_xyz, dtype=float),
        moving_start_z=np.asarray(task.moving_start_z, dtype=float),
        morphology_name=np.asarray(spec.name),
        topology_id=np.asarray(spec.topology_id),
        architecture_id=np.asarray("yaw_pitch_yaw_pitch"),
        link_lengths_m=np.asarray(spec.link_lengths_m, dtype=float),
        left_axes=np.asarray(spec.left_axes, dtype=float),
        right_axes=np.asarray(spec.right_axes, dtype=float),
        lower_limits=np.asarray(spec.lower_limits, dtype=float),
        upper_limits=np.asarray(spec.upper_limits, dtype=float),
        link_proxy_radius_m=np.asarray(spec.link_proxy_radius_m, dtype=float),
        joint_proxy_radius_m=np.asarray(spec.joint_proxy_radius_m, dtype=float),
        collision_inflation_m=np.asarray(spec.collision_inflation_m, dtype=float),
        timestamp_utc=np.asarray(timestamp),
        ik_seed=np.asarray(99, dtype=np.int64),
        ik_seed_count=np.asarray(10, dtype=np.int64),
        yaw_samples=np.asarray(1, dtype=np.int64),
        local_max_nfev=np.asarray(150, dtype=np.int64),
        normal_max_nfev=np.asarray(400, dtype=np.int64),
    )


def save_q_goals(paths: ProjectPaths, output_dir: Path) -> list[dict[str, object]]:
    geometry = BaselineGeometry.from_project(paths)
    spec = fixed_axis8_spec(geometry)
    model = MorphologyModel(spec)
    settings = DesignEvaluationSettings(
        seed_count=10,
        yaw_samples=1,
        local_max_nfev=150,
        normal_max_nfev=400,
        random_seed=99,
        run_collision=False,
        run_trajectory=False,
    )
    evaluator = MorphologyTaskEvaluator(model, settings=settings)
    tasks = build_task_suite(paths, max_targets_per_task=1)
    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(tasks):
        result = evaluator.evaluate_task(task, target_limit=1, collision=False, trajectory=False)
        if not result.target_results:
            raise RuntimeError(f"{task.task_id}没有目标结果")
        target = result.target_results[0]
        if not target.normal_best.success:
            raise RuntimeError(
                f"{task.task_id}未满足正式position+normal容差: "
                f"position={target.normal_best.position_error_m:.9g} m, "
                f"normal={target.normal_best.normal_error_deg:.9g} deg"
            )
        path = output_dir / f"optimized_8r_axis8_task_{task_index}.npz"
        _save_goal(
            path,
            task_index=task_index,
            task=task,
            target=target,
            spec=spec,
            model=model,
            timestamp=timestamp,
        )
        rows.append(
            {
                "task_index": task_index,
                "task_id": task.task_id,
                "path": str(path),
                "position_error_m": float(target.normal_best.position_error_m),
                "normal_error_deg": float(target.normal_best.normal_error_deg),
                "target_segment_id": int(target.sample.segment_id),
                "target_s_m": float(target.sample.s_m),
                "target_xyz": target.sample.xyz_m.tolist(),
                "q": target.normal_best.q.tolist(),
            }
        )
        print(
            f"saved task {task_index}: {task.task_id} -> {path} "
            f"position={target.normal_best.position_error_m:.9g} m "
            f"normal={target.normal_best.normal_error_deg:.9g} deg"
        )

    manifest = output_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_index", "task_id", "path", "position_error_m",
                "normal_error_deg", "target_segment_id", "target_s_m",
                "target_xyz", "q",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    # Keep an aggregate artifact for tools that consume one NPZ, while the
    # per-task files above remain the authoritative immediately-saved goals.
    np.savez_compressed(
        output_dir / "q_goals.npz",
        task_ids=np.asarray([str(row["task_id"]) for row in rows]),
        q_goals=np.asarray([row["q"] for row in rows], dtype=float),
        timestamp_utc=np.asarray(timestamp),
        morphology_name=np.asarray(spec.name),
        topology_id=np.asarray(spec.topology_id),
        link_lengths_m=np.asarray(spec.link_lengths_m, dtype=float),
        left_axes=np.asarray(spec.left_axes, dtype=float),
        right_axes=np.asarray(spec.right_axes, dtype=float),
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="q_goal输出目录，默认models/design_results/q_goals",
    )
    args = parser.parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    output_dir = args.output_dir or paths.repo_root / "models" / "design_results" / "q_goals"
    rows = save_q_goals(paths, output_dir)
    print(f"saved {len(rows)} q_goals to {output_dir}")


if __name__ == "__main__":
    main()
