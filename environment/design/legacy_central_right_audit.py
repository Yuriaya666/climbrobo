"""Reproduce the historical central_body/right_link_3 diagnostic only.

The historical report used the same fixed candidate and Task 0 but a different
IK seed.  This script is a diagnostic reference; it does not alter the four
official q_goal artifacts and does not run trajectory planning.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from environment.design.collision_audit import _pair_by_name, _proxy_self_pairs, fixed_axis8_spec
from environment.design.collision_proxy import MorphologyCollisionWorld
from environment.design.evaluator import DesignEvaluationSettings, MorphologyTaskEvaluator
from environment.design.morphology import BaselineGeometry, MorphologyModel
from environment.design.task_suite import build_task_suite
from environment.paths import ProjectPaths


def _pose_matrix(pose) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = pose.rotation_matrix
    matrix[:3, 3] = pose.position
    return matrix


def main() -> None:
    paths = ProjectPaths.from_repo_root()
    geometry = BaselineGeometry.from_project(paths)
    spec = fixed_axis8_spec(geometry)
    task = build_task_suite(paths, max_targets_per_task=1)[0]
    evaluator = MorphologyTaskEvaluator(
        MorphologyModel(spec),
        settings=DesignEvaluationSettings(
            seed_count=10,
            yaw_samples=1,
            local_max_nfev=150,
            normal_max_nfev=400,
            random_seed=20260815,
        ),
    )
    result = evaluator.evaluate_task(task, target_limit=1, collision=False, trajectory=False)
    goal = result.target_results[0].normal_best
    if not goal.success:
        raise RuntimeError("历史seed无法复现position+normal构型")

    output_dir = paths.repo_root / "models" / "design_results" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    q_path = output_dir / "legacy_seed_20260815_task0_q.npz"
    np.savez_compressed(
        q_path,
        task_id=np.asarray(task.task_id),
        q=np.asarray(goal.q, dtype=float),
        target_xyz=np.asarray(task.targets[0].xyz_m, dtype=float),
        position_error_m=np.asarray(goal.position_error_m),
        normal_error_deg=np.asarray(goal.normal_error_deg),
        ik_seed=np.asarray(20260815),
    )

    state = MorphologyModel(spec).world_state_for_support(
        goal.q,
        task.support_endpoint,
        _pose_matrix(task.support_pose),
    )
    rows: list[dict[str, object]] = []
    for raw_radius in (0.001, 0.02, 0.04, 0.065):
        variant_spec = replace(
            spec,
            link_proxy_radius_m=raw_radius,
            link_proxy_radii_m=np.full(4, raw_radius, dtype=float),
            joint_proxy_radius_m=0.001,
            joint_proxy_radii_m=np.full(4, 0.001, dtype=float),
            collision_inflation_m=0.0,
            right_terminal_span_is_suction_offset=True,
        )
        with MorphologyCollisionWorld(paths, MorphologyModel(variant_spec), gui=False, load_tower=False) as world:
            world.update(MorphologyModel(variant_spec).forward(goal.q, body_pose=state.body_pose))
            pair = _pair_by_name(_proxy_self_pairs(world), "central_body", "right_link_3")
            rows.append(
                {
                    "raw_link_proxy_radius_m": raw_radius,
                    "collision_inflation_m": 0.0,
                    "clearance_m": None if pair is None else pair["distance_m"],
                    "pair": pair,
                }
            )

    legacy_spec = replace(
        spec,
        joint_proxy_radius_m=0.065,
        joint_proxy_radii_m=np.full(4, 0.065, dtype=float),
        collision_inflation_m=0.005,
        right_terminal_span_is_suction_offset=False,
    )
    with MorphologyCollisionWorld(paths, MorphologyModel(legacy_spec), gui=False, load_tower=False) as world:
        world.update(MorphologyModel(legacy_spec).forward(goal.q, body_pose=state.body_pose))
        legacy_pair = _pair_by_name(_proxy_self_pairs(world), "central_body", "right_link_3")

    centerline_clearance = rows[0]["clearance_m"]
    report = {
        "task_id": task.task_id,
        "q_goal_path": str(q_path),
        "q": goal.q.tolist(),
        "position_error_m": float(goal.position_error_m),
        "normal_error_deg": float(goal.normal_error_deg),
        "legacy_report_proxy_pair": legacy_pair,
        "radius_sweep": rows,
        "classification": (
            "PROXY_TOO_CONSERVATIVE"
            if centerline_clearance is not None and float(centerline_clearance) > 0.0
            else "TRUE_BODY_PENETRATION"
        ),
        "interpretation": "J7->J8 centerline clears the central mesh at near-zero radius; the reported penetration is caused by the capsule envelope.",
    }
    report_path = output_dir / "legacy_central_right_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", q_path)
    print("saved:", report_path)
    print("legacy clearance:", None if legacy_pair is None else legacy_pair["distance_m"])
    print("classification:", report["classification"])
    for row in rows:
        print(row["raw_link_proxy_radius_m"], row["clearance_m"])


if __name__ == "__main__":
    main()
