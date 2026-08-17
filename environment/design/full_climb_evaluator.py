"""粗粒度整塔接触图前筛和首个关键边验证。"""

from __future__ import annotations

import json
from pathlib import Path

from environment.design.contact_graph import coarse_contact_nodes, first_upward_node, graph_summary
from environment.design.evaluator import DesignEvaluationSettings, MorphologyTaskEvaluator
from environment.design.morphology import MorphologyModel
from environment.design.task_suite import TaskSpec
from environment.paths import ProjectPaths
from environment.attach_lines import AttachLineSet


def evaluate_first_climb_edges(
    model: MorphologyModel,
    tasks: tuple[TaskSpec, ...],
    *,
    paths: ProjectPaths | None = None,
    run_trajectory: bool = True,
    output_path: Path | None = None,
) -> dict[str, object]:
    """验证从当前Step 2状态向上第一条边，失败即记录首个断开区域。

    这是整塔contact-state graph的第一阶段前筛。它不会把一条局部成功
    边伪装成完整48 m路线；只有所有边和后续状态都验证后才会生成整塔路线。
    """

    paths = paths or ProjectPaths.from_repo_root()
    evaluator = MorphologyTaskEvaluator(
        model,
        settings=DesignEvaluationSettings(
            seed_count=10,
            yaw_samples=1,
            local_max_nfev=150,
            normal_max_nfev=400,
            run_collision=True,
            run_trajectory=run_trajectory,
            rrt_max_iterations=800,
            rrt_seed_count=2,
            random_seed=20260815,
        ),
    )
    edge_rows = []
    for task in tasks:
        lines = AttachLineSet.load_npz(
            paths.attach_lines_for_surface_npz(task.target_surface),
            expected_surface_name=task.target_surface,
        )
        nodes = coarse_contact_nodes(lines, spacing_m=0.5, min_z_m=task.moving_start_z)
        first = first_upward_node(nodes, current_z_m=task.moving_start_z, min_progress_m=0.02)
        if first is None:
            edge_rows.append({"task_id": task.task_id, "status": "NO_UPWARD_NODE"})
            continue
        task_for_edge = task
        result = evaluator.evaluate_task(task_for_edge, target_limit=1, collision=True, trajectory=run_trajectory)
        target = result.target_results[0] if result.target_results else None
        edge_rows.append(
            {
                "task_id": task.task_id,
                "status": "SUCCESS" if result.success else "FAILED",
                "failure_type": result.failure_type,
                "first_target_z_m": float(first.sample.xyz_m[2]),
                "target_z_m": float(target.sample.xyz_m[2]) if target else None,
                "target_position_error_m": float(target.normal_best.position_error_m) if target else None,
                "target_normal_error_deg": float(target.normal_best.normal_error_deg) if target else None,
                "goal_valid": bool(target.goal_valid) if target else False,
                "straight_success": bool(target.straight_success) if target else False,
                "rrt_success": bool(target.rrt_success) if target else False,
                "minimum_clearance_m": target.minimum_clearance_m if target else None,
                "critical_link": (target.goal_collision.critical_link if target and target.goal_collision else None),
                "collision_kind": (target.goal_collision.kind if target and target.goal_collision else None),
                "critical_position_m": (target.goal_collision.critical_position_m.tolist() if target and target.goal_collision and target.goal_collision.critical_position_m is not None else None),
            }
        )
    successful_heights = [row["target_z_m"] for row in edge_rows if row.get("status") == "SUCCESS" and row.get("target_z_m") is not None]
    failed = next((row for row in edge_rows if row.get("status") == "FAILED"), None)
    report = {
        "design_name": model.spec.name,
        "initial_height_m": max(task.moving_start_z for task in tasks),
        "first_edge_success_count": len(successful_heights),
        "task_count": len(edge_rows),
        "maximum_verified_height_m": max(successful_heights, default=max(task.moving_start_z for task in tasks)),
        "first_disconnected_region": failed,
        "edges": edge_rows,
        "whole_tower_complete": False,
        "note": "当前阶段只验证Step 2之后的首个contact-state edge，未伪造完整整塔路线。",
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
