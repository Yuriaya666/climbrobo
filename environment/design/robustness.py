"""最终候选的杆长和目标扰动鲁棒性/敏感性分析。"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from environment.attach_lines import AttachLineSample
from environment.design.evaluator import DesignEvaluationSettings, MorphologyTaskEvaluator
from environment.design.morphology import MorphologyModel, MorphologySpec
from environment.design.task_suite import TaskSpec


def _perturbed_task(task: TaskSpec, delta: np.ndarray) -> TaskSpec:
    samples = []
    for sample in task.targets:
        samples.append(
            AttachLineSample(
                segment_id=sample.segment_id,
                s_m=sample.s_m,
                xyz_m=sample.xyz_m + delta,
                normal=sample.normal,
                uv_m=sample.uv_m,
            )
        )
    return replace(task, targets=tuple(samples))


def run_robustness(
    model: MorphologyModel,
    tasks: tuple[TaskSpec, ...],
    *,
    output_path: Path,
    length_factors: tuple[float, ...] = (0.90, 0.95, 1.0, 1.05, 1.10),
    target_perturbations_m: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0), (0.005, 0.0, 0.0), (-0.005, 0.0, 0.0), (0.0, 0.005, 0.0), (0.0, -0.005, 0.0), (0.0, 0.0, 0.005), (0.0, 0.0, -0.005)),
) -> list[dict[str, object]]:
    rows = []
    for factor in length_factors:
        scaled = np.asarray(model.spec.link_lengths_m, dtype=float) * factor
        scaled_spec = replace(model.spec, link_lengths_m=scaled)
        scaled_model = MorphologyModel(scaled_spec)
        evaluator = MorphologyTaskEvaluator(
            scaled_model,
            settings=DesignEvaluationSettings(
                seed_count=2,
                yaw_samples=1,
                local_max_nfev=120,
                normal_max_nfev=220,
                random_seed=20260815,
            ),
        )
        for task in tasks:
            for delta_tuple in target_perturbations_m:
                delta = np.asarray(delta_tuple, dtype=float)
                result = evaluator.evaluate_task(
                    _perturbed_task(task, delta),
                    target_limit=1,
                    collision=False,
                    trajectory=False,
                )
                target = result.target_results[0] if result.target_results else None
                rows.append(
                    {
                        "design_name": scaled_model.spec.name,
                        "length_factor": factor,
                        "task_id": task.task_id,
                        "dx_m": delta[0],
                        "dy_m": delta[1],
                        "dz_m": delta[2],
                        "position_success": bool(target and target.position_best.success),
                        "normal_success": bool(target and target.normal_best.success),
                        "position_error_m": float(target.position_best.position_error_m) if target else None,
                        "normal_position_error_m": float(target.normal_best.position_error_m) if target else None,
                        "normal_error_deg": float(target.normal_best.normal_error_deg) if target else None,
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0]) if rows else ["design_name"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows

