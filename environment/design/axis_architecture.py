"""第二阶段有限离散关节轴架构诊断。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from environment.design.evaluator import DesignEvaluationSettings, MorphologyTaskEvaluator
from environment.design.length_optimizer import LengthSearchConfig, length_bounds
from environment.design.morphology import MorphologyModel, MorphologySpec, with_axis_architecture
from environment.design.task_suite import TaskSpec


def _axis(name: str, sign: float = 1.0) -> np.ndarray:
    values = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    return sign * np.asarray(values[name], dtype=float)


FINITE_AXIS_ARCHITECTURES: tuple[tuple[str, np.ndarray], ...] = (
    (
        "urdf_baseline",
        np.asarray([_axis("x", -1), _axis("z", 1), _axis("x", -1), _axis("z", -1)]),
    ),
    (
        "pitch_yaw_pitch_yaw",
        np.asarray([_axis("x", -1), _axis("y", 1), _axis("x", -1), _axis("y", -1)]),
    ),
    (
        "pitch_yaw_yaw_yaw",
        np.asarray([_axis("x", -1), _axis("y", 1), _axis("y", -1), _axis("y", -1)]),
    ),
    (
        "yaw_pitch_yaw_pitch",
        np.asarray([_axis("z", -1), _axis("x", 1), _axis("z", -1), _axis("x", 1)]),
    ),
    (
        "yaw_pitch_pitch_yaw",
        np.asarray([_axis("z", -1), _axis("x", 1), _axis("x", -1), _axis("z", 1)]),
    ),
    (
        "pitch_roll_pitch_roll",
        np.asarray([_axis("x", -1), _axis("z", 1), _axis("y", -1), _axis("z", -1)]),
    ),
)


@dataclass(frozen=True)
class AxisArchitectureResult:
    architecture_id: str
    design_name: str
    task_count: int
    normal_success_count: int
    position_success_count: int
    worst_position_error_m: float
    worst_normal_error_deg: float
    worst_normal_position_error_m: float
    task_failure_types: dict[str, str]


def optimize_axis_lengths(
    base_spec: MorphologySpec,
    tasks: tuple[TaskSpec, ...],
    *,
    architecture_id: str,
    axes: np.ndarray,
    output_path: Path | None = None,
    seed: int = 20260815,
    maxiter: int = 2,
    popsize: int = 3,
    seed_count: int = 2,
) -> tuple[MorphologySpec, dict[str, object]]:
    """在一个有限轴架构内继续优化对称杆长，优先满足position+normal。"""

    bounds = length_bounds(base_spec, LengthSearchConfig())
    cache: dict[tuple[float, ...], dict[str, object]] = {}

    def evaluate(lengths: np.ndarray) -> dict[str, object]:
        key = tuple(np.round(np.asarray(lengths, dtype=float), 8))
        if key in cache:
            return cache[key]
        spec = with_axis_architecture(
            MorphologySpec(
                **{
                    **base_spec.__dict__,
                    "link_lengths_m": np.asarray(lengths, dtype=float),
                }
            ),
            axes,
            architecture_id=architecture_id,
        )
        evaluator = MorphologyTaskEvaluator(
            MorphologyModel(spec),
            settings=DesignEvaluationSettings(
                seed_count=seed_count,
                yaw_samples=1,
                local_max_nfev=130,
                normal_max_nfev=240,
                random_seed=seed,
            ),
        )
        success = 0
        residuals = []
        position_errors = []
        normal_errors = []
        failures = {}
        for task in tasks:
            result = evaluator.evaluate_task(task, target_limit=1, collision=False, trajectory=False)
            if not result.target_results:
                failures[task.task_id] = "NO_TARGET"
                continue
            target = result.target_results[0]
            if target.normal_best.success:
                success += 1
            else:
                failures[task.task_id] = target.normal_best.failure_type or "NORMAL_WORKSPACE"
            residuals.append(target.normal_best.position_error_m + 0.05 * np.deg2rad(target.normal_best.normal_error_deg))
            position_errors.append(target.position_best.position_error_m)
            normal_errors.append(target.normal_best.normal_error_deg)
        metrics = {
            "normal_success_count": success,
            "task_count": len(tasks),
            "worst_goal_residual_m": max(residuals, default=float("inf")),
            "worst_position_error_m": max(position_errors, default=float("inf")),
            "worst_normal_error_deg": max(normal_errors, default=float("inf")),
            "task_failure_types": failures,
        }
        cache[key] = metrics
        return metrics

    def objective(lengths: np.ndarray) -> float:
        metrics = evaluate(lengths)
        return (int(metrics["task_count"]) - int(metrics["normal_success_count"])) * 1000.0 + float(metrics["worst_goal_residual_m"])

    result = differential_evolution(
        objective,
        bounds=list(bounds),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        updating="immediate",
        workers=1,
    )
    lengths = np.asarray(result.x, dtype=float)
    spec = with_axis_architecture(
        MorphologySpec(**{**base_spec.__dict__, "link_lengths_m": lengths}),
        axes,
        architecture_id=architecture_id,
    )
    metrics = evaluate(lengths)
    metrics = {**metrics, "architecture_id": architecture_id, "lengths_m": lengths.tolist(), "de_nfev": int(result.nfev), "de_nit": int(result.nit)}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(metrics))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)
    return spec, metrics


def evaluate_axis_architectures(
    base_spec: MorphologySpec,
    tasks: tuple[TaskSpec, ...],
    *,
    output_path: Path | None = None,
    seed_count: int = 3,
    local_max_nfev: int = 140,
    normal_max_nfev: int = 260,
) -> tuple[AxisArchitectureResult, ...]:
    results: list[AxisArchitectureResult] = []
    for architecture_id, axes in FINITE_AXIS_ARCHITECTURES:
        spec = with_axis_architecture(base_spec, axes, architecture_id=architecture_id)
        evaluator = MorphologyTaskEvaluator(
            MorphologyModel(spec),
            settings=DesignEvaluationSettings(
                seed_count=seed_count,
                yaw_samples=1,
                local_max_nfev=local_max_nfev,
                normal_max_nfev=normal_max_nfev,
                random_seed=20260815,
            ),
        )
        position_errors: list[float] = []
        normal_errors: list[float] = []
        normal_position_errors: list[float] = []
        failures: dict[str, str] = {}
        position_success = 0
        normal_success = 0
        for task in tasks:
            evaluation = evaluator.evaluate_task(task, target_limit=1, collision=False, trajectory=False)
            if not evaluation.target_results:
                failures[task.task_id] = "NO_TARGET"
                continue
            target = evaluation.target_results[0]
            position_errors.append(target.position_best.position_error_m)
            normal_errors.append(target.normal_best.normal_error_deg)
            normal_position_errors.append(target.normal_best.position_error_m)
            if target.position_best.success:
                position_success += 1
            if target.normal_best.success:
                normal_success += 1
            else:
                failures[task.task_id] = target.normal_best.failure_type or "NORMAL_WORKSPACE"
        results.append(
            AxisArchitectureResult(
                architecture_id=architecture_id,
                design_name=spec.name,
                task_count=len(tasks),
                normal_success_count=normal_success,
                position_success_count=position_success,
                worst_position_error_m=max(position_errors, default=float("inf")),
                worst_normal_error_deg=max(normal_errors, default=float("inf")),
                worst_normal_position_error_m=max(normal_position_errors, default=float("inf")),
                task_failure_types=failures,
            )
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "architecture_id", "design_name", "task_count", "normal_success_count",
                    "position_success_count", "worst_position_error_m", "worst_normal_error_deg",
                    "worst_normal_position_error_m", "task_failure_types",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow({
                    "architecture_id": result.architecture_id,
                    "design_name": result.design_name,
                    "task_count": result.task_count,
                    "normal_success_count": result.normal_success_count,
                    "position_success_count": result.position_success_count,
                    "worst_position_error_m": result.worst_position_error_m,
                    "worst_normal_error_deg": result.worst_normal_error_deg,
                    "worst_normal_position_error_m": result.worst_normal_position_error_m,
                    "task_failure_types": result.task_failure_types,
                })
    return tuple(results)
