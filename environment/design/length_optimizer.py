"""6R拓扑和8R杆长的任务驱动粗优化。"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from environment.design.evaluator import DesignEvaluationSettings, MorphologyTaskEvaluator
from environment.design.morphology import BaselineGeometry, MorphologyModel, MorphologySpec
from environment.design.task_suite import TaskSpec


@dataclass(frozen=True)
class LengthSearchConfig:
    seed: int = 20260815
    maxiter: int = 8
    popsize: int = 6
    polish: bool = False
    target_limit: int = 1
    seed_count: int = 1
    yaw_samples: int = 1
    local_max_nfev: int = 70
    normal_max_nfev: int = 100
    lower_scale: float = 0.60
    upper_scale: float = 1.80
    upper_absolute_m: float = 0.38
    selection: str = "position"


@dataclass(frozen=True)
class DesignSearchResult:
    name: str
    topology_id: str
    dof: int
    per_side_dof: int
    remove_pair_index: int | None
    link_lengths_m: np.ndarray
    nominal_link_lengths_m: np.ndarray
    bounds: tuple[tuple[float, float], ...]
    task_success_count: int
    task_count: int
    position_success_count: int
    worst_position_error_m: float
    worst_normal_error_deg: float
    task_failure_types: dict[str, str]
    runtime_s: float
    de_seed: int
    de_nfev: int
    de_nit: int
    boundary_hits: tuple[int, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "topology_id": self.topology_id,
            "dof": self.dof,
            "per_side_dof": self.per_side_dof,
            "remove_pair_index": self.remove_pair_index,
            "link_lengths_m": self.link_lengths_m.tolist(),
            "nominal_link_lengths_m": self.nominal_link_lengths_m.tolist(),
            "bounds": [list(bound) for bound in self.bounds],
            "task_success_count": self.task_success_count,
            "task_count": self.task_count,
            "position_success_count": self.position_success_count,
            "worst_position_error_m": self.worst_position_error_m,
            "worst_normal_error_deg": self.worst_normal_error_deg,
            "task_failure_types": self.task_failure_types,
            "runtime_s": self.runtime_s,
            "de_seed": self.de_seed,
            "de_nfev": self.de_nfev,
            "de_nit": self.de_nit,
            "boundary_hits": list(self.boundary_hits),
        }


def build_spec(
    geometry: BaselineGeometry,
    *,
    kind: str,
    link_lengths_m: np.ndarray,
    remove_pair_index: int | None = None,
    collision_inflation_m: float = 0.005,
) -> MorphologySpec:
    if kind == "baseline_8r":
        return MorphologySpec.baseline_8r(geometry, collision_inflation_m=collision_inflation_m)
    if kind == "optimized_8r":
        return MorphologySpec.optimized_8r(
            geometry,
            np.asarray(link_lengths_m, dtype=float),
            collision_inflation_m=collision_inflation_m,
        )
    if kind == "6r":
        if remove_pair_index is None:
            raise ValueError("6R必须给出remove_pair_index")
        return MorphologySpec.six_r_topology(
            geometry,
            remove_pair_index,
            link_lengths_m=np.asarray(link_lengths_m, dtype=float),
            collision_inflation_m=collision_inflation_m,
        )
    raise ValueError(f"未知设计类型：{kind}")


def length_bounds(spec: MorphologySpec, config: LengthSearchConfig) -> tuple[tuple[float, float], ...]:
    bounds = []
    for nominal in spec.nominal_link_lengths_m:
        low = max(0.05, float(nominal) * config.lower_scale)
        high = min(config.upper_absolute_m, float(nominal) * config.upper_scale)
        if high <= low:
            raise ValueError(f"杆长边界无效：nominal={nominal}, low={low}, high={high}")
        bounds.append((low, high))
    return tuple(bounds)


def evaluate_lengths(
    geometry: BaselineGeometry,
    tasks: tuple[TaskSpec, ...],
    *,
    kind: str,
    lengths: np.ndarray,
    remove_pair_index: int | None,
    config: LengthSearchConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    spec = build_spec(
        geometry,
        kind=kind,
        link_lengths_m=lengths,
        remove_pair_index=remove_pair_index,
    )
    evaluator = MorphologyTaskEvaluator(
        MorphologyModel(spec),
        settings=DesignEvaluationSettings(
            local_max_nfev=config.local_max_nfev,
            normal_max_nfev=config.normal_max_nfev,
            seed_count=config.seed_count,
            yaw_samples=config.yaw_samples,
            random_seed=config.seed,
        ),
    )
    task_rows: dict[str, object] = {}
    position_errors = []
    normal_errors = []
    goal_residuals = []
    position_success = 0
    task_success = 0
    failures: dict[str, str] = {}
    for task in tasks:
        result = evaluator.evaluate_task(task, target_limit=config.target_limit, collision=False, trajectory=False)
        if not result.target_results:
            failures[task.task_id] = "NO_TARGET"
            continue
        target = result.target_results[0]
        position_error = float(target.position_best.position_error_m)
        normal_error = float(target.normal_best.normal_error_deg)
        goal_residual = float(
            target.normal_best.position_error_m
            + 0.05 * np.deg2rad(normal_error)
        )
        position_errors.append(position_error)
        normal_errors.append(normal_error)
        goal_residuals.append(goal_residual)
        position_ok = position_error <= evaluator.settings.position_tolerance_m
        normal_ok = target.normal_best.success
        if position_ok:
            position_success += 1
        if normal_ok:
            task_success += 1
        else:
            failures[task.task_id] = target.failure_type or target.normal_best.failure_type or "NORMAL_WORKSPACE"
        task_rows[task.task_id] = {
            "position_error_m": position_error,
            "normal_best_position_error_m": float(target.normal_best.position_error_m),
            "normal_error_deg": normal_error,
            "position_success": position_ok,
            "normal_success": normal_ok,
            "failure_type": target.failure_type,
            "target_segment_id": target.sample.segment_id,
            "target_s_m": target.sample.s_m,
            "target_xyz_m": target.sample.xyz_m.tolist(),
        }
    metrics = {
        "task_success_count": task_success,
        "position_success_count": position_success,
        "task_count": len(tasks),
        "worst_position_error_m": max(position_errors, default=float("inf")),
        "worst_normal_error_deg": max(normal_errors, default=float("inf")),
        "worst_goal_residual_m": max(goal_residuals, default=float("inf")),
        "task_failure_types": failures,
    }
    return metrics, task_rows


def optimize_lengths(
    geometry: BaselineGeometry,
    tasks: tuple[TaskSpec, ...],
    *,
    kind: str,
    remove_pair_index: int | None,
    config: LengthSearchConfig,
    output_dir: Path | None = None,
) -> DesignSearchResult:
    """用DE最小化任务集的最坏位置误差，成功数优先于误差。"""

    if kind == "6r":
        nominal_spec = MorphologySpec.six_r_topology(geometry, int(remove_pair_index))
    elif kind == "optimized_8r":
        nominal_spec = MorphologySpec.baseline_8r(geometry)
    else:
        nominal_spec = MorphologySpec.baseline_8r(geometry)
    bounds = length_bounds(nominal_spec, config)
    cache: dict[tuple[float, ...], tuple[dict[str, object], dict[str, object]]] = {}
    evaluation_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    def evaluate(vector: np.ndarray) -> tuple[dict[str, object], dict[str, object]]:
        key = tuple(np.round(np.asarray(vector, dtype=float), 8))
        if key not in cache:
            cache[key] = evaluate_lengths(
                geometry,
                tasks,
                kind=kind,
                lengths=np.asarray(vector, dtype=float),
                remove_pair_index=remove_pair_index,
                config=config,
            )
            metrics, _ = cache[key]
            evaluation_rows.append({"lengths_m": list(key), **metrics})
        return cache[key]

    def objective(vector: np.ndarray) -> float:
        metrics, _ = evaluate(vector)
        # 先最大化通过任务数量，再最小化最坏位置误差；这是分层筛选编码，
        # 不是把碰撞、工作空间等混成主观加权总分。
        if config.selection == "normal":
            failed = int(metrics["task_count"]) - int(metrics["task_success_count"])
            return failed * 1000.0 + float(metrics["worst_goal_residual_m"])
        failed = int(metrics["task_count"]) - int(metrics["position_success_count"])
        return failed * 1000.0 + float(metrics["worst_position_error_m"])

    if kind == "baseline_8r":
        # Baseline必须保留真实杆长，不能让DE返回一组无效的“基线长度”。
        best_lengths = np.asarray(nominal_spec.link_lengths_m, dtype=float)
        metrics, task_rows = evaluate(best_lengths)
        de_nfev = 1
        de_nit = 0
    else:
        result = differential_evolution(
            objective,
            bounds=list(bounds),
            seed=config.seed,
            maxiter=config.maxiter,
            popsize=config.popsize,
            polish=config.polish,
            updating="immediate",
            workers=1,
            disp=False,
        )
        best_lengths = np.asarray(result.x, dtype=float)
        metrics, task_rows = evaluate(best_lengths)
        de_nfev = int(result.nfev)
        de_nit = int(result.nit)
    spec = build_spec(
        geometry,
        kind=kind,
        link_lengths_m=best_lengths,
        remove_pair_index=remove_pair_index,
    )
    boundary_hits = tuple(
        index
        for index, (value, bound) in enumerate(zip(best_lengths, bounds))
        if min(abs(value - bound[0]), abs(value - bound[1])) <= 1e-5
    )
    search_result = DesignSearchResult(
        name=spec.name,
        topology_id=spec.topology_id,
        dof=spec.dof,
        per_side_dof=spec.per_side_dof,
        remove_pair_index=remove_pair_index,
        link_lengths_m=best_lengths,
        nominal_link_lengths_m=spec.nominal_link_lengths_m.copy(),
        bounds=bounds,
        task_success_count=int(metrics["task_success_count"]),
        task_count=int(metrics["task_count"]),
        position_success_count=int(metrics["position_success_count"]),
        worst_position_error_m=float(metrics["worst_position_error_m"]),
        worst_normal_error_deg=float(metrics["worst_normal_error_deg"]),
        task_failure_types=dict(metrics["task_failure_types"]),
        runtime_s=time.perf_counter() - started,
        de_seed=config.seed,
        de_nfev=de_nfev,
        de_nit=de_nit,
        boundary_hits=boundary_hits,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{spec.topology_id}.json").write_text(
            json.dumps(search_result.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output_dir / f"{spec.topology_id}_evaluations.csv").open("w", encoding="utf-8", newline="") as handle:
            if evaluation_rows:
                fields = sorted({key for row in evaluation_rows for key in row})
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(evaluation_rows)
        (output_dir / f"{spec.topology_id}_tasks.json").write_text(
            json.dumps(task_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return search_result
