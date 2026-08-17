"""关键attach目标的位置工作空间独立验证。"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from environment.design.morphology import MorphologyModel
from environment.design.task_suite import TaskSpec


@dataclass(frozen=True)
class WorkspaceRun:
    design_name: str
    task_id: str
    seed: int
    best_error_m: float
    best_q: np.ndarray
    achieved_xyz_m: np.ndarray
    evaluations: int
    runtime_s: float


def run_position_workspace(
    model: MorphologyModel,
    task: TaskSpec,
    *,
    target_index: int = 0,
    seeds: tuple[int, ...] = (1, 2, 3),
    maxiter: int = 80,
    popsize: int = 8,
    polish: bool = True,
) -> tuple[WorkspaceRun, ...]:
    sample = task.targets[target_index]
    lower = model.spec.lower_limits
    upper = model.spec.upper_limits
    runs = []
    for seed in seeds:
        started = time.perf_counter()

        def objective(q: np.ndarray) -> float:
            state = model.world_state_for_support(q, task.support_endpoint, _pose_matrix(task))
            return float(np.linalg.norm(state.suction_pose(task.moving_endpoint)[:3, 3] - sample.xyz_m))

        result = differential_evolution(
            objective,
            bounds=list(zip(lower, upper)),
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            polish=False,
            updating="immediate",
            workers=1,
        )
        polished = least_squares(
            lambda q: _position_residual(model, task, sample.xyz_m, q),
            x0=np.clip(result.x, lower, upper),
            bounds=(lower, upper),
            max_nfev=3000,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        q = np.clip(polished.x if polish else result.x, lower, upper)
        state = model.world_state_for_support(q, task.support_endpoint, _pose_matrix(task))
        xyz = state.suction_pose(task.moving_endpoint)[:3, 3].copy()
        runs.append(
            WorkspaceRun(
                design_name=model.spec.name,
                task_id=task.task_id,
                seed=seed,
                best_error_m=float(np.linalg.norm(xyz - sample.xyz_m)),
                best_q=q.copy(),
                achieved_xyz_m=xyz,
                evaluations=int(result.nfev),
                runtime_s=time.perf_counter() - started,
            )
        )
    return tuple(runs)


def _pose_matrix(task: TaskSpec) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = task.support_pose.rotation_matrix
    pose[:3, 3] = task.support_pose.position
    return pose


def _position_residual(model, task, target, q):
    state = model.world_state_for_support(q, task.support_endpoint, _pose_matrix(task))
    return state.suction_pose(task.moving_endpoint)[:3, 3] - target


def save_workspace_runs(path: Path, runs: tuple[WorkspaceRun, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["design_name", "task_id", "seed", "best_error_m", "achieved_x", "achieved_y", "achieved_z", "evaluations", "runtime_s", "best_q"])
        for run in runs:
            writer.writerow([
                run.design_name,
                run.task_id,
                run.seed,
                run.best_error_m,
                *run.achieved_xyz_m.tolist(),
                run.evaluations,
                run.runtime_s,
                run.best_q.tolist(),
            ])

