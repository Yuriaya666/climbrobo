"""PyBullet GUI查看参数化候选在一个真实任务目标下的构型。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p

from environment.design.axis_architecture import FINITE_AXIS_ARCHITECTURES
from environment.design.collision_proxy import MorphologyCollisionWorld
from environment.design.evaluator import DesignEvaluationSettings, MorphologyTaskEvaluator
from environment.design.morphology import BaselineGeometry, MorphologyModel, MorphologySpec, with_axis_architecture
from environment.design.task_suite import build_task_suite
from environment.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GUI查看6R/8R参数化机构候选")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--design", choices=("baseline8", "best6", "best8", "axis8"), default="axis8")
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--keep-open", action="store_true")
    return parser


def _load_spec(paths: ProjectPaths, name: str) -> MorphologySpec:
    geometry = BaselineGeometry.from_project(paths)
    if name == "baseline8":
        return MorphologySpec.baseline_8r(geometry)
    if name == "best6":
        data = json.loads((paths.repo_root / "models/design_results/best_6r.json").read_text(encoding="utf-8"))
        return MorphologySpec.six_r_topology(
            geometry,
            int(data["remove_pair_index"]),
            link_lengths_m=np.asarray(data["link_lengths_m"], dtype=float),
        )
    if name == "best8":
        data = json.loads((paths.repo_root / "models/design_results/best_8r.json").read_text(encoding="utf-8"))
        return MorphologySpec.optimized_8r(geometry, np.asarray(data["link_lengths_m"], dtype=float))
    data = json.loads((paths.repo_root / "models/design_results/best_8r.json").read_text(encoding="utf-8"))
    base = MorphologySpec.optimized_8r(geometry, np.asarray(data["link_lengths_m"], dtype=float))
    axes = dict(FINITE_AXIS_ARCHITECTURES)["yaw_pitch_yaw_pitch"]
    return with_axis_architecture(base, axes, architecture_id="yaw_pitch_yaw_pitch")


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    spec = _load_spec(paths, args.design)
    model = MorphologyModel(spec)
    tasks = build_task_suite(paths, max_targets_per_task=1)
    task = tasks[args.task_index % len(tasks)]
    evaluator = MorphologyTaskEvaluator(
        model,
        settings=DesignEvaluationSettings(seed_count=8, yaw_samples=1, local_max_nfev=150, normal_max_nfev=400),
    )
    result = evaluator.evaluate_task(task, target_limit=1, collision=False, trajectory=False)
    if not result.target_results:
        raise RuntimeError("任务没有目标")
    target = result.target_results[0]
    state = model.world_state_for_support(
        target.normal_best.q,
        task.support_endpoint,
        np.block([[task.support_pose.rotation_matrix, task.support_pose.position[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]),
    )
    with MorphologyCollisionWorld(paths, model, gui=True) as world:
        world.update(state)
        world.check(allowed_endpoint_positions={task.support_endpoint: task.support_xyz, task.moving_endpoint: target.sample.xyz_m})
        p.addUserDebugLine(target.sample.xyz_m.tolist(), state.suction_pose(task.moving_endpoint)[:3, 3].tolist(), [1.0, 0.1, 0.1], lineWidth=4.0)
        p.addUserDebugText(f"{spec.name} / {task.task_id}", [0.0, 0.0, 0.2], textColorRGB=[1.0, 1.0, 1.0], textSize=1.4)
        print("设计构型已显示")
        print("design:", spec.name)
        print("task:", task.task_id)
        print("target xyz:", target.sample.xyz_m.tolist())
        print("best position error:", target.normal_best.position_error_m)
        print("best normal error deg:", target.normal_best.normal_error_deg)
        if args.keep_open:
            while p.isConnected():
                time.sleep(0.1)
        else:
            time.sleep(3.0)


if __name__ == "__main__":
    main()

