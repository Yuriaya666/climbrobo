"""GUI playback for one collision-audit state.

This script only displays a fixed state.  It does not run IK, trajectory
planning, RRT, or any design search.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pybullet as p

from environment.design.collision_audit import (
    _actual_body_poses,
    _baseline_states,
    _pair_by_name,
    _pose_matrix,
    _proxy_self_pairs,
    fixed_axis8_spec,
)
from environment.design.morphology import BaselineGeometry, MorphologyModel
from environment.paths import ProjectPaths
from environment.design.collision_proxy import MorphologyCollisionWorld
from environment.design.task_suite import build_task_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="显示固定参数化8R碰撞审计状态")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--state", choices=("initial", "step1_final", "step2_final"), default="initial")
    parser.add_argument(
        "--q-goal",
        type=Path,
        default=None,
        help="已保存的单个q_goal NPZ；提供后显示该任务终点而不是Baseline状态",
    )
    parser.add_argument("--pair", default="central_body,right_link_3", help="要高亮的代理pair")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    paths = ProjectPaths.from_repo_root(args.repo_root)
    geometry = BaselineGeometry.from_project(paths)
    spec = fixed_axis8_spec(geometry)
    model = MorphologyModel(spec)
    selected_name = args.state
    task_id = None
    if args.q_goal is None:
        states = _baseline_states(paths)
        selected = next(state for state in states if state.name == args.state)
        body_poses, _ = _actual_body_poses(paths, states)
        state = model.forward(
            geometry.baseline_joint_vector(selected.q_urdf),
            body_pose=_pose_matrix(body_poses[selected.name]),
        )
    else:
        with np.load(args.q_goal, allow_pickle=True) as data:
            q = np.asarray(data["q"], dtype=float)
            task_id = str(np.asarray(data["task_id"]).item())
        task = next(item for item in build_task_suite(paths, max_targets_per_task=1) if item.task_id == task_id)
        state = model.world_state_for_support(q, task.support_endpoint, _pose_matrix(task.support_pose))
        selected_name = f"q_goal:{task_id}"

    first, second = (item.strip() for item in args.pair.split(",", maxsplit=1))
    with MorphologyCollisionWorld(paths, model, gui=True) as world:
        world.update(state)
        pair = _pair_by_name(_proxy_self_pairs(world), first, second)
        body_by_name = {name: body for body, name in world._body_names.items()}
        if first in body_by_name:
            p.changeVisualShape(body_by_name[first], -1, rgbaColor=[0.95, 0.08, 0.08, 1.0])
        if second in body_by_name:
            p.changeVisualShape(body_by_name[second], -1, rgbaColor=[1.0, 0.85, 0.05, 1.0])
        if pair is not None:
            point_a = np.asarray(pair["point_on_a_m"], dtype=float)
            point_b = np.asarray(pair["point_on_b_m"], dtype=float)
            p.addUserDebugLine(point_a.tolist(), point_b.tolist(), [1.0, 0.0, 1.0], lineWidth=5.0)
            midpoint = 0.5 * (point_a + point_b)
            p.addUserDebugText(
                f"{first} <-> {second}: {float(pair['distance_m']):.4f} m",
                midpoint.tolist(),
                textColorRGB=[1.0, 0.0, 1.0],
                textSize=1.2,
            )
            print("pair:", first, second)
            print("clearance_m:", pair["distance_m"])
            print("point_on_a_m:", pair["point_on_a_m"])
            print("point_on_b_m:", pair["point_on_b_m"])
        else:
            print("指定pair在当前近距离审计窗口内没有碰撞/近距离记录:", first, second)

        # Show the J7->J8 centerline explicitly so the right_link_3 audit is
        # not inferred only from the capsule's shaded volume.
        j7 = state.right_joint_poses[2][:3, 3]
        j8 = state.right_joint_poses[3][:3, 3]
        p.addUserDebugLine(j7.tolist(), j8.tolist(), [0.1, 1.0, 0.1], lineWidth=5.0)
        p.addUserDebugText("J7", j7.tolist(), textColorRGB=[0.1, 1.0, 0.1], textSize=1.2)
        p.addUserDebugText("J8", j8.tolist(), textColorRGB=[0.1, 1.0, 0.1], textSize=1.2)
        print("J7 world:", j7.tolist())
        print("J8 world:", j8.tolist())
        print("right_link_3 centerline length m:", float(np.linalg.norm(j8 - j7)))
        print("right_link_3 capsule radius m:", float(spec.link_proxy_radius_m + spec.collision_inflation_m))
        print("GUI playback state:", selected_name)
        deadline = time.time() + max(0.1, args.seconds)
        while time.time() < deadline:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
