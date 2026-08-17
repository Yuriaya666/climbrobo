"""固定8R候选的碰撞代理标定与多分支终点搜索。

本脚本只处理当前 ``OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`` 和原四个首层任务。
它不优化杆长、不搜索轴架构、不做整塔规划；只有在找到合法终点后，才按
现有预算尝试 Straight，再按现有预算尝试 RRT-Connect。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.optimize import least_squares, minimize

from environment.attach_lines import AttachLineSample
from environment.candidates import CandidatePoint
from environment.design.collision_audit import (
    _actual_body_poses,
    _actual_self_pairs,
    _baseline_states,
    _pose_matrix,
    _proxy_self_pairs,
    fixed_axis8_spec,
)
from environment.design.collision_proxy import MorphologyCollisionWorld
from environment.design.morphology import BaselineGeometry, MorphologyModel, MorphologySpec
from environment.design.proxy_calibration import ProxyCalibration, calibrate_proxy_dimensions
from environment.design.task_suite import TaskSpec, build_task_suite
from environment.one_step_planner import AttachmentPoseBuilder
from environment.paths import ProjectPaths
from environment.rrt_connect import RRTConnect
from environment.scene import PyBulletScene
from environment.transforms import angle_between_vectors_rad


NOMINAL_INFLATION_M = 0.005
EXTRA_INFLATIONS_M = (0.005, 0.010)
DEFAULT_YAW_SAMPLES = 8
DEFAULT_SEEDS_PER_YAW = 8
DEFAULT_NORMAL_ONLY_SEEDS = 32
DEFAULT_NORMAL_MAX_NFEV = 400
DEFAULT_BRANCH_DEDUP_RAD = 1e-3
ALLOWED_CONTACT_RADIUS_M = 0.09
COLLISION_QUERY_DISTANCE_M = 0.05


@dataclass(frozen=True)
class SearchConfig:
    yaw_samples: int = DEFAULT_YAW_SAMPLES
    seeds_per_yaw: int = DEFAULT_SEEDS_PER_YAW
    normal_only_seeds: int = DEFAULT_NORMAL_ONLY_SEEDS
    normal_max_nfev: int = DEFAULT_NORMAL_MAX_NFEV
    random_seed: int = 20260816
    branch_dedup_rad: float = DEFAULT_BRANCH_DEDUP_RAD
    refinement_top: int = 2
    refinement_maxiter: int = 12
    rrt_max_iterations: int = 1200
    rrt_seed_count: int = 2
    rrt_step_size_rad: float = 0.2
    rrt_edge_resolution_rad: float = 0.08


def _scalar(value: object) -> object:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def _task_pose(task_pose) -> np.ndarray:
    return _pose_matrix(task_pose)


def _joint_margin(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    values = np.asarray(q, dtype=float)
    return float(np.min(np.minimum(values - lower, upper - values)))


def _wrapped_distance(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.arctan2(np.sin(np.asarray(first) - np.asarray(second)), np.cos(np.asarray(first) - np.asarray(second)))
    return float(np.linalg.norm(delta))


def _target_candidate(task: TaskSpec, sample: AttachLineSample) -> CandidatePoint:
    legacy_foot = "foot1" if task.target_surface == "surface1" else "foot2"
    return CandidatePoint(
        foot_name=legacy_foot,
        point_id=int(sample.segment_id),
        region_id=int(sample.segment_id),
        xyz_m=np.asarray(sample.xyz_m, dtype=float),
        normal=np.asarray(sample.normal, dtype=float),
        uv_m=np.asarray(sample.uv_m, dtype=float),
        surface_name=task.target_surface,
    )


def _target_pose(task: TaskSpec, sample: AttachLineSample, yaw_rad: float) -> np.ndarray:
    pose = AttachmentPoseBuilder().build(
        _target_candidate(task, sample),
        preferred_y_reference_world=np.array([0.0, 0.0, 1.0], dtype=float),
        yaw_rad=float(yaw_rad),
    )
    return _task_pose(pose)


def _normal_error_deg(model: MorphologyModel, task: TaskSpec, q: np.ndarray, sample: AttachLineSample) -> float:
    pose = model.world_state_for_support(q, task.support_endpoint, _task_pose(task.support_pose)).suction_pose(task.moving_endpoint)
    return float(np.degrees(angle_between_vectors_rad(pose[:3, 2], -np.asarray(sample.normal, dtype=float))))


def _position_error(model: MorphologyModel, task: TaskSpec, q: np.ndarray, sample: AttachLineSample) -> float:
    pose = model.world_state_for_support(q, task.support_endpoint, _task_pose(task.support_pose)).suction_pose(task.moving_endpoint)
    return float(np.linalg.norm(pose[:3, 3] - np.asarray(sample.xyz_m, dtype=float)))


def _normal_residual(model: MorphologyModel, task: TaskSpec, sample: AttachLineSample, q: np.ndarray) -> np.ndarray:
    pose = model.world_state_for_support(q, task.support_endpoint, _task_pose(task.support_pose)).suction_pose(task.moving_endpoint)
    desired_z = -np.asarray(sample.normal, dtype=float)
    return np.concatenate((pose[:3, 3] - np.asarray(sample.xyz_m, dtype=float), np.cross(pose[:3, 2], desired_z)))


def _full_pose_residual(model: MorphologyModel, task: TaskSpec, sample: AttachLineSample, yaw_rad: float, q: np.ndarray) -> np.ndarray:
    actual = model.world_state_for_support(q, task.support_endpoint, _task_pose(task.support_pose)).suction_pose(task.moving_endpoint)
    target = _target_pose(task, sample, yaw_rad)
    orientation = 0.5 * (
        np.cross(actual[:3, 0], target[:3, 0])
        + np.cross(actual[:3, 1], target[:3, 1])
        + np.cross(actual[:3, 2], target[:3, 2])
    )
    return np.concatenate((actual[:3, 3] - target[:3, 3], orientation))


def _endpoint_for_name(name: str) -> str | None:
    if name == "base_end_mesh" or name in {"left_joint_4", "left_link_4"}:
        return "base_end"
    if name == "l8_end_mesh" or name in {"right_joint_4", "right_link_4"}:
        return "l8_end"
    return None


def _proxy_items(world: MorphologyCollisionWorld) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    if world.central_id is not None:
        items.append((world.central_id, "central_body"))
    items.extend((body, f"{endpoint}_mesh") for endpoint, body in world.endpoint_ids.items())
    items.extend((body, world._body_names[body]) for body in (*world.segment_ids["left"], *world.segment_ids["right"], *world.joint_ids["left"], *world.joint_ids["right"]))
    return items


def _closest_record(points: tuple[object, ...], name_a: str, name_b: str, *, allowed: bool = False) -> dict[str, object] | None:
    if not points:
        return None
    point = min(points, key=lambda item: float(item[8]))
    distance = float(point[8])
    return {
        "link_a": name_a,
        "link_b": name_b,
        "distance_m": distance,
        "penetration_m": max(0.0, -distance),
        "allowed": bool(allowed),
        "point_on_a_m": np.asarray(point[5], dtype=float).tolist(),
        "point_on_b_m": np.asarray(point[6], dtype=float).tolist(),
    }


def _collision_records(
    world: MorphologyCollisionWorld,
    allowed_endpoint_positions: dict[str, np.ndarray],
    *,
    query_distance_m: float = COLLISION_QUERY_DISTANCE_M,
) -> list[dict[str, object]]:
    items = _proxy_items(world)
    records: list[dict[str, object]] = []
    if world.tower_id is not None:
        for body, name in items:
            record = _closest_record(
                tuple(p.getClosestPoints(body, world.tower_id, distance=query_distance_m)),
                name,
                "Tower",
            )
            if record is None:
                continue
            endpoint = _endpoint_for_name(name)
            if endpoint is not None and endpoint in allowed_endpoint_positions:
                point = np.asarray(record["point_on_a_m"], dtype=float)
                record["allowed"] = bool(np.linalg.norm(point - allowed_endpoint_positions[endpoint]) <= ALLOWED_CONTACT_RADIUS_M)
            records.append(record)
    for index, (body_a, name_a) in enumerate(items):
        for body_b, name_b in items[index + 1 :]:
            if tuple(sorted((body_a, body_b))) in world._allowed_adjacent_pairs:
                continue
            record = _closest_record(
                tuple(p.getClosestPoints(body_a, body_b, distance=query_distance_m)),
                name_a,
                name_b,
            )
            if record is not None:
                records.append(record)
    return records


def _collision_summary(
    world: MorphologyCollisionWorld,
    state,
    task: TaskSpec,
    sample: AttachLineSample,
    *,
    query_distance_m: float = 0.02,
) -> dict[str, object]:
    world.update(state)
    allowed_positions = {
        task.support_endpoint: np.asarray(task.support_xyz, dtype=float),
        task.moving_endpoint: np.asarray(sample.xyz_m, dtype=float),
    }
    official = world.check(
        allowed_endpoint_positions=allowed_positions,
        allowed_contact_radius_m=ALLOWED_CONTACT_RADIUS_M,
    )
    records = _collision_records(world, allowed_positions, query_distance_m=query_distance_m)
    disallowed = [row for row in records if not row.get("allowed", False)]
    bad = [row for row in disallowed if float(row["distance_m"]) <= 0.0]
    critical = min(disallowed, key=lambda row: float(row["distance_m"]), default=None)
    return {
        "pass": bool(official.ok and not bad),
        # No record within the query window means the clearance is at least
        # that window, not mathematically infinite.
        "minimum_clearance_m": float(critical["distance_m"]) if critical is not None else float(query_distance_m),
        "critical_pair": None if critical is None else f"{critical['link_a']}↔{critical['link_b']}",
        "critical": critical,
        "bad_pairs": bad,
        "bad_pair_counts": dict(Counter(f"{row['link_a']}↔{row['link_b']}" for row in bad)),
        "self_collision": any(row["link_b"] != "Tower" and row["distance_m"] <= 0.0 for row in bad),
        "tower_collision": any(row["link_b"] == "Tower" and row["distance_m"] <= 0.0 for row in bad),
    }


def _load_saved_q_goals(q_goal_dir: Path) -> dict[str, np.ndarray]:
    goals: dict[str, np.ndarray] = {}
    for path in sorted(q_goal_dir.glob("optimized_8r_axis8_task_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            task_id = str(np.asarray(data["task_id"]).item())
            q = np.asarray(data["q"], dtype=float)
        if q.shape != (8,):
            raise ValueError(f"{path} 的 q 不是8维")
        goals[task_id] = q.copy()
    return goals


def _write_dimensions_csv(path: Path, calibration: ProxyCalibration, inflation_m: float) -> list[dict[str, object]]:
    rows = calibration.dimension_rows(collision_inflation_m=inflation_m)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "component", "source_geometry", "proxy_type", "kinematic_length",
        "physical_proxy_length", "nominal_radius_m", "safety_inflation_m",
        "effective_radius_m", "source_derivation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _baseline_sanity(paths: ProjectPaths) -> dict[str, object]:
    states = _baseline_states(paths)
    geometry = BaselineGeometry.from_project(paths)
    body_poses, _ = _actual_body_poses(paths, states)
    actual_rows: list[dict[str, object]] = []
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_tower()
        scene.load_robot(states[0].base_pose)
        for audit_state in states:
            scene.set_base_pose(audit_state.base_pose)
            scene.reset_joints(audit_state.q_urdf)
            pairs = _actual_self_pairs(scene)
            bad = [row for row in pairs if not row["allowed_adjacent"] and float(row["distance_m"]) <= 0.0]
            actual_rows.append({"state": audit_state.name, "non_adjacent_self_collision_count": len(bad), "pairs": bad})
    spec = MorphologySpec.baseline_8r(geometry, collision_inflation_m=NOMINAL_INFLATION_M)
    model = MorphologyModel(spec)
    proxy_rows: list[dict[str, object]] = []
    with MorphologyCollisionWorld(paths, model, gui=False) as world:
        for audit_state in states:
            state = model.forward(
                geometry.baseline_joint_vector(audit_state.q_urdf),
                body_pose=_pose_matrix(body_poses[audit_state.name]),
            )
            world.update(state)
            pairs = _proxy_self_pairs(world)
            bad = [row for row in pairs if not row["allowed_adjacent"] and float(row["distance_m"]) <= 0.0]
            proxy_rows.append({"state": audit_state.name, "non_adjacent_self_collision_count": len(bad), "pairs": bad})
    return {
        "actual_urdf": actual_rows,
        "calibrated_proxy": proxy_rows,
        "actual_counts": [row["non_adjacent_self_collision_count"] for row in actual_rows],
        "proxy_counts": [row["non_adjacent_self_collision_count"] for row in proxy_rows],
        "status": "PASS" if all(row["non_adjacent_self_collision_count"] == 0 for row in actual_rows + proxy_rows) else "FAIL",
    }


def _audit_saved_goals(
    paths: ProjectPaths,
    spec: MorphologySpec,
    goals: dict[str, np.ndarray],
    tasks: tuple[TaskSpec, ...],
) -> dict[str, object]:
    task_map = {task.task_id: task for task in tasks}
    model = MorphologyModel(spec)
    variants: list[dict[str, object]] = []
    for label, inflation in (
        ("nominal", NOMINAL_INFLATION_M),
        ("nominal_plus_5mm", NOMINAL_INFLATION_M + 0.005),
        ("nominal_plus_10mm", NOMINAL_INFLATION_M + 0.010),
    ):
        variant = replace(spec, collision_inflation_m=inflation)
        rows: list[dict[str, object]] = []
        with MorphologyCollisionWorld(paths, MorphologyModel(variant), gui=False) as world:
            variant_model = MorphologyModel(variant)
            for task_id, q in sorted(goals.items()):
                task = task_map.get(task_id)
                if task is None:
                    continue
                sample = task.targets[0]
                state = variant_model.world_state_for_support(q, task.support_endpoint, _task_pose(task.support_pose))
                position_error = _position_error(variant_model, task, q, sample)
                normal_error = _normal_error_deg(variant_model, task, q, sample)
                collision = _collision_summary(world, state, task, sample)
                rows.append(
                    {
                        "task_id": task_id,
                        "position_error_m": position_error,
                        "normal_error_deg": normal_error,
                        "collision_pass": collision["pass"],
                        "minimum_clearance_m": collision["minimum_clearance_m"],
                        "critical_pair": collision["critical_pair"],
                        "penetration_m": None if collision["critical"] is None else collision["critical"]["penetration_m"],
                        "self_collision": collision["self_collision"],
                        "tower_collision": collision["tower_collision"],
                    }
                )
        variants.append({"label": label, "inflation_m": inflation, "collision_pass_count": sum(row["collision_pass"] for row in rows), "tasks": rows})
    return {
        "q_goal_count": len(goals),
        "variants": variants,
        "all_saved": len(goals) == len(tasks),
    }


def _seed_list(task: TaskSpec, model: MorphologyModel, goals: dict[str, np.ndarray], rng: np.random.Generator, count: int) -> list[np.ndarray]:
    lower = model.spec.lower_limits
    upper = model.spec.upper_limits
    seeds = [np.clip(np.asarray(task.start_q, dtype=float), lower, upper)]
    if task.task_id in goals:
        seeds.append(np.clip(goals[task.task_id], lower, upper))
    while len(seeds) < count:
        seeds.append(rng.uniform(lower, upper))
    return seeds[:count]


def _run_ik_search_for_task(
    task: TaskSpec,
    sample: AttachLineSample,
    model: MorphologyModel,
    goals: dict[str, np.ndarray],
    config: SearchConfig,
) -> dict[str, object]:
    lower = model.spec.lower_limits
    upper = model.spec.upper_limits
    rng = np.random.default_rng(config.random_seed + sum(ord(char) for char in task.task_id))
    valid_candidates: list[dict[str, object]] = []
    raw_attempts = 0
    valid_count = 0
    yaw_values = np.linspace(0.0, 2.0 * math.pi, config.yaw_samples, endpoint=False)

    def append_attempt(q: np.ndarray, mode: str, yaw: float) -> None:
        nonlocal valid_count
        q = np.clip(np.asarray(q, dtype=float), lower, upper)
        position_error = _position_error(model, task, q, sample)
        normal_error = _normal_error_deg(model, task, q, sample)
        if position_error > 0.005 or normal_error > 3.0:
            return
        valid_count += 1
        if any(_wrapped_distance(q, item["q"]) <= config.branch_dedup_rad for item in valid_candidates):
            return
        valid_candidates.append({
            "q": q.copy(),
            "mode": mode,
            "yaw_rad": float(yaw),
            "position_error_m": position_error,
            "normal_error_deg": normal_error,
        })

    for yaw_index, yaw in enumerate(yaw_values):
        seeds = _seed_list(task, model, goals, rng, config.seeds_per_yaw)
        for seed_index, seed in enumerate(seeds):
            raw_attempts += 1
            result = least_squares(
                lambda q, yaw_value=float(yaw): _full_pose_residual(model, task, sample, yaw_value, q),
                x0=np.clip(seed, lower, upper),
                bounds=(lower, upper),
                max_nfev=config.normal_max_nfev,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
            )
            append_attempt(result.x, "yaw_conditioned", float(yaw))

    for seed in _seed_list(task, model, goals, rng, config.normal_only_seeds):
        raw_attempts += 1
        result = least_squares(
            lambda q: _normal_residual(model, task, sample, q),
            x0=np.clip(seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=config.normal_max_nfev,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        append_attempt(result.x, "normal_only", 0.0)

    return {
        "task_id": task.task_id,
        "yaw_samples": config.yaw_samples,
        "seeds_per_yaw": config.seeds_per_yaw,
        "normal_only_seeds": config.normal_only_seeds,
        "raw_ik_attempts": raw_attempts,
        "position_normal_converged_count": valid_count,
        "unique_ik_branch_count": len(valid_candidates),
        "candidates": valid_candidates,
    }


def _collision_filter(
    task: TaskSpec,
    sample: AttachLineSample,
    model: MorphologyModel,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    collision_free: list[dict[str, object]] = []
    pair_counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    with MorphologyCollisionWorld(ProjectPaths.from_repo_root(), model, gui=False) as world:
        for candidate in candidates:
            q = np.asarray(candidate["q"], dtype=float)
            state = model.world_state_for_support(q, task.support_endpoint, _task_pose(task.support_pose))
            collision = _collision_summary(world, state, task, sample)
            if collision["pass"]:
                # Positive-clearance ranking is expensive against the dense
                # Tower mesh, so only legal candidates receive the wider
                # clearance query.
                collision = _collision_summary(
                    world,
                    state,
                    task,
                    sample,
                    query_distance_m=COLLISION_QUERY_DISTANCE_M,
                )
            if not collision["pass"]:
                pair_counts.update(collision["bad_pair_counts"])
            row = {
                "mode": candidate["mode"],
                "yaw_rad": candidate["yaw_rad"],
                "position_error_m": candidate["position_error_m"],
                "normal_error_deg": candidate["normal_error_deg"],
                "minimum_clearance_m": collision["minimum_clearance_m"],
                "critical_pair": collision["critical_pair"],
                "collision_free": collision["pass"],
                "joint_limit_margin_rad": _joint_margin(q, model.spec.lower_limits, model.spec.upper_limits),
                "q": q.tolist(),
            }
            rows.append(row)
            if collision["pass"]:
                collision_free.append({**candidate, "collision": collision, "joint_limit_margin_rad": row["joint_limit_margin_rad"]})
    collision_free.sort(
        key=lambda item: (
            float(item["collision"]["minimum_clearance_m"]) if np.isfinite(item["collision"]["minimum_clearance_m"]) else COLLISION_QUERY_DISTANCE_M,
            float(item["joint_limit_margin_rad"]),
        ),
        reverse=True,
    )
    return {
        "candidate_rows": rows,
        "collision_free": collision_free,
        "collision_free_endpoint_count": len(collision_free),
        "collision_pair_counts": dict(pair_counts),
        "best_collision_free": collision_free[0] if collision_free else None,
    }


def _refine_candidate(
    task: TaskSpec,
    sample: AttachLineSample,
    model: MorphologyModel,
    seed: np.ndarray,
    config: SearchConfig,
) -> np.ndarray:
    lower = model.spec.lower_limits
    upper = model.spec.upper_limits
    position_scale = 0.005
    normal_scale = 3.0
    with MorphologyCollisionWorld(ProjectPaths.from_repo_root(), model, gui=False) as world:
        def objective(q: np.ndarray) -> float:
            values = np.clip(np.asarray(q, dtype=float), lower, upper)
            position_error = _position_error(model, task, values, sample)
            normal_error = _normal_error_deg(model, task, values, sample)
            state = model.world_state_for_support(values, task.support_endpoint, _task_pose(task.support_pose))
            collision = _collision_summary(world, state, task, sample)
            clearance = collision["minimum_clearance_m"]
            if not np.isfinite(clearance):
                clearance = COLLISION_QUERY_DISTANCE_M
            clearance = float(np.clip(clearance, -0.10, COLLISION_QUERY_DISTANCE_M))
            return (
                100.0 * (position_error / position_scale) ** 2
                + 100.0 * (normal_error / normal_scale) ** 2
                - clearance / COLLISION_QUERY_DISTANCE_M
            )

        result = minimize(
            objective,
            np.clip(seed, lower, upper),
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"maxiter": config.refinement_maxiter, "ftol": 1e-10, "maxls": 10},
        )
    return np.clip(result.x, lower, upper)


def _collision_aware_task_search(
    paths: ProjectPaths,
    task: TaskSpec,
    model: MorphologyModel,
    goals: dict[str, np.ndarray],
    config: SearchConfig,
) -> dict[str, object]:
    sample = task.targets[0]
    started = time.perf_counter()
    ik = _run_ik_search_for_task(task, sample, model, goals, config)
    filtered = _collision_filter(task, sample, model, ik["candidates"])
    refinement_attempts = 0
    refined_candidates: list[dict[str, object]] = []
    if not filtered["collision_free"] and filtered["candidate_rows"]:
        ranked = sorted(
            filtered["candidate_rows"],
            key=lambda row: float(row["minimum_clearance_m"]) if np.isfinite(row["minimum_clearance_m"]) else COLLISION_QUERY_DISTANCE_M,
            reverse=True,
        )[: config.refinement_top]
        for row in ranked:
            refinement_attempts += 1
            refined_q = _refine_candidate(task, sample, model, np.asarray(row["q"], dtype=float), config)
            position_error = _position_error(model, task, refined_q, sample)
            normal_error = _normal_error_deg(model, task, refined_q, sample)
            if position_error <= 0.005 and normal_error <= 3.0:
                refined_candidates.append(
                    {
                        "q": refined_q,
                        "mode": "collision_aware_refinement",
                        "yaw_rad": float(row["yaw_rad"]),
                        "position_error_m": position_error,
                        "normal_error_deg": normal_error,
                    }
                )
        if refined_candidates:
            refined_filtered = _collision_filter(task, sample, model, refined_candidates)
            filtered["candidate_rows"].extend(refined_filtered["candidate_rows"])
            filtered["collision_free"].extend(refined_filtered["collision_free"])
            filtered["collision_free"].sort(
                key=lambda item: float(item["collision"]["minimum_clearance_m"]) if np.isfinite(item["collision"]["minimum_clearance_m"]) else COLLISION_QUERY_DISTANCE_M,
                reverse=True,
            )
            filtered["collision_free_endpoint_count"] = len(filtered["collision_free"])
            filtered["best_collision_free"] = filtered["collision_free"][0] if filtered["collision_free"] else None
            for pair, count in refined_filtered["collision_pair_counts"].items():
                filtered["collision_pair_counts"][pair] = filtered["collision_pair_counts"].get(pair, 0) + count
    result = {
        "task_id": task.task_id,
        "target_xyz_m": np.asarray(sample.xyz_m, dtype=float).tolist(),
        "target_normal": np.asarray(sample.normal, dtype=float).tolist(),
        "target_segment_id": int(sample.segment_id),
        "target_s_m": float(sample.s_m),
        "ik": {key: value for key, value in ik.items() if key != "candidates"},
        "collision_free_endpoint_count": filtered["collision_free_endpoint_count"],
        "collision_pair_counts": filtered["collision_pair_counts"],
        "best_collision_free": filtered["best_collision_free"],
        "candidate_rows": filtered["candidate_rows"],
        "refinement_attempts": refinement_attempts,
        "runtime_s": time.perf_counter() - started,
    }
    return result


def _path_valid(
    world: MorphologyCollisionWorld,
    model: MorphologyModel,
    task: TaskSpec,
    sample: AttachLineSample,
    q: np.ndarray,
    *,
    allow_start: bool,
    allow_goal: bool,
) -> tuple[bool, float, str | None]:
    state = model.world_state_for_support(q, task.support_endpoint, _task_pose(task.support_pose))
    allowed = {task.support_endpoint: np.asarray(task.support_xyz, dtype=float)}
    if allow_start:
        allowed[task.moving_endpoint] = np.asarray(task.moving_start_xyz, dtype=float)
    if allow_goal:
        allowed[task.moving_endpoint] = np.asarray(sample.xyz_m, dtype=float)
    report = _collision_summary(world, state, task, sample)
    # Re-evaluate with the path-specific contact allowance because the common
    # endpoint summary always includes the goal position.
    world.update(state)
    official = world.check(allowed_endpoint_positions=allowed, allowed_contact_radius_m=ALLOWED_CONTACT_RADIUS_M)
    return bool(official.ok), float(report["minimum_clearance_m"]), report["critical_pair"]


def _save_trajectory(
    output_dir: Path,
    task: TaskSpec,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    q_path: np.ndarray,
    method: str,
    minimum_clearance_m: float,
    planning_time_s: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"optimized_8r_axis8_{task.task_id}"
    npz_path = output_dir / f"{stem}.npz"
    csv_path = output_dir / f"{stem}.csv"
    np.savez_compressed(
        npz_path,
        task_id=np.asarray(task.task_id),
        q_start=np.asarray(q_start, dtype=float),
        q_goal=np.asarray(q_goal, dtype=float),
        q_path=np.asarray(q_path, dtype=float),
        method=np.asarray(method),
        minimum_clearance_m=np.asarray(minimum_clearance_m),
        planning_time_s=np.asarray(planning_time_s),
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *[f"q{i + 1}" for i in range(q_path.shape[1])]])
        for index, row in enumerate(q_path):
            writer.writerow([index, *row.tolist()])
    return {"npz": str(npz_path), "csv": str(csv_path), "method": method, "minimum_clearance_m": minimum_clearance_m, "planning_time_s": planning_time_s}


def _save_collision_free_q_goal(
    output_dir: Path,
    task_index: int,
    task: TaskSpec,
    sample: AttachLineSample,
    best: dict[str, object],
    model: MorphologyModel,
) -> dict[str, object]:
    """保存通过 position+normal 和终点碰撞检查的最佳 branch。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"optimized_8r_axis8_task_{task_index}.npz"
    q = np.asarray(best["q"], dtype=float)
    state = model.world_state_for_support(q, task.support_endpoint, _task_pose(task.support_pose))
    achieved = state.suction_pose(task.moving_endpoint)
    np.savez_compressed(
        path,
        task_id=np.asarray(task.task_id),
        task_index=np.asarray(task_index, dtype=np.int64),
        q=q,
        target_xyz=np.asarray(sample.xyz_m, dtype=float),
        target_normal=np.asarray(sample.normal, dtype=float),
        achieved_xyz=np.asarray(achieved[:3, 3], dtype=float),
        achieved_normal=np.asarray(achieved[:3, 2], dtype=float),
        position_error_m=np.asarray(best["position_error_m"]),
        normal_error_deg=np.asarray(best["normal_error_deg"]),
        minimum_clearance_m=np.asarray(best["collision"]["minimum_clearance_m"]),
        joint_limit_margin_rad=np.asarray(best["joint_limit_margin_rad"]),
        support_endpoint=np.asarray(task.support_endpoint),
        moving_endpoint=np.asarray(task.moving_endpoint),
        support_surface=np.asarray(task.support_surface),
        target_surface=np.asarray(task.target_surface),
        morphology_name=np.asarray(model.spec.name),
        topology_id=np.asarray(model.spec.topology_id),
        link_lengths_m=np.asarray(model.spec.link_lengths_m, dtype=float),
        link_proxy_radii_m=np.asarray(model.spec.link_proxy_radii_m, dtype=float),
        joint_proxy_radii_m=np.asarray(model.spec.joint_proxy_radii_m, dtype=float),
        collision_inflation_m=np.asarray(model.spec.collision_inflation_m),
        ik_mode=np.asarray(best["mode"]),
        ik_yaw_rad=np.asarray(best["yaw_rad"]),
    )
    return {"path": str(path), "task_index": task_index}


def _plan_if_endpoint_valid(paths: ProjectPaths, task: TaskSpec, sample: AttachLineSample, model: MorphologyModel, best: dict[str, object], config: SearchConfig, trajectory_dir: Path) -> dict[str, object]:
    q_goal = np.asarray(best["q"], dtype=float)
    q_start = np.asarray(task.start_q, dtype=float)
    started = time.perf_counter()
    with MorphologyCollisionWorld(paths, model, gui=False) as world:
        straight = np.linspace(q_start, q_goal, 40)
        straight_ok = True
        minimum = float("inf")
        failure = None
        for index, q in enumerate(straight):
            ok, clearance, critical = _path_valid(world, model, task, sample, q, allow_start=index == 0, allow_goal=index == len(straight) - 1)
            minimum = min(minimum, clearance)
            if not ok:
                straight_ok = False
                failure = critical or "COLLISION"
                break
        if straight_ok:
            artifact = _save_trajectory(trajectory_dir, task, q_start, q_goal, straight, "Straight", minimum, time.perf_counter() - started)
            return {"straight_success": True, "rrt_success": False, "failure": None, "artifact": artifact}

        rrt_stats: list[dict[str, object]] = []
        for seed_index in range(config.rrt_seed_count):
            planner = RRTConnect(
                model.spec.lower_limits,
                model.spec.upper_limits,
                step_size_rad=config.rrt_step_size_rad,
                max_iterations=config.rrt_max_iterations,
                edge_resolution_rad=config.rrt_edge_resolution_rad,
                random_seed=config.random_seed + seed_index,
            )
            path = planner.plan(
                q_start,
                q_goal,
                is_state_valid=lambda q: _path_valid(
                    world,
                    model,
                    task,
                    sample,
                    np.asarray(q),
                    allow_start=_wrapped_distance(np.asarray(q), q_start) <= 1e-8,
                    allow_goal=_wrapped_distance(np.asarray(q), q_goal) <= 1e-8,
                )[0],
                is_segment_valid=lambda first, second: all(
                    _path_valid(
                        world,
                        model,
                        task,
                        sample,
                        np.asarray(q),
                        allow_start=index == 0,
                        allow_goal=index == len(segment) - 1,
                    )[0]
                    for segment in [np.linspace(first, second, max(2, int(np.linalg.norm(second - first) / config.rrt_edge_resolution_rad) + 1))]
                    for index, q in enumerate(segment)
                ),
            )
            stats = planner.last_stats
            rrt_stats.append({"seed": planner.random_seed, "iterations": stats.iterations, "tree_nodes": stats.tree_nodes, "success": path is not None})
            if path is not None:
                q_path = np.asarray(path, dtype=float)
                path_clearance = float("inf")
                for index, q in enumerate(q_path):
                    _, clearance, _ = _path_valid(
                        world,
                        model,
                        task,
                        sample,
                        q,
                        allow_start=index == 0,
                        allow_goal=index == len(q_path) - 1,
                    )
                    path_clearance = min(path_clearance, clearance)
                artifact = _save_trajectory(trajectory_dir, task, q_start, q_goal, q_path, "RRT-Connect", path_clearance, time.perf_counter() - started)
                return {"straight_success": False, "rrt_success": True, "failure": failure, "rrt_stats": rrt_stats, "artifact": artifact}
        return {"straight_success": False, "rrt_success": False, "failure": failure, "rrt_stats": rrt_stats, "artifact": None}


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _format_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return ">= query window"
    return f"{number:.{digits}g}"


def _write_report(
    path: Path,
    *,
    calibration: ProxyCalibration,
    dimension_rows: list[dict[str, object]],
    baseline: dict[str, object],
    goal_audit: dict[str, object],
    searches: list[dict[str, object]],
    plot_paths: list[str],
    q_goal_dir: Path,
    trajectory_dir: Path,
) -> None:
    any_endpoint = any(int(row["collision_free_endpoint_count"]) > 0 for row in searches)
    any_trajectory = any(row.get("trajectory", {}).get("straight_success") or row.get("trajectory", {}).get("rrt_success") for row in searches)
    if any_trajectory:
        conclusion = "CASE A：至少一个 position+normal 合法终点完成了无碰撞轨迹。"
    elif any_endpoint:
        conclusion = "CASE B：至少一个合法终点存在，但本轮轨迹规划未完成。"
    elif baseline["status"] != "PASS":
        conclusion = "CASE D：校准代理未通过 Baseline sanity check，不能把终点碰撞解释为机构结论。"
    else:
        conclusion = "CASE C/E：校准 proxy 下本轮多分支仍未得到合法终点；这是固定任务的 configuration-space 证据，但由于没有独立电机 CAD，仍需保留 proxy 不确定性。"
    lines = [
        "# Collision Proxy Calibration and Collision-Aware IK",
        "",
        "## 1. Background",
        "",
        "本报告固定 `OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`，不重新优化杆长、轴架构或整塔路径。旧 `65 mm` 统一 link radius 来自人工默认值；同时，单个 position+normal q_goal 碰撞不能证明同一目标没有其他 IK branch。",
        "",
        "## 2. Baseline Collision Audit",
        "",
        f"Baseline sanity status: `{baseline['status']}`。真实 URDF 非相邻 self-collision counts: `{baseline['actual_counts']}`；校准 proxy counts: `{baseline['proxy_counts']}`。",
        "",
        "| state | real URDF bad pairs | calibrated proxy bad pairs |",
        "|---|---:|---:|",
    ]
    for actual, proxy in zip(baseline["actual_urdf"], baseline["calibrated_proxy"]):
        lines.append(f"| {actual['state']} | {actual['non_adjacent_self_collision_count']} | {proxy['non_adjacent_self_collision_count']} |")
    lines += [
        "",
        "左端 terminal capsule 已从碰撞代理显示/碰撞体中移除，因为 `base_link.STL` 已经代表该物理末端结构；不改变 FK 或吸盘中心。",
        "",
        "## 3. Real Geometry Measurement",
        "",
        "连杆主轴由 STL PCA 最大特征方向确定；连杆 raw radius = PCA 横向两个 extent 的最大值的一半。关节没有独立电机 CAD，因此先测量 child-link 原点 `55 mm` 邻域内顶点距离的 `99th percentile`，再用真实基线关节间距审计得到的 `40 mm raw` 上限作为独立 joint sphere。该值是可解释的 audit envelope，不冒充电机 CAD；最终电机定型仍需独立外壳模型。",
        "",
        "| mesh | axial length (m) | cross width (m) | cross height (m) | equivalent radius (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("base_link", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"):
        measurement = calibration.mesh_measurements[name]
        lines.append(f"| {name}.STL | {_format_float(measurement.axial_length_m)} | {_format_float(measurement.cross_section_width_m)} | {_format_float(measurement.cross_section_height_m)} | {_format_float(measurement.equivalent_radius_m)} |")
    lines += [
        "",
        "## 4. Final Collision Proxy Definition",
        "",
        "- central body：真实 `L4.STL` mesh；",
        "- link body：Capsule，使用每一类镜像 STL 的 per-link radius，杆长随当前 MorphologySpec 变化；",
        "- joint/motor：独立 Sphere，使用局部 STL 邻域 profile；",
        "- endpoint/suction：`base_link.STL` 与 `L8.STL` 真实 mesh；",
        "- safety inflation：nominal `+5 mm`，另测试 `+10 mm` 和 `+15 mm` 总膨胀；",
        "",
        "| component | source | nominal radius (m) | effective radius (m) |",
        "|---|---|---:|---:|",
    ]
    for row in dimension_rows:
        lines.append(f"| {row['component']} | {row['source_geometry']} | {_format_float(row['nominal_radius_m'])} | {_format_float(row['effective_radius_m'])} |")
    lines += [
        "",
        "## 5. Collision Model Visualization",
        "",
    ]
    for plot_path in plot_paths:
        lines.append(f"- `{plot_path}`")
    lines += [
        "",
        "## 6. Existing Four q_goal Re-evaluation",
        "",
        f"已读取 `{q_goal_dir}` 中保存的四个 q_goal；没有重新 IK。",
        "",
        "| variant | collision pass | task 0 | task 1 | task 2 | task 3 |",
        "|---|---:|---|---|---|---|",
    ]
    for variant in goal_audit["variants"]:
        by_task = {row["task_id"]: row for row in variant["tasks"]}
        cells = []
        for task in sorted(by_task):
            row = by_task[task]
            cells.append(f"{row['critical_pair'] or 'PASS'} ({_format_float(row['minimum_clearance_m'])} m)")
        lines.append(f"| {variant['label']} | {variant['collision_pass_count']}/4 | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 7. Collision-Aware IK Search Method",
        "",
        "每个固定目标执行 yaw-conditioned full-pose seeds 与 normal-only seeds；所有候选最终只按正式 `position <= 5 mm`、`normal <= 3 deg`、joint limits 和 support constraint 判定。通过 position+normal 的候选先去重，再逐个进入真实 Tower + self collision checker；合法终点按 minimum clearance 和 joint-limit margin 排序。若普通 multi-start 无合法终点，最多对最接近的若干 branch 做有限的 collision-aware L-BFGS-B 局部 refinement。报告中的 `0.05 m` endpoint clearance 表示在 `50 mm` 近距离查询窗口内没有更近的非接触 pair，即 `>= 50 mm` 的下界，不是无限远。",
        "",
    ]
    for index, result in enumerate(searches):
        ik = result["ik"]
        best = result.get("best_collision_free")
        lines += [
            f"## {8 + index}. Task {index} Results",
            "",
            f"- task: `{result['task_id']}`",
            f"- target xyz: `{result['target_xyz_m']}`",
            f"- target normal: `{result['target_normal']}`",
            f"- IK attempts: `{ik['raw_ik_attempts']}`",
            f"- position-normal converged: `{ik['position_normal_converged_count']}`",
            f"- unique IK branches: `{ik['unique_ik_branch_count']}`",
            f"- collision-free endpoints: `{result['collision_free_endpoint_count']}`",
            f"- dominant rejected pairs: `{result['collision_pair_counts']}`",
            f"- refinement attempts: `{result['refinement_attempts']}`",
        ]
        if best is None:
            lines.append("- best collision-free q: `none`")
        else:
            lines += [
                f"- best collision-free clearance: `{_format_float(best['collision']['minimum_clearance_m'])} m`",
                f"- best joint-limit margin: `{_format_float(best['joint_limit_margin_rad'])} rad`",
                f"- best q: `{np.asarray(best['q']).tolist()}`",
            ]
        lines.append(f"- collision-free q_goal artifact: `{result.get('collision_free_q_goal_artifact')}`")
        trajectory = result.get("trajectory", {})
        lines += [
            f"- Straight: `{trajectory.get('straight_success', False)}`",
            f"- RRT-Connect: `{trajectory.get('rrt_success', False)}`",
            f"- trajectory artifact: `{trajectory.get('artifact')}`",
            "",
        ]
    lines += [
        "## 12. What the 8R Redundancy Actually Provides",
        "",
        "本轮把同一个目标的多组 position+normal IK branch 分开统计。若 branch 数量大于1但 collision-free endpoint 为0，则冗余在运动学层存在，但在当前固定 Tower/central-body/proxy 几何下没有转化为合法终点；若有合法 endpoint，则其 clearance 和碰撞 pair 记录说明冗余实际绕开了什么。",
        "",
        "## 13. Current Conclusion",
        "",
        conclusion,
        "",
        "## 14. Generated Files",
        "",
        f"- `{path}`（本轮完整独立报告）",
        f"- `{q_goal_dir}`（四个历史 q_goal 输入目录）",
        f"- `{trajectory_dir.parent / 'collision_free_q_goals'}`（本轮多分支筛选得到的合法 endpoint）",
        f"- `{trajectory_dir}`（仅在轨迹成功时生成）",
        "- `models/design_results/collision_proxy_dimensions.csv`（碰撞代理尺寸表）",
        "- `environment/design/proxy_calibration.py`（真实 STL 尺寸标定模块）",
        "- `environment/design/collision_aware_ik.py`（固定候选多分支 IK 与审计脚本）",
        "- `environment/design/plot_collision_model.py`（碰撞体 PNG/PyBullet 可视化脚本）",
        "",
        "## 15. Recommended Next Step",
        "",
        "先检查碰撞体 PNG 和本报告中的 branch/pair 统计；在确认 proxy 尺寸和 endpoint 搜索结论后，再决定是否进入局部机构设计修改。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(paths: ProjectPaths, *, config: SearchConfig | None = None) -> dict[str, object]:
    config = config or SearchConfig()
    paths.validate_required_files()
    geometry = BaselineGeometry.from_project(paths)
    calibration = calibrate_proxy_dimensions(geometry.central_mesh_path.parent)
    dimension_path = paths.repo_root / "models" / "design_results" / "collision_proxy_dimensions.csv"
    dimension_rows = _write_dimensions_csv(dimension_path, calibration, NOMINAL_INFLATION_M)
    baseline = _baseline_sanity(paths)
    fixed_spec = fixed_axis8_spec(geometry)
    model = MorphologyModel(fixed_spec)
    tasks = build_task_suite(paths, max_targets_per_task=1)
    q_goal_dir = paths.repo_root / "models" / "design_results" / "q_goals"
    goals = _load_saved_q_goals(q_goal_dir)
    goal_audit = _audit_saved_goals(paths, fixed_spec, goals, tasks)
    searches: list[dict[str, object]] = []
    trajectory_dir = paths.repo_root / "models" / "design_results" / "trajectories"
    collision_free_goal_dir = paths.repo_root / "models" / "design_results" / "collision_free_q_goals"
    for task_index, task in enumerate(tasks):
        result = _collision_aware_task_search(paths, task, model, goals, config)
        best = result.get("best_collision_free")
        if best is not None:
            result["collision_free_q_goal_artifact"] = _save_collision_free_q_goal(
                collision_free_goal_dir,
                task_index,
                task,
                task.targets[0],
                best,
                model,
            )
            result["trajectory"] = _plan_if_endpoint_valid(paths, task, task.targets[0], model, best, config, trajectory_dir)
        else:
            result["trajectory"] = {"straight_success": False, "rrt_success": False, "artifact": None, "reason": "没有合法终点，按要求未运行 Straight/RRT"}
        searches.append(result)
        print(
            task.task_id,
            "attempts", result["ik"]["raw_ik_attempts"],
            "unique", result["ik"]["unique_ik_branch_count"],
            "collision_free", result["collision_free_endpoint_count"],
            "best", None if best is None else best["collision"]["minimum_clearance_m"],
        )
    plot_dir = paths.repo_root / "models" / "design_results" / "plots"
    plot_paths = [
        str(plot_dir / "collision_model_optimized_8r_isometric.png"),
        str(plot_dir / "collision_model_optimized_8r_side.png"),
    ]
    report_path = paths.repo_root / "docs" / "collision_proxy_and_collision_aware_ik.md"
    report = {
        "design_name": fixed_spec.name,
        "link_lengths_m": fixed_spec.link_lengths_m.tolist(),
        "link_proxy_radii_m": fixed_spec.link_proxy_radii_m.tolist() if fixed_spec.link_proxy_radii_m is not None else None,
        "joint_proxy_radii_m": fixed_spec.joint_proxy_radii_m.tolist() if fixed_spec.joint_proxy_radii_m is not None else None,
        "collision_inflation_m": NOMINAL_INFLATION_M,
        "calibration": {
            "mesh_measurements": {name: item.as_dict() for name, item in calibration.mesh_measurements.items()},
            "joint_measurements": {name: item.as_dict() for name, item in calibration.joint_measurements.items()},
        },
        "baseline": baseline,
        "saved_q_goal_audit": goal_audit,
        "searches": searches,
        "generated": {
            "dimension_csv": str(dimension_path),
            "plot_paths": plot_paths,
            "report_path": str(report_path),
            "q_goal_dir": str(q_goal_dir),
            "collision_free_q_goal_dir": str(collision_free_goal_dir),
            "trajectory_dir": str(trajectory_dir),
        },
        "search_config": config.__dict__,
    }
    diagnostic_path = paths.repo_root / "models" / "design_results" / "diagnostics" / "collision_proxy_and_collision_aware_ik.json"
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(
        report_path,
        calibration=calibration,
        dimension_rows=dimension_rows,
        baseline=baseline,
        goal_audit=goal_audit,
        searches=searches,
        plot_paths=plot_paths,
        q_goal_dir=q_goal_dir,
        trajectory_dir=trajectory_dir,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--yaw-samples", type=int, default=DEFAULT_YAW_SAMPLES)
    parser.add_argument("--seeds-per-yaw", type=int, default=DEFAULT_SEEDS_PER_YAW)
    parser.add_argument("--normal-only-seeds", type=int, default=DEFAULT_NORMAL_ONLY_SEEDS)
    args = parser.parse_args()
    config = SearchConfig(
        yaw_samples=max(1, args.yaw_samples),
        seeds_per_yaw=max(1, args.seeds_per_yaw),
        normal_only_seeds=max(1, args.normal_only_seeds),
    )
    report = run(ProjectPaths.from_repo_root(args.repo_root), config=config)
    print("completed:", report["generated"]["report_path"])


if __name__ == "__main__":
    main()
