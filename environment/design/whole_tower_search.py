"""Coarse-to-fine contact-state search for continuous tower climbing.

This module evaluates a fixed morphology from the validated Step 2 region.  A
contact state stores both attached feet, the support suction pose, and the
current joint configuration.  Every edge therefore means one physical move:
the current moving foot leaves its contact, reaches a new attach-line node,
and becomes the next support after rebase.

The search is deliberately separate from the morphology optimizer.  It reuses
the existing parameterized FK, collision proxy, IK evaluator, and
RRT-Connect implementation, but does not change link lengths or axes.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environment.attach_lines import AttachLineSample
from environment.design.collision_audit import fixed_axis8_spec
from environment.design.collision_aware_ik import (
    SearchConfig,
    _collision_filter,
    _run_ik_search_for_task,
)
from environment.design.contact_graph import ContactNode, coarse_contact_nodes
from environment.design.collision_proxy import MorphologyCollisionWorld
from environment.design.evaluator import DesignEvaluationSettings, MorphologyTaskEvaluator
from environment.design.morphology import (
    BaselineGeometry,
    MorphologyModel,
    MorphologySpec,
)
from environment.design.task_suite import TaskSpec
from environment.paths import ProjectPaths
from environment.rrt_connect import RRTConnect
from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrameSet
from environment.trajectory_io import SavedTrajectory
from environment.transforms import RigidTransform, angle_between_vectors_rad, normalize


SEARCH_CHECKPOINT = "whole_tower_search_checkpoint.json"


@dataclass(frozen=True)
class WholeTowerConfig:
    # The real line endpoints are always retained by sample_uniform; 1 m is
    # sufficient for the verified maximum vertical gap (about 1.09 m) while
    # keeping the global contact graph tractable.
    coarse_spacing_m: float = 1.0
    minimum_progress_m: float = 0.02
    maximum_step_height_m: float = 1.20
    goal_tolerance_m: float = 0.02
    candidates_per_surface: int = 1
    max_expansions: int = 300
    seed_count: int = 3
    fallback_seed_count: int = 8
    high_confidence_seed_count: int = 32
    high_confidence_height_m: float = 36.0
    yaw_samples: int = 1
    local_max_nfev: int = 120
    normal_max_nfev: int = 220
    rrt_max_iterations: int = 1200
    rrt_seed_count: int = 2
    rrt_step_size_rad: float = 0.2
    rrt_edge_resolution_rad: float = 0.08
    random_seed: int = 20260817
    position_tolerance_m: float = 0.005
    normal_tolerance_deg: float = 3.0
    allowed_contact_radius_m: float = 0.09
    run_fine_trajectories: bool = True
    max_fine_routes: int = 20


@dataclass(frozen=True)
class ClimbState:
    state_id: str
    depth: int
    support_endpoint: str
    support_surface: str
    support_node_id: str
    support_sample: AttachLineSample
    support_pose: np.ndarray
    moving_endpoint: str
    moving_surface: str
    moving_node_id: str
    moving_sample: AttachLineSample
    q: np.ndarray
    height_m: float
    route_edge_ids: tuple[str, ...] = ()


def _pose_matrix(pose: RigidTransform) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = pose.rotation_matrix
    result[:3, 3] = pose.position
    return result


def _rigid_pose(matrix: np.ndarray) -> RigidTransform:
    value = np.asarray(matrix, dtype=float)
    return RigidTransform.from_rotation_matrix(value[:3, 3], value[:3, :3])


def _sample_dict(sample: AttachLineSample) -> dict[str, object]:
    return {
        "segment_id": int(sample.segment_id),
        "s_m": float(sample.s_m),
        "xyz_m": np.asarray(sample.xyz_m, dtype=float).tolist(),
        "normal": np.asarray(sample.normal, dtype=float).tolist(),
        "uv_m": np.asarray(sample.uv_m, dtype=float).tolist(),
    }


def _sample_from_dict(value: dict[str, object]) -> AttachLineSample:
    return AttachLineSample(
        segment_id=int(value["segment_id"]),
        s_m=float(value["s_m"]),
        xyz_m=np.asarray(value["xyz_m"], dtype=float),
        normal=normalize(np.asarray(value["normal"], dtype=float), name="contact normal"),
        uv_m=np.asarray(value["uv_m"], dtype=float),
    )


def _node_dict(node: ContactNode) -> dict[str, object]:
    return {
        "node_id": node.node_id,
        "surface_name": node.surface_name,
        "segment_id": int(node.segment_id),
        "s_m": float(node.s_m),
        "sample": _sample_dict(node.sample),
    }


def _state_dict(state: ClimbState) -> dict[str, object]:
    return {
        "state_id": state.state_id,
        "depth": int(state.depth),
        "support_endpoint": state.support_endpoint,
        "support_surface": state.support_surface,
        "support_node_id": state.support_node_id,
        "support_sample": _sample_dict(state.support_sample),
        "support_pose": np.asarray(state.support_pose, dtype=float).tolist(),
        "moving_endpoint": state.moving_endpoint,
        "moving_surface": state.moving_surface,
        "moving_node_id": state.moving_node_id,
        "moving_sample": _sample_dict(state.moving_sample),
        "q": np.asarray(state.q, dtype=float).tolist(),
        "height_m": float(state.height_m),
        "route_edge_ids": list(state.route_edge_ids),
    }


def _state_from_dict(value: dict[str, object]) -> ClimbState:
    return ClimbState(
        state_id=str(value["state_id"]),
        depth=int(value["depth"]),
        support_endpoint=str(value["support_endpoint"]),
        support_surface=str(value["support_surface"]),
        support_node_id=str(value["support_node_id"]),
        support_sample=_sample_from_dict(value["support_sample"]),
        support_pose=np.asarray(value["support_pose"], dtype=float),
        moving_endpoint=str(value["moving_endpoint"]),
        moving_surface=str(value["moving_surface"]),
        moving_node_id=str(value["moving_node_id"]),
        moving_sample=_sample_from_dict(value["moving_sample"]),
        q=np.asarray(value["q"], dtype=float),
        height_m=float(value["height_m"]),
        route_edge_ids=tuple(str(item) for item in value.get("route_edge_ids", [])),
    )


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _nearest_node(nodes: tuple[ContactNode, ...], point: np.ndarray) -> ContactNode:
    if not nodes:
        raise ValueError("附着线没有可用节点")
    target = np.asarray(point, dtype=float)
    return min(nodes, key=lambda node: float(np.linalg.norm(node.sample.xyz_m - target)))


def _load_nodes(paths: ProjectPaths, config: WholeTowerConfig) -> dict[str, tuple[ContactNode, ...]]:
    from environment.attach_lines import AttachLineSet

    nodes: dict[str, tuple[ContactNode, ...]] = {}
    for surface in ("surface1", "surface2"):
        lines = AttachLineSet.load_npz(
            paths.attach_lines_for_surface_npz(surface),
            expected_surface_name=surface,
        )
        nodes[surface] = coarse_contact_nodes(
            lines,
            spacing_m=config.coarse_spacing_m,
        )
    return nodes


def _baseline_world_state(paths: ProjectPaths) -> tuple[np.ndarray, RigidTransform, RigidTransform, np.ndarray]:
    """Load the actual Step 2 poses used to seed a new morphology."""

    saved = SavedTrajectory.load(paths.successful_step2_trajectory_npz)
    if saved.base_position_m is None or saved.base_orientation_xyzw is None:
        raise ValueError("Step 2轨迹缺少base pose，无法建立全塔初始状态")
    frames = SuctionFrameSet.load(paths.suction_config)
    base_pose = RigidTransform(
        position=saved.base_position_m[-1],
        quaternion_xyzw=saved.base_orientation_xyzw[-1],
    )
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_robot(base_pose)
        scene.reset_joints(saved.trajectory_rad[-1])
        support_pose = scene.get_suction_pose(frames.l8_end)
        moving_pose = scene.get_suction_pose(frames.base_end)
    geometry = BaselineGeometry.from_project(paths)
    return (
        geometry.baseline_joint_vector(np.asarray(saved.trajectory_rad[-1], dtype=float)),
        support_pose,
        moving_pose,
        np.asarray(saved.trajectory_rad[-1], dtype=float),
    )


def _model_initial_state(
    paths: ProjectPaths,
    model: MorphologyModel,
    nodes: dict[str, tuple[ContactNode, ...]],
    config: WholeTowerConfig,
) -> tuple[ClimbState | None, dict[str, object]]:
    """Find a valid design-specific configuration at the existing Step 2 contacts."""

    baseline_q, support_pose, moving_pose, _ = _baseline_world_state(paths)
    if model.spec.per_side_dof == 4:
        q_seed = baseline_q.copy()
    else:
        q_seed = np.concatenate(
            (
                baseline_q[:4][list(model.spec.left_active_indices)],
                baseline_q[4:][list(model.spec.right_active_indices)],
            )
        )
    support_node = _nearest_node(nodes["surface2"], support_pose.position)
    moving_node = _nearest_node(nodes["surface1"], moving_pose.position)
    task = TaskSpec(
        task_id="initial_contact_state",
        label="Step 2 initial contact state",
        support_endpoint="l8_end",
        support_surface="surface2",
        moving_endpoint="base_end",
        target_surface="surface1",
        support_pose=support_pose,
        support_xyz=support_pose.position.copy(),
        support_normal=normalize(support_pose.z_axis, name="initial support normal"),
        start_q=q_seed.copy(),
        moving_start_xyz=moving_pose.position.copy(),
        moving_start_z=float(moving_pose.position[2]),
        targets=(moving_node.sample,),
        critical=True,
    )
    evaluator = MorphologyTaskEvaluator(
        model,
        settings=DesignEvaluationSettings(
            seed_count=max(12, config.seed_count * 4),
            yaw_samples=max(1, config.yaw_samples),
            local_max_nfev=max(config.local_max_nfev, 180),
            normal_max_nfev=max(config.normal_max_nfev, 400),
            random_seed=config.random_seed,
            run_collision=True,
            run_trajectory=False,
            position_tolerance_m=config.position_tolerance_m,
            normal_tolerance_deg=config.normal_tolerance_deg,
        ),
    )
    result = evaluator.evaluate_task(task, target_limit=1, collision=True, trajectory=False)
    if not result.target_results:
        return None, {"status": "NO_INITIAL_TARGET", "failure_type": "NO_TARGET"}
    target = result.target_results[0]
    diagnostics = {
        "status": "PASS" if target.goal_valid else "FAIL",
        "position_error_m": float(target.normal_best.position_error_m),
        "normal_error_deg": float(target.normal_best.normal_error_deg),
        "goal_valid": bool(target.goal_valid),
        "minimum_clearance_m": target.minimum_clearance_m,
        "critical_pair": target.goal_collision.critical_link if target.goal_collision else None,
        "failure_type": target.failure_type,
        "q": np.asarray(target.normal_best.q, dtype=float).tolist(),
        "support_node": _node_dict(support_node),
        "moving_node": _node_dict(moving_node),
    }
    if not target.goal_valid:
        return None, diagnostics
    q = np.asarray(target.normal_best.q, dtype=float)
    model_state = model.world_state_for_support(q, "l8_end", _pose_matrix(support_pose))
    moving_world_pose = model_state.suction_pose("base_end")
    return (
        ClimbState(
            state_id="state_0000",
            depth=0,
            support_endpoint="l8_end",
            support_surface="surface2",
            support_node_id=support_node.node_id,
            support_sample=support_node.sample,
            support_pose=_pose_matrix(support_pose),
            moving_endpoint="base_end",
            moving_surface="surface1",
            moving_node_id=moving_node.node_id,
            moving_sample=moving_node.sample,
            q=q,
            height_m=float(max(support_node.sample.xyz_m[2], moving_node.sample.xyz_m[2])),
        ),
        diagnostics,
    )


def _dynamic_task(
    model: MorphologyModel,
    state: ClimbState,
    target: ContactNode,
) -> TaskSpec:
    current = model.world_state_for_support(
        state.q,
        state.support_endpoint,
        state.support_pose,
    )
    moving_pose = current.suction_pose(state.moving_endpoint)
    return TaskSpec(
        task_id=f"{state.state_id}_to_{target.node_id}",
        label="dynamic contact-state edge",
        support_endpoint=state.support_endpoint,
        support_surface=state.support_surface,
        moving_endpoint=state.moving_endpoint,
        target_surface=target.surface_name,
        support_pose=_rigid_pose(state.support_pose),
        support_xyz=state.support_sample.xyz_m.copy(),
        support_normal=normalize(state.support_sample.normal, name="support surface normal"),
        start_q=state.q.copy(),
        moving_start_xyz=moving_pose[:3, 3].copy(),
        moving_start_z=float(moving_pose[2, 3]),
        targets=(target.sample,),
        critical=bool(abs(float(target.sample.xyz_m[2]) - 1.122) < 0.02),
    )


def _candidate_targets(
    state: ClimbState,
    nodes: dict[str, tuple[ContactNode, ...]],
    config: WholeTowerConfig,
) -> list[ContactNode]:
    current_height = max(
        float(state.support_sample.xyz_m[2]),
        float(state.moving_sample.xyz_m[2]),
    )
    lower = current_height + config.minimum_progress_m
    upper = current_height + config.maximum_step_height_m
    selected: list[ContactNode] = []
    for surface in ("surface1", "surface2"):
        valid = [
            node
            for node in nodes[surface]
            if lower < float(node.sample.xyz_m[2]) <= upper
            and not (
                surface == state.support_surface
                and np.linalg.norm(node.sample.xyz_m - state.support_sample.xyz_m) < 0.124
            )
        ]
        valid.sort(key=lambda node: (float(node.sample.xyz_m[2]), node.segment_id, node.s_m))
        selected.extend(valid[: config.candidates_per_surface])
    selected.sort(key=lambda node: (float(node.sample.xyz_m[2]), node.surface_name, node.segment_id, node.s_m))
    return selected


def _edge_key(state: ClimbState, target: ContactNode) -> str:
    return f"{state.state_id}|{state.moving_endpoint}|{target.node_id}"


def _new_state(
    model: MorphologyModel,
    state: ClimbState,
    target: ContactNode,
    q_goal: np.ndarray,
    edge_id: str,
) -> ClimbState:
    old_state = model.world_state_for_support(
        q_goal,
        state.support_endpoint,
        state.support_pose,
    )
    target_pose = old_state.suction_pose(state.moving_endpoint)
    return ClimbState(
        state_id=f"state_{state.depth + 1:04d}_{target.node_id.replace(':', '_')}",
        depth=state.depth + 1,
        support_endpoint=state.moving_endpoint,
        support_surface=target.surface_name,
        support_node_id=target.node_id,
        support_sample=target.sample,
        support_pose=target_pose.copy(),
        moving_endpoint=state.support_endpoint,
        moving_surface=state.support_surface,
        moving_node_id=state.support_node_id,
        moving_sample=state.support_sample,
        q=np.asarray(q_goal, dtype=float).copy(),
        height_m=float(max(target.sample.xyz_m[2], state.support_sample.xyz_m[2])),
        route_edge_ids=state.route_edge_ids + (edge_id,),
    )


def _endpoint_edge(
    model: MorphologyModel,
    evaluator: MorphologyTaskEvaluator,
    state: ClimbState,
    target: ContactNode,
    config: WholeTowerConfig,
) -> tuple[dict[str, object], ClimbState | None]:
    task = _dynamic_task(model, state, target)
    result = evaluator.evaluate_task(task, target_limit=1, collision=True, trajectory=False)
    # The coarse pass is intentionally cheap.  A target that has a formal
    # position+normal solution but no legal endpoint gets an independent
    # branch pass using the same seed budget that established the local
    # multi-branch result.  This avoids treating one colliding IK branch as a
    # configuration-space proof while keeping the normal case inexpensive.
    if result.target_results:
        first = result.target_results[0]
        fallback_count = (
            config.high_confidence_seed_count
            if float(target.sample.xyz_m[2]) <= config.high_confidence_height_m
            else config.fallback_seed_count
        )
        if first.normal_best.success and not first.goal_valid and fallback_count > config.seed_count:
            fallback = MorphologyTaskEvaluator(
                model,
                settings=DesignEvaluationSettings(
                    position_tolerance_m=config.position_tolerance_m,
                    normal_tolerance_deg=config.normal_tolerance_deg,
                    local_max_nfev=max(config.local_max_nfev, 140),
                    normal_max_nfev=max(config.normal_max_nfev, 400),
                    seed_count=fallback_count,
                    yaw_samples=max(1, config.yaw_samples),
                    random_seed=config.random_seed + 1009,
                    run_collision=True,
                    run_trajectory=False,
                ),
            )
            result = fallback.evaluate_task(task, target_limit=1, collision=True, trajectory=False)
    edge_id = _edge_key(state, target)
    if not result.target_results:
        return ({"edge_id": edge_id, "status": "FAILED", "failure_type": "NO_TARGET"}, None)
    evaluated = result.target_results[0]
    row: dict[str, object] = {
        "edge_id": edge_id,
        "source_state_id": state.state_id,
        "target_node": _node_dict(target),
        "support_endpoint": state.support_endpoint,
        "moving_endpoint": state.moving_endpoint,
        "position_error_m": float(evaluated.normal_best.position_error_m),
        "normal_error_deg": float(evaluated.normal_best.normal_error_deg),
        "endpoint_valid": bool(evaluated.goal_valid),
        "failure_type": evaluated.failure_type,
        "minimum_clearance_m": evaluated.minimum_clearance_m,
        "critical_pair": evaluated.goal_collision.critical_link if evaluated.goal_collision else None,
        "critical_kind": evaluated.goal_collision.kind if evaluated.goal_collision else None,
        "q_goal": np.asarray(evaluated.normal_best.q, dtype=float).tolist(),
    }
    if not evaluated.goal_valid:
        return row, None
    successor = _new_state(model, state, target, evaluated.normal_best.q, edge_id)
    row["successor_state_id"] = successor.state_id
    return row, successor


def _allowed_positions(task: TaskSpec, sample: AttachLineSample, *, allow_start: bool, allow_goal: bool) -> dict[str, np.ndarray]:
    positions = {task.support_endpoint: np.asarray(task.support_xyz, dtype=float)}
    if allow_start:
        positions[task.moving_endpoint] = np.asarray(task.moving_start_xyz, dtype=float)
    if allow_goal:
        positions[task.moving_endpoint] = np.asarray(sample.xyz_m, dtype=float)
    return positions


def _state_valid_for_edge(
    world: MorphologyCollisionWorld,
    model: MorphologyModel,
    task: TaskSpec,
    sample: AttachLineSample,
    q: np.ndarray,
    *,
    allow_start: bool,
    allow_goal: bool,
    config: WholeTowerConfig,
) -> tuple[bool, float, str | None]:
    state = model.world_state_for_support(q, task.support_endpoint, _pose_matrix(task.support_pose))
    moving = state.suction_pose(task.moving_endpoint)
    if allow_start and float(np.linalg.norm(moving[:3, 3] - task.moving_start_xyz)) > config.position_tolerance_m:
        return False, -float("inf"), "MOVING_START_DRIFT"
    if allow_goal:
        if float(np.linalg.norm(moving[:3, 3] - sample.xyz_m)) > config.position_tolerance_m:
            return False, -float("inf"), "MOVING_GOAL_DRIFT"
        normal_error = float(np.degrees(angle_between_vectors_rad(moving[:3, 2], -sample.normal)))
        if normal_error > config.normal_tolerance_deg:
            return False, -float("inf"), "MOVING_GOAL_NORMAL"
    world.update(state)
    report = world.check(
        allowed_endpoint_positions=_allowed_positions(task, sample, allow_start=allow_start, allow_goal=allow_goal),
        allowed_contact_radius_m=config.allowed_contact_radius_m,
    )
    return bool(report.ok), float(report.minimum_clearance_m), report.critical_link or report.kind


def _fine_validate_edge(
    paths: ProjectPaths,
    model: MorphologyModel,
    state: ClimbState,
    target: ContactNode,
    q_goal: np.ndarray,
    config: WholeTowerConfig,
    output_dir: Path,
    edge_index: int,
) -> dict[str, object]:
    task = _dynamic_task(model, state, target)
    q_start = np.asarray(state.q, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    started = time.perf_counter()
    with MorphologyCollisionWorld(paths, model, gui=False) as world:
        straight = np.linspace(q_start, q_goal, 40)
        minimum = float("inf")
        straight_ok = True
        failure = None
        for index, q in enumerate(straight):
            ok, clearance, critical = _state_valid_for_edge(
                world, model, task, target.sample, q,
                allow_start=index == 0,
                allow_goal=index == len(straight) - 1,
                config=config,
            )
            minimum = min(minimum, clearance)
            if not ok:
                straight_ok = False
                failure = critical
                break
        path = straight if straight_ok else None
        method = "Straight" if straight_ok else ""
        rrt_stats: list[dict[str, object]] = []
        if path is None:
            for seed_index in range(config.rrt_seed_count):
                planner = RRTConnect(
                    model.spec.lower_limits,
                    model.spec.upper_limits,
                    step_size_rad=config.rrt_step_size_rad,
                    max_iterations=config.rrt_max_iterations,
                    edge_resolution_rad=config.rrt_edge_resolution_rad,
                    random_seed=config.random_seed + seed_index,
                )

                def valid(q: np.ndarray) -> bool:
                    is_start = float(np.linalg.norm(np.asarray(q) - q_start)) <= 1e-8
                    is_goal = float(np.linalg.norm(np.asarray(q) - q_goal)) <= 1e-8
                    return _state_valid_for_edge(
                        world, model, task, target.sample, np.asarray(q),
                        allow_start=is_start,
                        allow_goal=is_goal,
                        config=config,
                    )[0]

                def valid_segment(first: np.ndarray, second: np.ndarray) -> bool:
                    count = max(2, int(np.linalg.norm(second - first) / config.rrt_edge_resolution_rad) + 1)
                    segment = np.linspace(first, second, count)
                    return all(
                        _state_valid_for_edge(
                            world, model, task, target.sample, q,
                            allow_start=index == 0,
                            allow_goal=index == len(segment) - 1,
                            config=config,
                        )[0]
                        for index, q in enumerate(segment)
                    )

                planned = planner.plan(
                    q_start,
                    q_goal,
                    is_state_valid=valid,
                    is_segment_valid=valid_segment,
                )
                stats = planner.last_stats
                rrt_stats.append({
                    "seed": planner.random_seed,
                    "iterations": stats.iterations,
                    "tree_nodes": stats.tree_nodes,
                    "success": planned is not None,
                    "start_valid": stats.start_valid,
                    "goal_valid": stats.goal_valid,
                })
                if planned is not None:
                    path = np.asarray(planned, dtype=float)
                    method = "RRT-Connect"
                    break
        if path is not None:
            minimum = float("inf")
            for index, q in enumerate(path):
                ok, clearance, critical = _state_valid_for_edge(
                    world, model, task, target.sample, q,
                    allow_start=index == 0,
                    allow_goal=index == len(path) - 1,
                    config=config,
                )
                minimum = min(minimum, clearance)
                if not ok:
                    path = None
                    failure = critical
                    method = ""
                    break
        success = path is not None
        artifact = None
        if success:
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = f"edge_{edge_index:04d}_{state.state_id}_to_{target.node_id.replace(':', '_')}"
            path_npz = output_dir / f"{stem}.npz"
            np.savez_compressed(
                path_npz,
                q_start=q_start,
                q_goal=q_goal,
                q_path=np.asarray(path, dtype=float),
                method=np.asarray(method),
                source_state_id=np.asarray(state.state_id),
                target_node_id=np.asarray(target.node_id),
                target_surface=np.asarray(target.surface_name),
                minimum_clearance_m=np.asarray(minimum),
            )
            artifact = str(path_npz)
        return {
            "success": success,
            "method": method or None,
            "failure_type": None if success else failure or "NO_COLLISION_FREE_PATH",
            "minimum_trajectory_clearance_m": minimum,
            "rrt_stats": rrt_stats,
            "trajectory_artifact": artifact,
            "planning_time_s": time.perf_counter() - started,
            "q_path_count": 0 if path is None else int(len(path)),
        }


def _branch_rescue_candidates(
    model: MorphologyModel,
    state: ClimbState,
    target: ContactNode,
    config: WholeTowerConfig,
    *,
    expanded: bool = False,
) -> list[dict[str, object]]:
    """Run a branch search for a path-blocked edge.

    The normal rescue pass stays at the established 96 attempts.  An edge
    whose legal branches all fail trajectory validation can request the
    expanded 320-attempt pass locally; this is still endpoint IK coverage,
    not a morphology search or an RRT budget change.
    """

    task = _dynamic_task(model, state, target)
    search_config = SearchConfig(
        yaw_samples=16 if expanded else 8,
        seeds_per_yaw=16 if expanded else 8,
        normal_only_seeds=64 if expanded else 32,
        normal_max_nfev=max(config.normal_max_nfev, 400),
        random_seed=config.random_seed + 313,
        refinement_top=0,
        refinement_maxiter=0,
    )
    ik = _run_ik_search_for_task(task, target.sample, model, {}, search_config)
    # The rescue search evaluates many branches.  Use the official collision
    # result for the bulk filter and reserve exhaustive pair records for the
    # selected branches; repeating every closest-point query for all 96 IK
    # attempts can destabilize PyBullet's concave-mesh distance routine.
    filtered = _collision_filter(task, target.sample, model, ik["candidates"], detailed=False)
    for index, candidate in enumerate(filtered["collision_free"]):
        candidate["branch_rank"] = index
        candidate["search_phase"] = "expanded" if expanded else "standard"
    return filtered["collision_free"]


def _existing_endpoint_candidate(
    paths: ProjectPaths,
    model: MorphologyModel,
    state: ClimbState,
    target: ContactNode,
    q: np.ndarray,
    config: WholeTowerConfig,
) -> dict[str, object] | None:
    """Check an old endpoint q against a possibly changed support pose.

    A branch rescue changes the support suction pose's yaw as well as q.  The
    next edge therefore cannot blindly reuse the old graph endpoint.  This
    helper cheaply tests whether it remains a formal position+normal and
    collision-valid endpoint before paying for a new multi-branch search.
    """

    task = _dynamic_task(model, state, target)
    values = np.asarray(q, dtype=float)
    world_state = model.world_state_for_support(
        values,
        task.support_endpoint,
        _pose_matrix(task.support_pose),
    )
    moving_pose = world_state.suction_pose(task.moving_endpoint)
    position_error = float(np.linalg.norm(moving_pose[:3, 3] - target.sample.xyz_m))
    normal_error = float(
        np.degrees(
            angle_between_vectors_rad(
                moving_pose[:3, 2],
                -np.asarray(target.sample.normal, dtype=float),
            )
        )
    )
    if position_error > config.position_tolerance_m or normal_error > config.normal_tolerance_deg:
        return None
    with MorphologyCollisionWorld(paths, model, gui=False) as world:
        valid, clearance, critical = _state_valid_for_edge(
            world,
            model,
            task,
            target.sample,
            values,
            allow_start=False,
            allow_goal=True,
            config=config,
        )
    if not valid:
        return None
    joint_margin = float(
        np.min(np.minimum(values - model.spec.lower_limits, model.spec.upper_limits - values))
    )
    return {
        "q": values.copy(),
        "position_error_m": position_error,
        "normal_error_deg": normal_error,
        "minimum_clearance_m": float(clearance),
        "critical_pair": critical,
        "joint_limit_margin_rad": joint_margin,
    }


def _contact_key(state: ClimbState) -> tuple[str, str, str]:
    return (state.support_endpoint, state.support_node_id, state.moving_node_id)


def _checkpoint_path(paths: ProjectPaths) -> Path:
    return paths.repo_root / "models" / "design_results" / "checkpoints" / SEARCH_CHECKPOINT


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_or_initialize(
    paths: ProjectPaths,
    model: MorphologyModel,
    nodes: dict[str, tuple[ContactNode, ...]],
    config: WholeTowerConfig,
    checkpoint: Path,
    resume: bool,
) -> tuple[dict[str, ClimbState], list[tuple[float, int, str]], dict[str, str], dict[str, object], int]:
    if resume and checkpoint.exists():
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        if data.get("design_name") == model.spec.name and data.get("status") in {
            "RUNNING",
            "INTERRUPTED_RESUMABLE",
            "NO_SOLUTION_FOUND_WITHIN_SEARCH_SPACE",
            "SUCCESS_ENDPOINT_GRAPH",
        }:
            states = {key: _state_from_dict(value) for key, value in data.get("states", {}).items()}
            frontier = [tuple(item) for item in data.get("frontier", [])]
            frontier = [(float(item[0]), int(item[1]), str(item[2])) for item in frontier]
            edge_cache = {str(key): str(value) for key, value in data.get("edge_cache", {}).items()}
            raw_edges = data.get("edges", {})
            metadata = {
                "edges": raw_edges,
                "initial": data.get("initial"),
                "goal_height_m": data.get("goal_height_m"),
                "expanded": int(data.get("expanded", 0)),
                "checkpoint_config": data.get("config", {}),
            }
            return states, frontier, edge_cache, metadata, int(data.get("expanded", 0))
    initial, initial_report = _model_initial_state(paths, model, nodes, config)
    if initial is None:
        metadata = {
            "edges": {},
            "initial": initial_report,
            "goal_height_m": max(float(node.sample.xyz_m[2]) for values in nodes.values() for node in values),
            "expanded": 0,
            "initial_state": None,
        }
        return {}, [], {}, metadata, 0
    states = {initial.state_id: initial}
    frontier = [(-initial.height_m, 0, initial.state_id)]
    metadata = {
        "edges": {},
        "initial": initial_report,
        "goal_height_m": max(float(node.sample.xyz_m[2]) for values in nodes.values() for node in values),
        "expanded": 0,
    }
    return states, frontier, {}, metadata, 0


def search_design(
    paths: ProjectPaths,
    model: MorphologyModel,
    *,
    config: WholeTowerConfig | None = None,
    resume: bool = True,
    report_path: Path | None = None,
) -> dict[str, object]:
    """Search one fixed morphology and persist every expansion as a checkpoint."""

    config = config or WholeTowerConfig()
    nodes = _load_nodes(paths, config)
    checkpoint = _checkpoint_path(paths)
    states, frontier, edge_cache, metadata, expanded = _load_or_initialize(
        paths, model, nodes, config, checkpoint, resume
    )
    # A resumed run may deliberately raise the high-confidence branch range.
    # Reopen only failed endpoint edges in that newly covered height band;
    # successful states and their q configurations remain valid cache entries.
    previous_config = metadata.get("checkpoint_config", {})
    previous_high = float(previous_config.get("high_confidence_height_m", 0.0))
    if config.high_confidence_height_m > previous_high:
        reopened_sources: set[str] = set()
        for edge_id, edge in list(metadata["edges"].items()):
            target = edge.get("target_node", {})
            xyz = target.get("sample", {}).get("xyz_m", [0.0, 0.0, 0.0])
            if (
                not bool(edge.get("endpoint_valid", False))
                and float(xyz[2]) <= config.high_confidence_height_m
            ):
                edge_cache.pop(edge_id, None)
                metadata["edges"].pop(edge_id, None)
                source_id = edge.get("source_state_id")
                if source_id is not None:
                    reopened_sources.add(str(source_id))
        for source_id in reopened_sources:
            source = states.get(source_id)
            if source is not None:
                frontier.append((-source.height_m, len(frontier) + 1, source_id))
    goal_height = float(metadata["goal_height_m"])
    if not states:
        report = {
            "status": "NO_INITIAL_CONTACT_STATE",
            "design_name": model.spec.name,
            "maximum_reachable_height_m": 0.0,
            "goal_height_m": goal_height,
            "initial": metadata["initial"],
            "expanded": 0,
            "state_count": 0,
            "edge_count": 0,
            "edges": {},
            "whole_tower_complete": False,
        }
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    evaluator = MorphologyTaskEvaluator(
        model,
        settings=DesignEvaluationSettings(
            position_tolerance_m=config.position_tolerance_m,
            normal_tolerance_deg=config.normal_tolerance_deg,
            local_max_nfev=config.local_max_nfev,
            normal_max_nfev=config.normal_max_nfev,
            seed_count=config.seed_count,
            yaw_samples=config.yaw_samples,
            random_seed=config.random_seed,
            run_collision=True,
            run_trajectory=False,
        ),
    )
    seen: dict[tuple[str, str, str], str] = {}
    for state_id, state in states.items():
        seen.setdefault(_contact_key(state), state_id)
    counter = max((item[1] for item in frontier), default=0) + 1
    status = "RUNNING"
    completed_state: ClimbState | None = None
    started = time.perf_counter()
    fine_edges: list[dict[str, object]] = []
    fine_success = False
    route_states: list[str] = []
    resolved_route_states: list[ClimbState] = []
    rescue_events: list[dict[str, object]] = []
    fine_cache: dict[str, dict[str, object]] = {}

    # A resumable run may already have expanded the highest state before the
    # goal tolerance was introduced or before a previous budget expired.  Do
    # not force the resumed search to walk the lower frontier again.
    existing_goal_states = [
        value for value in states.values()
        if value.height_m >= goal_height - config.goal_tolerance_m
    ]
    if existing_goal_states:
        completed_state = max(existing_goal_states, key=lambda value: value.height_m)
        status = "SUCCESS_ENDPOINT_GRAPH"

    def persist(current_status: str) -> None:
        payload = {
            "status": current_status,
            "design_name": model.spec.name,
            "topology_id": model.spec.topology_id,
            "link_lengths_m": model.spec.link_lengths_m.tolist(),
            "config": config.__dict__,
            "goal_height_m": goal_height,
            "expanded": expanded,
            "states": {key: _state_dict(value) for key, value in states.items()},
            "frontier": frontier,
            "edge_cache": edge_cache,
            "edges": metadata["edges"],
            "initial": metadata["initial"],
            "maximum_reachable_height_m": max((value.height_m for value in states.values()), default=0.0),
            "completed_state_id": None if completed_state is None else completed_state.state_id,
            "fine_route": {
                "route_state_ids": route_states,
                "states": [_state_dict(value) for value in resolved_route_states],
                "edges": fine_edges,
                "rescue_events": rescue_events,
            },
        }
        _write_checkpoint(checkpoint, payload)

    persist(status)
    try:
        while frontier and expanded < config.max_expansions and completed_state is None:
            _, _, state_id = heapq.heappop(frontier)
            state = states.get(state_id)
            if state is None:
                continue
            if state.height_m >= goal_height - config.goal_tolerance_m:
                completed_state = state
                status = "SUCCESS_ENDPOINT_GRAPH"
                break
            expanded += 1
            targets = _candidate_targets(state, nodes, config)
            if not targets:
                metadata["edges"][f"{state.state_id}|NO_TARGET"] = {
                    "edge_id": f"{state.state_id}|NO_TARGET",
                    "source_state_id": state.state_id,
                    "status": "FAILED",
                    "failure_type": "NO_UPWARD_NODE_WITHIN_STEP_BOUND",
                }
            for target in targets:
                edge_id = _edge_key(state, target)
                if edge_id in edge_cache:
                    continue
                edge, successor = _endpoint_edge(model, evaluator, state, target, config)
                edge_cache[edge_id] = "done"
                metadata["edges"][edge_id] = edge
                if successor is None:
                    continue
                key = _contact_key(successor)
                previous_id = seen.get(key)
                if previous_id is not None and states[previous_id].height_m >= successor.height_m - 1e-8:
                    continue
                states[successor.state_id] = successor
                seen[key] = successor.state_id
                heapq.heappush(frontier, (-successor.height_m, counter, successor.state_id))
                counter += 1
            persist(status)
            if expanded % 5 == 0:
                print(
                    f"{model.spec.name}: expanded={expanded}, states={len(states)}, "
                    f"frontier={len(frontier)}, max_height={max(value.height_m for value in states.values()):.3f} m",
                    flush=True,
                )
        if completed_state is None:
            if not frontier:
                status = "NO_SOLUTION_WITHIN_GRAPH"
            elif expanded >= config.max_expansions:
                status = "INTERRUPTED_RESUMABLE"
    except KeyboardInterrupt:
        status = "INTERRUPTED_RESUMABLE"
    persist(status)

    # The endpoint graph is useful for proving reachability, but a climbing
    # route is only successful after every selected edge passes trajectory
    # validation.  Try several highest terminal routes because a single
    # endpoint-valid q branch may have a blocked joint-space path.
    def route_for_terminal(terminal: ClimbState) -> list[str]:
        result = [terminal.state_id]
        current = terminal
        while current.route_edge_ids:
            edge_id = current.route_edge_ids[-1]
            source = states[metadata["edges"][edge_id]["source_state_id"]]
            result.append(source.state_id)
            current = source
        result.reverse()
        return result

    def _node_for_id(node_id: str) -> ContactNode:
        for values in nodes.values():
            for node in values:
                if node.node_id == node_id:
                    return node
        raise KeyError(f"找不到附着节点: {node_id}")

    def _fine_route_with_branch_rescue(
        candidate_route_states: list[str],
        route_number: int,
    ) -> tuple[bool, list[ClimbState], list[dict[str, object]], list[dict[str, object]]]:
        """Validate a route while rebuilding all downstream states after rescue.

        The endpoint graph keeps one representative q per contact state.  When
        that q has a blocked joint-space path, a collision-free IK branch can
        have a different suction yaw.  Rebase makes that yaw part of the next
        support pose, so every later edge must be re-evaluated from the rescued
        state instead of reusing the old graph q values.
        """

        current = states[candidate_route_states[0]]
        actual_states: list[ClimbState] = [current]
        actual_fine: list[dict[str, object]] = []
        route_rescues: list[dict[str, object]] = []
        diverged = False
        route_dir = (
            paths.repo_root
            / "models"
            / "design_results"
            / "whole_tower_routes"
            / model.spec.topology_id
            / "rescue_routes"
            / f"route_{route_number:02d}_{candidate_route_states[-1]}"
        )

        for edge_index, successor_id in enumerate(candidate_route_states[1:]):
            graph_successor = states[successor_id]
            edge_id = graph_successor.route_edge_ids[-1]
            edge = metadata["edges"][edge_id]
            target = _node_for_id(str(edge["target_node"]["node_id"]))
            original_q = np.asarray(edge["q_goal"], dtype=float)
            direct_fine: dict[str, object] | None = None

            if not diverged:
                if edge_id in fine_cache:
                    direct_fine = fine_cache[edge_id]
                else:
                    direct_fine = _fine_validate_edge(
                        paths,
                        model,
                        current,
                        target,
                        original_q,
                        config,
                        paths.repo_root
                        / "models"
                        / "design_results"
                        / "whole_tower_routes"
                        / model.spec.topology_id,
                        edge_index,
                    )
                    fine_cache[edge_id] = direct_fine
            else:
                # After rebase, the old q may or may not still solve the same
                # absolute target.  Do not send an invalid endpoint to RRT.
                old_endpoint = _existing_endpoint_candidate(
                    paths,
                    model,
                    current,
                    target,
                    original_q,
                    config,
                )
                if old_endpoint is not None:
                    direct_fine = _fine_validate_edge(
                        paths,
                        model,
                        current,
                        target,
                        original_q,
                        config,
                        route_dir,
                        10000 + route_number * 1000 + edge_index,
                    )

            if direct_fine is not None and bool(direct_fine["success"]):
                next_state = (
                    graph_successor
                    if not diverged
                    else _new_state(model, current, target, original_q, edge_id)
                )
                actual_fine.append({"edge_id": edge_id, **direct_fine})
                actual_states.append(next_state)
                current = next_state
                continue

            candidate_attempts: list[dict[str, object]] = []
            selected: tuple[dict[str, object], dict[str, object]] | None = None
            all_rescue_candidates: list[dict[str, object]] = []
            next_candidate_index = 0
            for search_phase, expanded in (("standard", False), ("expanded", True)):
                if search_phase == "expanded" and selected is not None:
                    break
                candidates = _branch_rescue_candidates(
                    model,
                    current,
                    target,
                    config,
                    expanded=expanded,
                )
                # The expanded pass can rediscover a standard branch.  Avoid
                # spending a second fixed RRT budget on numerically identical
                # endpoint configurations.
                for candidate in candidates:
                    q_candidate = np.asarray(candidate["q"], dtype=float)
                    duplicate = any(
                        float(
                            np.linalg.norm(
                                np.arctan2(
                                    np.sin(q_candidate - np.asarray(previous["q"], dtype=float)),
                                    np.cos(q_candidate - np.asarray(previous["q"], dtype=float)),
                                )
                            )
                        )
                        <= 1e-3
                        for previous in all_rescue_candidates
                    )
                    if not duplicate:
                        all_rescue_candidates.append(candidate)

                for candidate in all_rescue_candidates[next_candidate_index:]:
                    q_candidate = np.asarray(candidate["q"], dtype=float)
                    branch_rank = len(candidate_attempts)
                    fine = _fine_validate_edge(
                        paths,
                        model,
                        current,
                        target,
                        q_candidate,
                        config,
                        route_dir,
                        20000 + route_number * 1000 + edge_index * 100 + branch_rank,
                    )
                    candidate_attempts.append(
                        {
                            "branch_rank": branch_rank,
                            "source_branch_rank": int(candidate.get("branch_rank", branch_rank)),
                            "search_phase": str(candidate.get("search_phase", search_phase)),
                            "q": q_candidate.tolist(),
                            "endpoint_clearance_m": float(
                                candidate.get("collision", {}).get("minimum_clearance_m", float("nan"))
                            ),
                            "joint_limit_margin_rad": float(candidate.get("joint_limit_margin_rad", float("nan"))),
                            "fine": fine,
                        }
                    )
                    if bool(fine["success"]):
                        selected = (candidate, fine)
                        break
                next_candidate_index = len(all_rescue_candidates)
                if selected is not None:
                    break

            rescue_record: dict[str, object] = {
                "edge_index": edge_index,
                "source_state_id": current.state_id,
                "target_node_id": target.node_id,
                "graph_edge_id": edge_id,
                "original_q_goal": original_q.tolist(),
                "original_fine": direct_fine,
                "candidate_count": len(all_rescue_candidates),
                "standard_candidate_count": sum(
                    1 for item in all_rescue_candidates if item.get("search_phase") == "standard"
                ),
                "expanded_candidate_count": sum(
                    1 for item in all_rescue_candidates if item.get("search_phase") == "expanded"
                ),
                "candidate_attempts": candidate_attempts,
                "selected_branch_rank": None,
            }
            if selected is None:
                route_rescues.append(rescue_record)
                return False, actual_states, actual_fine, route_rescues

            selected_candidate, selected_fine = selected
            selected_q = np.asarray(selected_candidate["q"], dtype=float)
            next_state = _new_state(model, current, target, selected_q, edge_id)
            selected_rank = next(
                int(item["branch_rank"])
                for item in candidate_attempts
                if np.linalg.norm(
                    np.arctan2(
                        np.sin(np.asarray(item["q"], dtype=float) - selected_q),
                        np.cos(np.asarray(item["q"], dtype=float) - selected_q),
                    )
                )
                <= 1e-3
            )
            rescue_record["selected_branch_rank"] = selected_rank
            rescue_record["selected_search_phase"] = str(selected_candidate.get("search_phase", "unknown"))
            rescue_record["selected_q"] = selected_q.tolist()
            rescue_record["selected_fine"] = selected_fine
            route_rescues.append(rescue_record)
            actual_fine.append({
                "edge_id": edge_id,
                "branch_rank": selected_rank,
                **selected_fine,
            })
            actual_states.append(next_state)
            current = next_state
            diverged = True

        return True, actual_states, actual_fine, route_rescues

    if completed_state is not None and config.run_fine_trajectories:
        terminal_states = sorted(
            (
                value for value in states.values()
                if value.height_m >= goal_height - config.goal_tolerance_m
            ),
            key=lambda value: (value.height_m, -value.depth),
            reverse=True,
        )[: config.max_fine_routes]
        try:
            for route_number, terminal in enumerate(terminal_states):
                candidate_route_states = route_for_terminal(terminal)
                route_ok, actual_route, candidate_fine, candidate_rescues = _fine_route_with_branch_rescue(
                    candidate_route_states,
                    route_number,
                )
                rescue_events.extend(candidate_rescues)
                if route_ok:
                    fine_success = True
                    route_states = [value.state_id for value in actual_route]
                    resolved_route_states = actual_route
                    fine_edges = candidate_fine
                    completed_state = actual_route[-1]
                    persist("SUCCESS_ENDPOINT_GRAPH")
                    break
                # Preserve the most complete failed route for diagnostics if no
                # terminal route survives fine validation.
                if not fine_edges or len(candidate_fine) > len(fine_edges):
                    route_states = [value.state_id for value in actual_route]
                    resolved_route_states = actual_route
                    fine_edges = candidate_fine
                persist("RUNNING_FINE")
        except KeyboardInterrupt:
            status = "INTERRUPTED_RESUMABLE"
            persist(status)

    maximum = max((value.height_m for value in states.values()), default=0.0)
    if fine_success:
        final_status = "SUCCESS_6R" if model.spec.dof == 6 else "SUCCESS_8R_REQUIRED"
    elif status == "INTERRUPTED_RESUMABLE":
        final_status = "INTERRUPTED_RESUMABLE"
    else:
        final_status = "NO_SOLUTION_FOUND_WITHIN_SEARCH_SPACE"
    report = {
        "status": final_status,
        "endpoint_graph_status": status,
        "design_name": model.spec.name,
        "topology_id": model.spec.topology_id,
        "dof": model.spec.dof,
        "per_side_dof": model.spec.per_side_dof,
        "link_lengths_m": model.spec.link_lengths_m.tolist(),
        "goal_height_m": goal_height,
        "maximum_reachable_height_m": maximum,
        "whole_tower_complete": bool(fine_success),
        "initial": metadata["initial"],
        "expanded": expanded,
        "state_count": len(states),
        "edge_count": len(metadata["edges"]),
        "route_state_ids": route_states,
        "fine_edges": fine_edges,
        "edges": metadata["edges"],
        "runtime_s": time.perf_counter() - started,
        "checkpoint": str(checkpoint),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    persist(final_status)
    return report


def _load_model(paths: ProjectPaths, design: str) -> MorphologyModel:
    geometry = BaselineGeometry.from_project(paths)
    if design == "axis8":
        return MorphologyModel(fixed_axis8_spec(geometry))
    if design == "best6r":
        result = json.loads((paths.repo_root / "models" / "design_results" / "best_6r.json").read_text(encoding="utf-8"))
        spec = MorphologySpec.six_r_topology(
            geometry,
            int(result["remove_pair_index"]),
            link_lengths_m=np.asarray(result["link_lengths_m"], dtype=float),
            collision_inflation_m=0.005,
        )
        return MorphologyModel(spec)
    if design == "baseline8":
        return MorphologyModel(MorphologySpec.baseline_8r(geometry, collision_inflation_m=0.005))
    raise ValueError(f"未知设计：{design}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="固定6R/8R候选的全塔contact-state搜索")
    parser.add_argument("--design", choices=("axis8", "best6r", "baseline8"), default="axis8")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--max-expansions", type=int, default=300)
    parser.add_argument("--candidates-per-surface", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-fine-trajectories", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    model = _load_model(paths, args.design)
    config = WholeTowerConfig(
        max_expansions=args.max_expansions,
        candidates_per_surface=max(1, args.candidates_per_surface),
        run_fine_trajectories=not args.no_fine_trajectories,
    )
    report_path = paths.repo_root / "models" / "design_results" / "whole_tower_routes" / f"{model.spec.topology_id}_search.json"
    report = search_design(paths, model, config=config, resume=args.resume, report_path=report_path)
    print(json.dumps({key: report[key] for key in ("status", "maximum_reachable_height_m", "expanded", "state_count", "edge_count")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
