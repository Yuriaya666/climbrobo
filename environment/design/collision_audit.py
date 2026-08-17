"""Independent collision audit for the parameterized 8R design model.

This module deliberately does not run IK or planning.  It compares the real
URDF collision model with the parameterized proxy at the three already known
baseline states and, when an explicit q-goal artifact is supplied, audits
those exact states as well.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pybullet as p
import trimesh

from environment.design.collision_proxy import MorphologyCollisionWorld
from environment.design.morphology import (
    BaselineGeometry,
    MorphologyModel,
    MorphologySpec,
    MorphologyState,
)
from environment.paths import ProjectPaths
from environment.scene import PyBulletScene
from environment.transforms import RigidTransform


# Large distance queries against the dense L4 concave mesh can return one
# point per triangle in PyBullet.  A small near-contact window is sufficient
# for collision auditing and avoids turning one pair query into hundreds of
# thousands of mesh points.
AUDIT_DISTANCE_M = 0.02
PAIR_QUERY_DISTANCE_M = AUDIT_DISTANCE_M


@dataclass(frozen=True)
class AuditState:
    name: str
    q_urdf: np.ndarray
    base_pose: RigidTransform


def _pose_matrix(pose: RigidTransform) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = pose.rotation_matrix
    result[:3, 3] = pose.position
    return result


def _array(value: object) -> list[float]:
    return np.asarray(value, dtype=float).tolist()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.keys()}


def _baseline_states(paths: ProjectPaths) -> tuple[AuditState, ...]:
    output = paths.repo_root / "models" / "candidate_output"
    step1 = _load_npz(output / "successful_one_step_trajectory.npz")
    step2 = _load_npz(output / "successful_step2_trajectory.npz")
    identity = RigidTransform.identity()
    step2_pose = RigidTransform(
        step2["base_position_m"][-1],
        step2["base_orientation_xyzw"][-1],
    )
    return (
        AuditState("initial", step1["trajectory_rad"][0], identity),
        AuditState("step1_final", step1["trajectory_rad"][-1], identity),
        AuditState("step2_final", step2["trajectory_rad"][-1], step2_pose),
    )


def _closest_points(
    body_a: int,
    body_b: int,
    *,
    link_a: int | None = None,
    link_b: int | None = None,
    distance: float = PAIR_QUERY_DISTANCE_M,
) -> tuple[object, ...]:
    kwargs: dict[str, int] = {}
    if link_a is not None:
        kwargs["linkIndexA"] = int(link_a)
    if link_b is not None:
        kwargs["linkIndexB"] = int(link_b)
    return tuple(p.getClosestPoints(body_a, body_b, distance, **kwargs))


def _point_record(point: object, *, name_a: str, name_b: str, allowed: bool = False) -> dict[str, object]:
    value = tuple(point)
    distance = float(value[8])
    return {
        "link_a": name_a,
        "link_b": name_b,
        "distance_m": distance,
        "penetration_m": max(0.0, -distance),
        "allowed_adjacent": bool(allowed),
        "point_on_a_m": _array(value[5]),
        "point_on_b_m": _array(value[6]),
        "normal_on_b": _array(value[7]),
    }


def _best_pair(
    points: Iterable[object],
    *,
    name_a: str,
    name_b: str,
    allowed: bool = False,
) -> dict[str, object] | None:
    points = tuple(points)
    if not points:
        return None
    point = min(points, key=lambda item: float(item[8]))
    return _point_record(point, name_a=name_a, name_b=name_b, allowed=allowed)


def _actual_self_pairs(scene: PyBulletScene) -> list[dict[str, object]]:
    if scene.robot_id is None:
        raise RuntimeError("真实机器人尚未加载")
    robot = scene.robot_id
    links = [-1, *range(p.getNumJoints(robot))]
    adjacent = scene.adjacent_link_pairs()
    records: list[dict[str, object]] = []
    for index, link_a in enumerate(links):
        for link_b in links[index + 1 :]:
            record = _best_pair(
                _closest_points(robot, robot, link_a=link_a, link_b=link_b),
                name_a=scene.link_name(link_a),
                name_b=scene.link_name(link_b),
                allowed=tuple(sorted((link_a, link_b))) in adjacent,
            )
            if record is not None:
                records.append(record)
    return sorted(records, key=lambda item: float(item["distance_m"]))


def _actual_tower_pairs(scene: PyBulletScene) -> list[dict[str, object]]:
    if scene.robot_id is None or scene.tower_id is None:
        raise RuntimeError("真实机器人或Tower尚未加载")
    records: list[dict[str, object]] = []
    for link in [-1, *range(p.getNumJoints(scene.robot_id))]:
        record = _best_pair(
            _closest_points(scene.robot_id, scene.tower_id, link_a=link, link_b=-1),
            name_a=scene.link_name(link),
            name_b="Tower",
        )
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda item: float(item["distance_m"]))


def _proxy_ids(world: MorphologyCollisionWorld) -> list[int]:
    values = [
        world.central_id,
        *world.endpoint_ids.values(),
        *world.segment_ids["left"],
        *world.segment_ids["right"],
        *world.joint_ids["left"],
        *world.joint_ids["right"],
    ]
    return [int(value) for value in values if value is not None]


def _proxy_self_pairs(world: MorphologyCollisionWorld) -> list[dict[str, object]]:
    ids = _proxy_ids(world)
    records: list[dict[str, object]] = []
    for index, body_a in enumerate(ids):
        for body_b in ids[index + 1 :]:
            record = _best_pair(
                _closest_points(body_a, body_b),
                name_a=world._body_names[body_a],
                name_b=world._body_names[body_b],
                allowed=tuple(sorted((body_a, body_b))) in world._allowed_adjacent_pairs,
            )
            if record is not None:
                records.append(record)
    return sorted(records, key=lambda item: float(item["distance_m"]))


def _proxy_tower_pairs(world: MorphologyCollisionWorld) -> list[dict[str, object]]:
    if world.tower_id is None:
        return []
    records: list[dict[str, object]] = []
    for body in _proxy_ids(world):
        record = _best_pair(
            _closest_points(body, world.tower_id),
            name_a=world._body_names[body],
            name_b="Tower",
        )
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda item: float(item["distance_m"]))


def _actual_body_poses(
    paths: ProjectPaths,
    states: tuple[AuditState, ...],
) -> tuple[dict[str, RigidTransform], dict[str, dict[str, object]]]:
    body_poses: dict[str, RigidTransform] = {}
    frame_rows: dict[str, dict[str, object]] = {}
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_robot(states[0].base_pose)
        for audit_state in states:
            scene.set_base_pose(audit_state.base_pose)
            scene.reset_joints(audit_state.q_urdf)
            l4_pose = scene.get_link_pose(scene.link_index("L4"))
            body_poses[audit_state.name] = l4_pose
            frame_rows[audit_state.name] = {
                "l4_world_position_m": _array(l4_pose.position),
                "l4_world_quaternion_xyzw": _array(l4_pose.quaternion_xyzw),
            }
    return body_poses, frame_rows


def _baseline_frame_check(
    paths: ProjectPaths,
    states: tuple[AuditState, ...],
    geometry: BaselineGeometry,
    model: MorphologyModel,
    body_poses: dict[str, RigidTransform],
) -> dict[str, object]:
    position_errors: list[float] = []
    state_rows: list[dict[str, object]] = []
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_robot(states[0].base_pose)
        for audit_state in states:
            scene.set_base_pose(audit_state.base_pose)
            scene.reset_joints(audit_state.q_urdf)
            param = model.forward(
                geometry.baseline_joint_vector(audit_state.q_urdf),
                body_pose=_pose_matrix(body_poses[audit_state.name]),
            )
            comparisons: list[float] = []
            for link_name, expected in zip(
                ("L4", "L3", "L2", "L1"), param.left_joint_poses
            ):
                actual = scene.get_link_pose(scene.link_index(link_name)).position
                comparisons.append(float(np.linalg.norm(actual - expected[:3, 3])))
            for link_name, expected in zip(
                ("L5", "L6", "L7", "L8"), param.right_joint_poses
            ):
                actual = scene.get_link_pose(scene.link_index(link_name)).position
                comparisons.append(float(np.linalg.norm(actual - expected[:3, 3])))
            position_errors.extend(comparisons)
            state_rows.append(
                {
                    "state": audit_state.name,
                    "max_joint_frame_position_error_m": max(comparisons, default=0.0),
                    "joint_frame_position_errors_m": comparisons,
                }
            )
    return {
        "max_position_error_m": max(position_errors, default=0.0),
        "states": state_rows,
        "transform_error_threshold_m": 1e-5,
        "status": "PASS" if max(position_errors, default=0.0) <= 1e-5 else "FAIL",
    }


def _mesh_report(path: Path) -> dict[str, object]:
    mesh = trimesh.load_mesh(str(path), force="mesh", process=False)
    return {
        "path": str(path),
        "local_bounds_m": np.asarray(mesh.bounds, dtype=float).tolist(),
        "local_extents_m": np.asarray(mesh.extents, dtype=float).tolist(),
        "vertex_count": int(len(mesh.vertices)),
    }


def _proxy_source_report(
    geometry: BaselineGeometry,
    spec: MorphologySpec,
    baseline_spec: MorphologySpec,
) -> dict[str, object]:
    """Record where the proxy dimensions actually came from."""

    return {
        "legacy_raw_radius_m": 0.065,
        "legacy_effective_radius_m": 0.070,
        "legacy_link_radius_source": "hard-coded MorphologySpec.from_geometry literal 0.065 m",
        "legacy_joint_radius_source": "same hard-coded literal was incorrectly reused for motor/joint spheres",
        "inflation_source": "fixed_axis8_spec collision_inflation_m=0.005 m",
        "current_link_raw_radius_m": float(spec.link_proxy_radius_m),
        "current_link_effective_radius_m": float(spec.link_proxy_radius_m + spec.collision_inflation_m),
        "current_joint_raw_radius_m": float(spec.joint_proxy_radius_m),
        "current_joint_effective_radius_m": float(spec.joint_proxy_radius_m + spec.collision_inflation_m),
        "joint_radius_correction_basis": {
            "baseline_joint_center_spacing_m": float(baseline_spec.link_lengths_m[1]),
            "maximum_raw_radius_before_5mm_inflation_overlap_m": float(
                0.5 * baseline_spec.link_lengths_m[1] - spec.collision_inflation_m
            ),
            "selected_audit_raw_radius_m": float(spec.joint_proxy_radius_m),
            "status": "audit envelope only; replace with measured motor CAD before final design",
        },
        "mesh_references": {
            "L1": _mesh_report(geometry.left.endpoint_mesh_path.parent / "L1.STL"),
            "L3": _mesh_report(geometry.left.endpoint_mesh_path.parent / "L3.STL"),
            "L7": _mesh_report(geometry.right.endpoint_mesh_path.parent / "L7.STL"),
        },
    }


def _right_link3_report(
    geometry: BaselineGeometry,
    spec: MorphologySpec,
    state: MorphologyState,
    *,
    body_mesh: dict[str, object],
) -> dict[str, object]:
    # right_link_3 is span index 2: J7 center -> J8 center.
    segment_index = 2
    start, end = state.right_span_segments[segment_index]
    return {
        "proxy_name": "right_link_3",
        "span_index": segment_index,
        "parent_joint": "J7",
        "child_joint": "J8",
        "parent_joint_parent_link": "L6",
        "child_joint_child_link": "L8",
        "directly_connected_to_central_body": False,
        "central_body_link": geometry.central_link_name,
        "joint_center_world_m": {
            "J7": _array(state.right_joint_poses[2][:3, 3]),
            "J8": _array(state.right_joint_poses[3][:3, 3]),
        },
        "capsule_start_world_m": _array(start),
        "capsule_end_world_m": _array(end),
        "capsule_center_world_m": _array(0.5 * (start + end)),
        "capsule_axis_length_m": float(np.linalg.norm(end - start)),
        "kinematic_joint_center_distance_m": float(np.linalg.norm(end - start)),
        "capsule_effective_length_m": float(np.linalg.norm(end - start)),
        "capsule_start_offset_from_J7_m": 0.0,
        "capsule_end_offset_from_J8_m": 0.0,
        "raw_link_proxy_radius_m": float(spec.link_proxy_radius_m),
        "safety_inflation_m": float(spec.collision_inflation_m),
        "capsule_radius_m": float(spec.link_proxy_radius_m + spec.collision_inflation_m),
        "central_body_mesh": body_mesh,
        "central_body_frame": "L4 link frame; parameterized body_pose is this frame, not base_link",
        "right_mount_pose_in_body_m": _array(spec.right_mount_pose[:3, 3]),
        "right_mount_parent_link": "L4",
        "mounting_relation": "J5 is the first right-side joint mounted from L4; right_link_3 is two joints farther out",
    }


def _load_explicit_q_goals(path: Path | None) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if path is None:
        return {}, {"status": "MISSING", "reason": "没有提供有限轴8R四个正式q_goal的持久化文件"}
    if not path.exists():
        return {}, {"status": "MISSING", "path": str(path), "reason": "指定q_goal文件不存在"}

    def scalar_text(value: object) -> str:
        array = np.asarray(value)
        return str(array.item() if array.ndim == 0 else array.tolist())

    goals: dict[str, np.ndarray] = {}
    source_files: list[str] = []
    if path.is_dir():
        candidates = sorted(path.glob("optimized_8r_axis8_task_*.npz"))
        if not candidates and (path / "q_goals.npz").exists():
            candidates = [path / "q_goals.npz"]
    else:
        candidates = [path]

    for candidate in candidates:
        data = _load_npz(candidate)
        if "q" in data and np.asarray(data["q"]).shape == (8,):
            task_id = scalar_text(data.get("task_id", candidate.stem))
            goals[task_id] = np.asarray(data["q"], dtype=float).copy()
            source_files.append(str(candidate))
            continue
        task_ids = data.get("task_ids")
        names = [scalar_text(value) for value in task_ids.tolist()] if task_ids is not None else []
        values = np.asarray(data.get("q_goals", np.empty((0, 8))), dtype=float)
        if values.ndim == 2 and values.shape[1] == 8:
            for index, value in enumerate(values):
                if index < len(names):
                    goals[names[index]] = value.copy()
                    source_files.append(str(candidate))
    return goals, {
        "status": "FOUND" if goals else "MISSING",
        "path": str(path),
        "source_files": source_files,
        "task_ids": sorted(goals),
    }


def fixed_axis8_spec(geometry: BaselineGeometry) -> MorphologySpec:
    """Return the selected finite-axis candidate without running a search."""

    spec = MorphologySpec.optimized_8r(
        geometry,
        np.asarray(
            [0.2944788185070823, 0.15865600335865368, 0.2098367563619965, 0.20845446162884873],
        ),
        collision_inflation_m=0.005,
    )
    axes_left = np.asarray(
        [[0, 0, -1], [1, 0, 0], [0, 0, -1], [1, 0, 0]],
        dtype=float,
    )
    axes_right = np.asarray(
        [[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]],
        dtype=float,
    )
    return replace(
        spec,
        name="OPTIMIZED_8R_YAW_PITCH_YAW_PITCH",
        topology_id="finite_axis_yaw_pitch_yaw_pitch",
        left_axes=axes_left,
        right_axes=axes_right,
        left_full_axes=axes_left,
        right_full_axes=axes_right,
    )


def _pair_by_name(records: Iterable[dict[str, object]], first: str, second: str) -> dict[str, object] | None:
    wanted = {first, second}
    for record in records:
        if {record["link_a"], record["link_b"]} == wanted:
            return record
    return None


def _radius_sensitivity(
    paths: ProjectPaths,
    model: MorphologyModel,
    state: MorphologyState,
    *,
    pair: tuple[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for radius in (0.001, 0.02, 0.04, 0.05, 0.065):
        spec = replace(
            model.spec,
            link_proxy_radius_m=radius,
            link_proxy_radii_m=np.full(4, radius, dtype=float),
            collision_inflation_m=0.0,
        )
        variant = MorphologyCollisionWorld(paths, MorphologyModel(spec), gui=False, load_tower=False)
        with variant:
            variant.update(state)
            record = _pair_by_name(_proxy_self_pairs(variant), pair[0], pair[1])
            rows.append(
                {
                    "link_proxy_radius_m": radius,
                    "collision_inflation_m": 0.0,
                    "pair": list(pair),
                    "clearance_m": None if record is None else record["distance_m"],
                }
            )
    return rows


def _audit_baseline(
    paths: ProjectPaths,
    states: tuple[AuditState, ...],
    geometry: BaselineGeometry,
    spec: MorphologySpec,
    model: MorphologyModel,
    body_poses: dict[str, RigidTransform],
) -> dict[str, object]:
    actual_rows: list[dict[str, object]] = []
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_tower()
        scene.load_robot(states[0].base_pose)
        for audit_state in states:
            scene.set_base_pose(audit_state.base_pose)
            scene.reset_joints(audit_state.q_urdf)
            self_pairs = _actual_self_pairs(scene)
            tower_pairs = _actual_tower_pairs(scene)
            actual_rows.append(
                {
                    "state": audit_state.name,
                    "self_pairs": self_pairs,
                    "non_adjacent_self_collisions": [
                        row for row in self_pairs if not row["allowed_adjacent"] and float(row["distance_m"]) <= 0.0
                    ],
                    "tower_pairs": tower_pairs,
                }
            )

    proxy_rows: list[dict[str, object]] = []
    sensitivity_state: MorphologyState | None = None
    with MorphologyCollisionWorld(paths, model, gui=False) as world:
        for audit_state in states:
            state = model.forward(
                geometry.baseline_joint_vector(audit_state.q_urdf),
                body_pose=_pose_matrix(body_poses[audit_state.name]),
            )
            world.update(state)
            self_pairs = _proxy_self_pairs(world)
            tower_pairs = _proxy_tower_pairs(world)
            proxy_rows.append(
                {
                    "state": audit_state.name,
                    "self_pairs": self_pairs,
                    "non_adjacent_self_collisions": [
                        row for row in self_pairs if not row["allowed_adjacent"] and float(row["distance_m"]) <= 0.0
                    ],
                    "tower_pairs": tower_pairs,
                }
            )
            if audit_state.name == "initial":
                sensitivity_state = state

    # PyBullet's legacy API uses a process-global default client in this
    # project.  Run variant worlds only after the main world has disconnected;
    # otherwise opening a variant would invalidate the outer audit world.
    sensitivity = None
    if sensitivity_state is not None:
        sensitivity = {
            "state": "initial",
            "central_body_right_link_3": _radius_sensitivity(
                paths, model, sensitivity_state, pair=("central_body", "right_link_3")
            ),
        }

    return {
        "state_names": [state.name for state in states],
        "actual_urdf": actual_rows,
        "parameterized_baseline_proxy": proxy_rows,
        "radius_sensitivity_reference": sensitivity,
    }


def _json_state(state: AuditState) -> dict[str, object]:
    return {
        "name": state.name,
        "q_urdf": _array(state.q_urdf),
        "base_position_m": _array(state.base_pose.position),
        "base_orientation_xyzw": _array(state.base_pose.quaternion_xyzw),
    }


def _tower_contact_is_allowed(
    row: dict[str, object],
    task,
    allowed_endpoint_positions: dict[str, np.ndarray],
    allowed_contact_radius_m: float,
) -> bool:
    name = str(row["link_a"])
    endpoint = None
    if name == "base_end_mesh":
        endpoint = "base_end"
    elif name == "l8_end_mesh":
        endpoint = "l8_end"
    elif name in {"left_link_4", "left_joint_4", "right_joint_4"}:
        endpoint = "base_end" if name.startswith("left_") else "l8_end"
    if endpoint is None or endpoint not in allowed_endpoint_positions:
        return False
    point = np.asarray(row["point_on_a_m"], dtype=float)
    return bool(np.linalg.norm(point - allowed_endpoint_positions[endpoint]) <= allowed_contact_radius_m)


def _right_link3_state_report(state: MorphologyState, spec: MorphologySpec) -> dict[str, object]:
    start, end = state.right_span_segments[2]
    return {
        "parent_joint": "J7",
        "child_joint": "J8",
        "start_world_m": _array(start),
        "end_world_m": _array(end),
        "joint_center_distance_m": float(np.linalg.norm(end - start)),
        "capsule_start_world_m": _array(start),
        "capsule_end_world_m": _array(end),
        "capsule_effective_length_m": float(np.linalg.norm(end - start)),
        "capsule_radius_m": float(spec.link_proxy_radius_m + spec.collision_inflation_m),
        "safety_inflation_m": float(spec.collision_inflation_m),
        "length_source": "fixed optimized link_lengths_m[2]",
        "geometry_source": "parameterized capsule; no physical CAD for the optimized link",
    }


def _audit_q_goal_variant(
    paths: ProjectPaths,
    spec: MorphologySpec,
    q_goals: dict[str, np.ndarray],
    *,
    label: str,
) -> dict[str, object]:
    """Audit exact q-goals under one fixed proxy configuration."""

    from environment.design.task_suite import build_task_suite
    from environment.transforms import angle_between_vectors_rad

    tasks = {task.task_id: task for task in build_task_suite(paths, max_targets_per_task=1)}
    model = MorphologyModel(spec)
    rows: list[dict[str, object]] = []
    with MorphologyCollisionWorld(paths, model, gui=False) as world:
        for task_id, q in sorted(q_goals.items()):
            if task_id not in tasks:
                rows.append({"task_id": task_id, "status": "UNKNOWN_TASK_ID", "proxy_label": label})
                continue
            task = tasks[task_id]
            sample = task.targets[0]
            state = model.world_state_for_support(q, task.support_endpoint, _pose_matrix(task.support_pose))
            moving_pose = state.suction_pose(task.moving_endpoint)
            position_error_m = float(np.linalg.norm(moving_pose[:3, 3] - sample.xyz_m))
            normal_error_deg = float(
                np.degrees(angle_between_vectors_rad(moving_pose[:3, 2], -np.asarray(sample.normal, dtype=float)))
            )
            world.update(state)
            allowed_positions = {
                task.support_endpoint: task.support_xyz,
                task.moving_endpoint: sample.xyz_m,
            }
            allowed = world.check(
                allowed_endpoint_positions=allowed_positions,
                allowed_contact_radius_m=0.09,
            )
            pairs = _proxy_self_pairs(world)
            tower_pairs = _proxy_tower_pairs(world)
            bad_self = [
                row for row in pairs
                if not row["allowed_adjacent"] and float(row["distance_m"]) <= 0.0
            ]
            bad_tower = [
                row for row in tower_pairs
                if float(row["distance_m"]) <= 0.0
                and not _tower_contact_is_allowed(row, task, allowed_positions, 0.09)
            ]
            clearance_records = [
                row for row in pairs if not row["allowed_adjacent"]
            ] + [
                row for row in tower_pairs
                if not _tower_contact_is_allowed(row, task, allowed_positions, 0.09)
            ]
            critical_record = min(
                clearance_records,
                key=lambda row: float(row["distance_m"]),
                default=None,
            )
            central_right = _pair_by_name(pairs, "central_body", "right_link_3")
            critical_pair = None
            critical_kind = None
            if critical_record is not None:
                critical_pair = f"{critical_record['link_a']}↔{critical_record['link_b']}"
                critical_kind = "TOWER_COLLISION" if critical_record["link_b"] == "Tower" else "SELF_COLLISION"
            clearance = float(critical_record["distance_m"]) if critical_record is not None else float("inf")
            rows.append(
                {
                    "proxy_label": label,
                    "task_id": task_id,
                    "status": "PASS" if allowed.ok else "FAIL",
                    "position_error_m": position_error_m,
                    "normal_error_deg": normal_error_deg,
                    "self_collision": "FAIL" if bad_self else "PASS",
                    "tower_collision": "FAIL" if bad_tower else "PASS",
                    "minimum_clearance_m": clearance,
                    "critical_pair": critical_pair,
                    "critical_kind": critical_kind,
                    "critical_position_m": None if critical_record is None else critical_record["point_on_a_m"],
                    "penetration_m": None if critical_record is None else max(0.0, -float(critical_record["distance_m"])),
                    "closest_point_on_a_m": None if critical_record is None else critical_record["point_on_a_m"],
                    "closest_point_on_b_m": None if critical_record is None else critical_record["point_on_b_m"],
                    "central_body_right_link_3": central_right,
                    "right_link_3_geometry": _right_link3_state_report(state, spec),
                    "non_adjacent_self_collisions": bad_self,
                    "tower_collisions": bad_tower,
                }
            )
    passed = sum(row.get("status") == "PASS" for row in rows)
    return {
        "label": label,
        "link_proxy_radius_m": float(spec.link_proxy_radius_m),
        "joint_proxy_radius_m": float(spec.joint_proxy_radius_m),
        "collision_inflation_m": float(spec.collision_inflation_m),
        "collision_pass_count": passed,
        "task_count": len(rows),
        "tasks": rows,
    }


def _audit_q_goals(
    paths: ProjectPaths,
    geometry: BaselineGeometry,
    spec: MorphologySpec,
    q_goals: dict[str, np.ndarray],
) -> dict[str, object]:
    if not q_goals:
        return {
            "status": "NOT_RUN",
            "reason": "没有提供四个已持久化的有限轴8R position+normal q_goal",
            "tasks": [],
            "before_collision_pass_count": None,
            "after_collision_pass_count": None,
            "safety_levels": [],
        }

    # Reconstruct the old representation only for a before/after audit.  It
    # uses the old joint radius and the duplicated right terminal capsule; it
    # is never used for planning.
    legacy_spec = replace(
        spec,
        joint_proxy_radius_m=0.065,
        joint_proxy_radii_m=np.full(4, 0.065, dtype=float),
        right_terminal_span_is_suction_offset=False,
    )
    before = _audit_q_goal_variant(paths, legacy_spec, q_goals, label="before_legacy_proxy")
    after = _audit_q_goal_variant(paths, spec, q_goals, label="after_nominal_proxy")
    safety_levels = [
        after,
        _audit_q_goal_variant(
            paths,
            replace(spec, collision_inflation_m=spec.collision_inflation_m + 0.005),
            q_goals,
            label="nominal_plus_5mm",
        ),
        _audit_q_goal_variant(
            paths,
            replace(spec, collision_inflation_m=spec.collision_inflation_m + 0.010),
            q_goals,
            label="nominal_plus_10mm",
        ),
    ]
    return {
        "status": "RUN",
        "tasks": after["tasks"],
        "before_collision_pass_count": before["collision_pass_count"],
        "after_collision_pass_count": after["collision_pass_count"],
        "before": before,
        "after": after,
        "safety_levels": safety_levels,
        "note": "所有档位复用同一批已保存q_goal，不重新IK。",
    }


def _flatten_pair_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_groups = (
        ("actual_urdf", report.get("baseline", {}).get("actual_urdf", [])),
        ("parameterized_baseline_proxy", report.get("baseline", {}).get("parameterized_baseline_proxy", [])),
        ("legacy_actual_urdf", report.get("baseline_before_proxy_correction", {}).get("actual_urdf", [])),
        (
            "legacy_parameterized_baseline_proxy",
            report.get("baseline_before_proxy_correction", {}).get("parameterized_baseline_proxy", []),
        ),
    )
    for model_name, states in baseline_groups:
        for state in states:
            for category in ("self_pairs", "tower_pairs"):
                for pair in state.get(category, []):
                    rows.append(
                        {
                            "model": model_name,
                            "state": state.get("state"),
                            "category": category,
                            **pair,
                        }
                    )
    return rows


def run_audit(
    paths: ProjectPaths,
    *,
    q_goals_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    output_dir = output_dir or paths.repo_root / "models" / "design_results" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    states = _baseline_states(paths)
    geometry = BaselineGeometry.from_project(paths)
    # Keep the candidate fixed; this is an audit, not another architecture search.
    spec = fixed_axis8_spec(geometry)
    model = MorphologyModel(spec)
    baseline_spec = MorphologySpec.baseline_8r(geometry)
    baseline_model = MorphologyModel(baseline_spec)
    legacy_spec = replace(
        baseline_spec,
        joint_proxy_radius_m=0.065,
        joint_proxy_radii_m=np.full(4, 0.065, dtype=float),
        link_proxy_radii_m=np.full(4, 0.065, dtype=float),
        right_terminal_span_is_suction_offset=False,
    )
    legacy_model = MorphologyModel(legacy_spec)
    body_poses, body_rows = _actual_body_poses(paths, states)
    frame_check = _baseline_frame_check(paths, states, geometry, baseline_model, body_poses)
    baseline_before = _audit_baseline(paths, states, geometry, legacy_spec, legacy_model, body_poses)
    baseline = _audit_baseline(paths, states, geometry, baseline_spec, baseline_model, body_poses)
    initial_state = model.forward(
        geometry.baseline_joint_vector(states[0].q_urdf),
        body_pose=_pose_matrix(body_poses[states[0].name]),
    )
    right_link3 = _right_link3_report(
        geometry,
        spec,
        initial_state,
        body_mesh=_mesh_report(geometry.central_mesh_path),
    )
    q_goals, q_status = _load_explicit_q_goals(q_goals_path)
    q_report = _audit_q_goals(paths, geometry, spec, q_goals)
    joint_spacing_m = float(baseline_spec.link_lengths_m[1])
    legacy_joint_total_radius_m = float(legacy_spec.joint_proxy_radius_m + legacy_spec.collision_inflation_m)
    proxy_diagnosis = {
        "classification": "PROXY_TOO_CONSERVATIVE",
        "affected_pairs": ["left_joint_2↔left_joint_3", "right_joint_2↔right_joint_3"],
        "legacy_observed_proxy_clearance_m": float(joint_spacing_m - 2.0 * legacy_joint_total_radius_m),
        "legacy_observed_proxy_penetration_m": float(max(0.0, 2.0 * legacy_joint_total_radius_m - joint_spacing_m)),
        "joint_center_spacing_m": joint_spacing_m,
        "legacy_joint_sphere_radius_m": legacy_joint_total_radius_m,
        "corrected_joint_sphere_radius_m": float(baseline_spec.joint_proxy_radius_m + baseline_spec.collision_inflation_m),
        "actual_urdf_non_adjacent_self_collision_count": [
            len(row["non_adjacent_self_collisions"]) for row in baseline["actual_urdf"]
        ],
        "legacy_parameterized_baseline_non_adjacent_self_collision_count": [
            len(row["non_adjacent_self_collisions"]) for row in baseline_before["parameterized_baseline_proxy"]
        ],
        "parameterized_baseline_non_adjacent_self_collision_count": [
            len(row["non_adjacent_self_collisions"])
            for row in baseline["parameterized_baseline_proxy"]
        ],
        "central_body_right_link_3_baseline_status": (
            "NO_NEAR_CONTACT_WITHIN_AUDIT_WINDOW"
            if all(
                _pair_by_name(row["self_pairs"], "central_body", "right_link_3") is None
                for row in baseline["parameterized_baseline_proxy"]
            )
            else "NEAR_CONTACT_RECORDED"
        ),
        "terminal_proxy_correction": "删除right_terminal_span_is_suction_offset对应的重复J8→吸盘capsule，保留真实L8.STL末端网格。",
        "note": "相邻关节球体的重叠来自错误复用的0.065 m关节半径；现改为独立0.040 m审计代理并保留0.005 m膨胀。该值不是最终电机CAD尺寸。",
    }
    report: dict[str, object] = {
        "design_name": spec.name,
        "architecture": "fixed finite YAW-PITCH-YAW-PITCH",
        "link_lengths_m": _array(spec.link_lengths_m),
        "collision_proxy": {
            "link_proxy_radius_m": spec.link_proxy_radius_m,
            "joint_proxy_radius_m": spec.joint_proxy_radius_m,
            "collision_inflation_m": spec.collision_inflation_m,
            "central_mesh": _mesh_report(geometry.central_mesh_path),
            "left_endpoint_mesh": _mesh_report(spec.left_endpoint_mesh_path),
            "right_endpoint_mesh": _mesh_report(spec.right_endpoint_mesh_path),
        },
        "proxy_dimension_source": _proxy_source_report(geometry, spec, baseline_spec),
        "states": [_json_state(state) for state in states],
        "actual_l4_body_poses": body_rows,
        "baseline_frame_check": frame_check,
        "right_link_3_topology": right_link3,
        "baseline_proxy_diagnosis": proxy_diagnosis,
        "q_goal_artifact": q_status,
        "baseline_before_proxy_correction": baseline_before,
        "baseline": baseline,
        "q_goal_audit": q_report,
        "conclusion": {
            "baseline_known_states_non_adjacent_self_collision_before_correction": any(
                row["non_adjacent_self_collisions"]
                for model_rows in (baseline_before["actual_urdf"], baseline_before["parameterized_baseline_proxy"])
                for row in model_rows
            ),
            "baseline_known_states_non_adjacent_self_collision_after_correction": any(
                row["non_adjacent_self_collisions"]
                for model_rows in (baseline["actual_urdf"], baseline["parameterized_baseline_proxy"])
                for row in model_rows
            ),
            "old_collision_result_invalidated_by_proxy_bug": bool(
                any(row["non_adjacent_self_collisions"] for row in baseline_before["parameterized_baseline_proxy"])
                and not any(row["non_adjacent_self_collisions"] for row in baseline["parameterized_baseline_proxy"])
            ),
            "q_goal_audit_completed": bool(q_goals),
            "reik_performed": False,
            "rrt_performed": False,
            "whole_tower_planning_performed": False,
            "pair_query_distance_m": PAIR_QUERY_DISTANCE_M,
        },
    }
    json_path = output_dir / "collision_audit.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_dir / "collision_audit_pairs.csv"
    rows = _flatten_pair_rows(report)
    fields = sorted({key for row in rows for key in row}) if rows else ["model", "state", "category"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="参数化8R独立碰撞审计，不运行IK/RRT")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--q-goals-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    report = run_audit(paths, q_goals_path=args.q_goals_npz, output_dir=args.output_dir)
    print("Collision audit completed")
    print("design:", report["design_name"])
    print("baseline frame check:", report["baseline_frame_check"]["status"], report["baseline_frame_check"]["max_position_error_m"])
    for model_name in ("actual_urdf", "parameterized_baseline_proxy"):
        rows = report["baseline"][model_name]
        print(model_name, "non-adjacent self collision counts:", [len(row["non_adjacent_self_collisions"]) for row in rows])
    print("right_link_3 direct central connection:", report["right_link_3_topology"]["directly_connected_to_central_body"])
    print("q_goal artifact:", report["q_goal_artifact"]["status"])
    print("q_goal audit:", report["q_goal_audit"]["status"])


if __name__ == "__main__":
    main()
