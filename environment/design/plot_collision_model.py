"""绘制当前参数化6R/8R碰撞模型的静态 PNG，并可选打开 PyBullet GUI。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import trimesh

from environment.design.collision_audit import fixed_axis8_spec
from environment.design.collision_proxy import MorphologyCollisionWorld
from environment.design.morphology import BaselineGeometry, MorphologyModel
from environment.design.task_suite import build_task_suite
from environment.paths import ProjectPaths


def _pose_points(path: Path, pose: np.ndarray, *, max_points: int = 10000) -> np.ndarray:
    mesh = trimesh.load_mesh(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    points = np.asarray(mesh.vertices, dtype=float)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
    return points @ pose[:3, :3].T + pose[:3, 3]


def _capsule_points(start: np.ndarray, end: np.ndarray, radius: float) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return start[None, :]
    axis = vector / length
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    points: list[np.ndarray] = []
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    for fraction in np.linspace(0.0, 1.0, 16):
        center = start + fraction * vector
        ring = np.asarray([center + radius * (np.cos(angle) * e1 + np.sin(angle) * e2) for angle in angles])
        points.append(ring)
    for center in (start, end):
        for polar in np.linspace(0.0, 0.5 * np.pi, 7):
            ring_radius = radius * np.sin(polar)
            offset = radius * np.cos(polar)
            direction = -axis if np.allclose(center, start) else axis
            ring_center = center + direction * offset
            ring = np.asarray([ring_center + ring_radius * (np.cos(angle) * e1 + np.sin(angle) * e2) for angle in angles])
            points.append(ring)
    return np.concatenate(points, axis=0)


def _sphere_points(center: np.ndarray, radius: float) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    theta = np.linspace(0.0, np.pi, 12)
    phi = np.linspace(0.0, 2.0 * np.pi, 24)
    rows = []
    for value in theta:
        rows.extend(
            center + radius * np.array([np.sin(value) * np.cos(angle), np.sin(value) * np.sin(angle), np.cos(value)])
            for angle in phi
        )
    return np.asarray(rows)


def _load_state(paths: ProjectPaths, model: MorphologyModel, q_goal: Path | None, state_name: str):
    if q_goal is not None:
        with np.load(q_goal, allow_pickle=True) as data:
            q = np.asarray(data["q"], dtype=float)
            task_id = str(np.asarray(data["task_id"]).item())
        task = next(item for item in build_task_suite(paths, max_targets_per_task=1) if item.task_id == task_id)
        return model.world_state_for_support(q, task.support_endpoint, np.block([[task.support_pose.rotation_matrix, task.support_pose.position[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])), f"q_goal:{task_id}"
    if state_name == "mid_limits":
        q = 0.5 * (model.spec.lower_limits + model.spec.upper_limits)
        return model.forward(q), "mid_limits / body identity"
    from environment.design.collision_audit import _actual_body_poses, _baseline_states

    geometry = BaselineGeometry.from_project(paths)
    states = _baseline_states(paths)
    body_poses, _ = _actual_body_poses(paths, states)
    selected = next(item for item in states if item.name == state_name)
    return model.forward(geometry.baseline_joint_vector(selected.q_urdf), body_pose=np.block([[body_poses[selected.name].rotation_matrix, body_poses[selected.name].position[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])), state_name


def _draw_view(
    output_path: Path,
    state,
    spec,
    geometry: BaselineGeometry,
    title: str,
    *,
    elev: float,
    azim: float,
) -> None:
    figure = plt.figure(figsize=(12, 9), dpi=160)
    axis = figure.add_subplot(111, projection="3d")
    all_points: list[np.ndarray] = []

    body_points = _pose_points(geometry.central_mesh_path, state.body_pose, max_points=18000)
    axis.scatter(body_points[:, 0], body_points[:, 1], body_points[:, 2], s=0.15, c="#7a5a42", alpha=0.28, label="central body L4 mesh")
    all_points.append(body_points)

    for endpoint, pose, path, color in (
        ("base_end", state.left_endpoint_link_pose, spec.left_endpoint_mesh_path, "#1b9e77"),
        ("l8_end", state.right_endpoint_link_pose, spec.right_endpoint_mesh_path, "#d95f02"),
    ):
        points = _pose_points(path, pose, max_points=5000)
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.3, c=color, alpha=0.45, label=f"{endpoint} mesh")
        all_points.append(points)

    for side, segments, joints, color, radii in (
        ("left", state.left_span_segments, state.left_joint_poses, "#2166ac", spec.link_proxy_radii_m),
        ("right", state.right_span_segments, state.right_joint_poses, "#b2182b", spec.link_proxy_radii_m),
    ):
        visible_count = len(segments) - 1
        for index, (start, end) in enumerate(segments):
            # The terminal segment is represented by the endpoint mesh, exactly
            # as in MorphologyCollisionWorld._build_world.
            if index >= visible_count:
                continue
            radius = float(radii[index] + spec.collision_inflation_m) if radii is not None and index < len(radii) else float(spec.link_proxy_radius_m + spec.collision_inflation_m)
            points = _capsule_points(start, end, radius)
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.25, c=color, alpha=0.20)
            axis.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, linewidth=1.0, linestyle="--")
            midpoint = 0.5 * (start + end)
            axis.text(midpoint[0], midpoint[1], midpoint[2], f"{side[0].upper()} L{index + 1}\nr={radius * 1000:.1f}mm", fontsize=7, color=color)
            all_points.append(points)
        for index, joint in enumerate(joints):
            original_index = spec.left_active_indices[index] if side == "left" else spec.right_active_indices[index]
            radius = float(spec.joint_proxy_radius_for_original_index(original_index) + spec.collision_inflation_m)
            points = _sphere_points(joint[:3, 3], radius)
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.3, c="#333333", alpha=0.25)
            axis.text(joint[0, 3], joint[1, 3], joint[2, 3], f"J{original_index + 1}", fontsize=7, color="#111111")
            all_points.append(points)

    cloud = np.concatenate(all_points, axis=0)
    lower = cloud.min(axis=0)
    upper = cloud.max(axis=0)
    center = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 0.2)
    axis.set_xlim(center[0] - span / 2, center[0] + span / 2)
    axis.set_ylim(center[1] - span / 2, center[1] + span / 2)
    axis.set_zlim(center[2] - span / 2, center[2] + span / 2)
    try:
        axis.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.view_init(elev=elev, azim=azim)
    axis.set_title(title)
    axis.legend(loc="upper left", fontsize=8)
    axis.text2D(
        0.02,
        0.02,
        "Capsule radii are measured per-link STL envelopes + 5 mm inflation;\njoint spheres use local STL evidence with a 40 mm audit cap, not a common 65 mm radius.",
        transform=axis.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _pybullet_view(paths: ProjectPaths, model: MorphologyModel, state, seconds: float) -> None:
    with MorphologyCollisionWorld(paths, model, gui=True) as world:
        world.update(state)
        for side, segments, color in (
            ("left", state.left_span_segments, [0.1, 0.4, 1.0]),
            ("right", state.right_span_segments, [1.0, 0.2, 0.1]),
        ):
            for start, end in segments:
                p.addUserDebugLine(start.tolist(), end.tolist(), color, lineWidth=3.0)
        for side, joints in (("left", state.left_joint_poses), ("right", state.right_joint_poses)):
            for index, joint in enumerate(joints):
                p.addUserDebugText(f"{side[0].upper()}J{index + 1}", joint[:3, 3].tolist(), textColorRGB=[1.0, 1.0, 1.0], textSize=1.0)
        deadline = time.time() + max(0.1, seconds)
        while time.time() < deadline and p.isConnected():
            time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--state", choices=("mid_limits", "initial", "step1_final", "step2_final"), default="mid_limits")
    parser.add_argument("--q-goal", type=Path, default=None)
    parser.add_argument("--pybullet", action="store_true")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    geometry = BaselineGeometry.from_project(paths)
    spec = fixed_axis8_spec(geometry)
    model = MorphologyModel(spec)
    state, title_state = _load_state(paths, model, args.q_goal, args.state)
    output_dir = args.output_dir or paths.repo_root / "models" / "design_results" / "plots"
    iso = output_dir / "collision_model_optimized_8r_isometric.png"
    side = output_dir / "collision_model_optimized_8r_side.png"
    title = f"{spec.name} collision model ({title_state})"
    _draw_view(iso, state, spec, geometry, title, elev=24.0, azim=-55.0)
    _draw_view(side, state, spec, geometry, title, elev=8.0, azim=0.0)
    print("saved:", iso)
    print("saved:", side)
    if args.pybullet:
        _pybullet_view(paths, model, state, args.seconds)


if __name__ == "__main__":
    main()
