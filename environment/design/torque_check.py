"""新参数化机构的准静态重力力矩估算。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from environment.design.morphology import BaselineGeometry, MorphologyModel
from environment.paths import ProjectPaths


@dataclass(frozen=True)
class TorqueEstimate:
    status: str
    max_joint_torque_nm: float
    per_joint_torque_nm: tuple[float, ...]
    mass_model: str
    note: str


def _mass_values(paths: ProjectPaths) -> dict[str, float]:
    root = ET.parse(paths.robot_urdf).getroot()
    result = {}
    for link in root.findall("link"):
        mass = link.find("inertial/mass")
        if mass is not None:
            result[link.attrib["name"]] = float(mass.attrib["value"])
    return result


def estimate_gravity_torque(
    paths: ProjectPaths,
    model: MorphologyModel,
    *,
    q: np.ndarray | None = None,
    body_pose: np.ndarray | None = None,
    gravity_m_s2: float = 9.81,
) -> TorqueEstimate:
    """用当前URDF质量标定每段线密度，估算设计杆件的重力力矩。"""

    masses = _mass_values(paths)
    if q is None:
        q = np.zeros(model.spec.dof, dtype=float)
    state = model.forward(np.asarray(q, dtype=float), body_pose=body_pose)
    left_mass_nominal = np.asarray([masses.get("L3", 0.0), masses.get("L2", 0.0), masses.get("L1", 0.0), masses.get("base_link", 0.0)])
    right_mass_nominal = np.asarray([masses.get("L6", 0.0), masses.get("L7", 0.0), masses.get("L8", 0.0), masses.get("L8", 0.0)])
    nominal_left = np.asarray(model.spec.left_full_nominal_lengths_m, dtype=float)
    nominal_right = np.asarray(model.spec.right_full_nominal_lengths_m, dtype=float)
    left_linear_density = left_mass_nominal / np.maximum(nominal_left, 1e-6)
    right_linear_density = right_mass_nominal / np.maximum(nominal_right, 1e-6)
    torques: list[float] = []

    for side, joint_poses, segments, active_indices, density in (
        ("left", state.left_joint_poses, state.left_span_segments, model.spec.left_active_indices, left_linear_density),
        ("right", state.right_joint_poses, state.right_span_segments, model.spec.right_active_indices, right_linear_density),
    ):
        for active_position, joint_pose in enumerate(joint_poses):
            joint_point = joint_pose[:3, 3]
            axis_world = joint_pose[:3, :3] @ (model.spec.left_full_axes[active_indices[active_position]] if side == "left" else model.spec.right_full_axes[active_indices[active_position]])
            torque = 0.0
            for segment_index, (start, end) in enumerate(segments):
                midpoint = 0.5 * (start + end)
                length = float(np.linalg.norm(end - start))
                local_density = density[segment_index] if segment_index < len(density) else float(np.mean(density))
                force = np.array([0.0, 0.0, -gravity_m_s2 * local_density * length])
                torque += abs(float(np.dot(np.cross(midpoint - joint_point, force), axis_world)))
            torques.append(torque)
    return TorqueEstimate(
        status="ESTIMATE_ONLY_NO_MOTOR_LIMITS",
        max_joint_torque_nm=float(max(torques, default=0.0)),
        per_joint_torque_nm=tuple(float(value) for value in torques),
        mass_model="URDF link masses mapped to parameterized segment lengths by nominal linear density",
        note="未在项目中发现每个电机的连续/峰值输出力矩上限，不能做通过/不通过判定。",
    )
