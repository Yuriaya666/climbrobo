from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrame
from environment.transforms import RigidTransform, angle_between_vectors_rad


@dataclass(frozen=True)
class SuctionIKResult:
    """数值IK求解结果。"""

    success: bool
    joints: np.ndarray
    position_error_m: float
    normal_error_deg: float
    iterations: int
    reason: str


class NumericalSuctionIKSolver:
    """
    面向吸盘功能坐标系的阻尼最小二乘IK。

    PyBullet自带IK对本URDF的惯性帧和姿态目标比较敏感，所以这里用
    PyBullet做正运动学，自己用有限差分求吸盘误差的雅可比。
    """

    def __init__(
        self,
        scene: PyBulletScene,
        suction_frame: SuctionFrame,
        *,
        position_tolerance_m: float,
        normal_tolerance_deg: float,
        max_iterations: int = 120,
        finite_difference_step: float = 1e-4,
        damping: float = 1e-3,
        max_joint_step_rad: float = 0.12,
        orientation_weight_m_per_rad: float = 0.05,
        orientation_mode: str = "full",
        jacobian_mode: str = "finite_difference",
    ) -> None:
        self.scene = scene
        self.suction_frame = suction_frame
        self.position_tolerance_m = float(position_tolerance_m)
        self.normal_tolerance_rad = math.radians(normal_tolerance_deg)
        self.max_iterations = int(max_iterations)
        self.finite_difference_step = float(finite_difference_step)
        self.damping = float(damping)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.orientation_weight_m_per_rad = float(orientation_weight_m_per_rad)
        if orientation_mode not in {"full", "normal_only", "position_only"}:
            raise ValueError("orientation_mode必须是full、normal_only或position_only")
        self.orientation_mode = orientation_mode
        if jacobian_mode not in {"finite_difference", "pybullet"}:
            raise ValueError("jacobian_mode必须是finite_difference或pybullet")
        self.jacobian_mode = jacobian_mode

    def solve(
        self,
        target_suction_pose: RigidTransform,
        seed_joints: list[np.ndarray],
    ) -> SuctionIKResult:
        results = self.solve_all(target_suction_pose, seed_joints)
        successful = [result for result in results if result.success]
        return min(successful or results, key=self._score)

    def solve_multi_start(
        self,
        target_suction_pose: RigidTransform,
        seeds: list[np.ndarray],
    ) -> SuctionIKResult:
        """显式执行多起点IK，并完整比较所有起点的结果。"""

        unique_seeds: list[np.ndarray] = []
        for seed in seeds:
            value = np.asarray(seed, dtype=float)
            if value.shape != (len(self.scene.joints),):
                raise ValueError("IK seed的shape与可动关节数量不一致")
            if not any(np.linalg.norm(value - old) < 1e-10 for old in unique_seeds):
                unique_seeds.append(value.copy())
        return self.solve(target_suction_pose, unique_seeds)

    def solve_all(
        self,
        target_suction_pose: RigidTransform,
        seed_joints: list[np.ndarray],
    ) -> list[SuctionIKResult]:
        """返回所有不同初值的结果，供轨迹层选择不同的IK分支。"""

        if not seed_joints:
            raise ValueError("至少需要一个IK初始种子")
        results: list[SuctionIKResult] = []
        unique_seeds: list[np.ndarray] = []
        for seed in seed_joints:
            value = np.asarray(seed, dtype=float)
            if value.shape != (len(self.scene.joints),):
                raise ValueError("IK seed的shape与可动关节数量不一致")
            if any(np.linalg.norm(value - old) < 1e-10 for old in unique_seeds):
                continue
            unique_seeds.append(value.copy())
            results.append(self._solve_from_seed(target_suction_pose, value))
        if not results:
            raise ValueError("没有有效的IK初始种子")
        return results

    def _solve_from_seed(
        self,
        target_suction_pose: RigidTransform,
        seed_joints: np.ndarray,
    ) -> SuctionIKResult:
        lower = self.scene.joint_lower_limits()
        upper = self.scene.joint_upper_limits()
        joints = self.scene.normalize_revolute_solutions(seed_joints)
        joints = np.minimum(np.maximum(joints, lower), upper)

        best_joints = joints.copy()
        best_position_error = float("inf")
        best_normal_error = float("inf")
        best_iteration = 0

        for iteration in range(self.max_iterations + 1):
            residual = self._residual(joints, target_suction_pose)
            position_error, normal_error = self._pose_errors(target_suction_pose)
            if self._is_better(position_error, normal_error, best_position_error, best_normal_error):
                best_joints = joints.copy()
                best_position_error = position_error
                best_normal_error = normal_error
                best_iteration = iteration

            normal_satisfied = (
                self.orientation_mode == "position_only"
                or normal_error <= self.normal_tolerance_rad
            )
            if position_error <= self.position_tolerance_m and normal_satisfied:
                return SuctionIKResult(
                    success=True,
                    joints=joints.copy(),
                    position_error_m=position_error,
                    normal_error_deg=math.degrees(normal_error),
                    iterations=iteration,
                    reason="收敛",
                )

            if iteration >= self.max_iterations:
                break

            jacobian = self._compute_jacobian(
                joints,
                residual,
                target_suction_pose,
            )
            normal_matrix = jacobian.T @ jacobian
            rhs = -jacobian.T @ residual
            damping_matrix = self.damping * np.eye(len(joints))
            try:
                delta = np.linalg.solve(normal_matrix + damping_matrix, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(normal_matrix + damping_matrix, rhs, rcond=None)[0]

            max_delta = float(np.max(np.abs(delta))) if len(delta) else 0.0
            if max_delta > self.max_joint_step_rad:
                delta *= self.max_joint_step_rad / max_delta

            joints = joints + delta
            joints = self.scene.normalize_revolute_solutions(joints)
            joints = np.minimum(np.maximum(joints, lower), upper)

        return SuctionIKResult(
            success=False,
            joints=best_joints,
            position_error_m=best_position_error,
            normal_error_deg=math.degrees(best_normal_error),
            iterations=best_iteration,
            reason=(
                f"未收敛：位置误差{best_position_error:.6f} m，"
                f"法向误差{math.degrees(best_normal_error):.3f} deg"
            ),
        )

    def _compute_jacobian(
        self,
        joints: np.ndarray,
        residual: np.ndarray,
        target_suction_pose: RigidTransform,
    ) -> np.ndarray:
        """选择已验证的原生雅可比或原有有限差分雅可比。"""

        if self.jacobian_mode == "finite_difference":
            return self._finite_difference_jacobian(joints, residual, target_suction_pose)

        linear, angular = self.scene.calculate_suction_jacobian(self.suction_frame)
        current_pose = self.scene.get_suction_pose(self.suction_frame)
        current_axes = current_pose.rotation_matrix
        target_axes = target_suction_pose.rotation_matrix

        orientation_map = np.zeros((3, 3), dtype=float)
        if self.orientation_mode == "normal_only":
            orientation_map = _skew(target_axes[:, 2]) @ _skew(current_axes[:, 2])
        else:
            for axis_index in range(3):
                orientation_map += 0.5 * (
                    _skew(target_axes[:, axis_index])
                    @ _skew(current_axes[:, axis_index])
                )

        return np.vstack(
            (
                -linear,
                self.orientation_weight_m_per_rad * orientation_map @ angular,
            )
        )


    def _finite_difference_jacobian(
        self,
        joints: np.ndarray,
        residual: np.ndarray,
        target_suction_pose: RigidTransform,
    ) -> np.ndarray:
        jacobian = np.zeros((6, len(joints)), dtype=float)
        lower = self.scene.joint_lower_limits()
        upper = self.scene.joint_upper_limits()

        for joint_index in range(len(joints)):
            perturbed = joints.copy()
            perturbed[joint_index] += self.finite_difference_step
            perturbed = self.scene.normalize_revolute_solutions(perturbed)
            perturbed = np.minimum(np.maximum(perturbed, lower), upper)
            actual_step = perturbed[joint_index] - joints[joint_index]
            if abs(actual_step) < 1e-12:
                continue
            perturbed_residual = self._residual(perturbed, target_suction_pose)
            jacobian[:, joint_index] = (perturbed_residual - residual) / actual_step

        return jacobian

    def _residual(
        self,
        joints: np.ndarray,
        target_suction_pose: RigidTransform,
    ) -> np.ndarray:
        self.scene.reset_joints(joints)
        current_pose = self.scene.get_suction_pose(self.suction_frame)
        position_error = target_suction_pose.position - current_pose.position
        orientation_error = self._orientation_error(current_pose, target_suction_pose)
        if self.orientation_mode == "normal_only":
            # 吸附约束只要求吸盘法向贴合；绕法向的切向旋转不影响接触。
            orientation_error = np.cross(current_pose.z_axis, target_suction_pose.z_axis)
        elif self.orientation_mode == "position_only":
            orientation_error = np.zeros(3, dtype=float)
        return np.concatenate(
            (
                position_error,
                self.orientation_weight_m_per_rad * orientation_error,
            )
        )

    def _pose_errors(self, target_suction_pose: RigidTransform) -> tuple[float, float]:
        current_pose = self.scene.get_suction_pose(self.suction_frame)
        position_error = float(np.linalg.norm(target_suction_pose.position - current_pose.position))
        normal_error = angle_between_vectors_rad(current_pose.z_axis, target_suction_pose.z_axis)
        return position_error, normal_error

    @staticmethod
    def _orientation_error(
        current_pose: RigidTransform,
        target_pose: RigidTransform,
    ) -> np.ndarray:
        current = current_pose.rotation_matrix
        target = target_pose.rotation_matrix
        return 0.5 * (
            np.cross(current[:, 0], target[:, 0])
            + np.cross(current[:, 1], target[:, 1])
            + np.cross(current[:, 2], target[:, 2])
        )

    @staticmethod
    def _is_better(
        position_error: float,
        normal_error: float,
        best_position_error: float,
        best_normal_error: float,
    ) -> bool:
        return position_error + 0.01 * normal_error < best_position_error + 0.01 * best_normal_error

    @staticmethod
    def _score(result: SuctionIKResult) -> float:
        return result.position_error_m + 0.01 * math.radians(result.normal_error_deg)


def _skew(vector: np.ndarray) -> np.ndarray:
    """返回向量对应的叉乘矩阵。"""

    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
