"""6R/8R机构设计的分层任务评价器。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from environment.attach_lines import AttachLineSample
from environment.design.collision_proxy import MorphologyCollisionWorld, ProxyCollisionResult
from environment.design.morphology import MorphologyModel
from environment.design.task_suite import TaskSpec
from environment.one_step_planner import AttachmentPoseBuilder
from environment.rrt_connect import RRTConnect
from environment.transforms import RigidTransform, angle_between_vectors_rad


@dataclass(frozen=True)
class DesignEvaluationSettings:
    position_tolerance_m: float = 0.005
    normal_tolerance_deg: float = 3.0
    same_surface_distance_m: float = 0.124
    local_max_nfev: int = 220
    normal_max_nfev: int = 360
    seed_count: int = 5
    yaw_samples: int = 16
    random_seed: int = 20260815
    run_collision: bool = False
    run_trajectory: bool = False
    rrt_max_iterations: int = 1200
    rrt_seed_count: int = 2
    rrt_step_size_rad: float = 0.2
    rrt_edge_resolution_rad: float = 0.08


@dataclass(frozen=True)
class IKAttempt:
    target_segment_id: int
    target_s_m: float
    yaw_rad: float
    q: np.ndarray
    xyz: np.ndarray
    position_error_m: float
    normal_error_deg: float
    success: bool
    failure_type: str


@dataclass(frozen=True)
class TargetEvaluation:
    sample: AttachLineSample
    position_best: IKAttempt
    normal_best: IKAttempt
    normal_candidates: tuple[IKAttempt, ...] = ()
    goal_valid: bool = False
    goal_collision: ProxyCollisionResult | None = None
    straight_success: bool = False
    straight_first_failure: str | None = None
    rrt_success: bool = False
    rrt_path: tuple[np.ndarray, ...] = ()
    rrt_stats: tuple[dict[str, object], ...] = ()
    minimum_clearance_m: float | None = None
    failure_type: str = ""


@dataclass(frozen=True)
class TaskEvaluation:
    task_id: str
    morphology_name: str
    target_results: tuple[TargetEvaluation, ...]
    success: bool
    best_target_index: int | None
    failure_type: str
    runtime_s: float


def _pose_matrix(pose: RigidTransform) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = pose.rotation_matrix
    matrix[:3, 3] = pose.position
    return matrix


def _sample_pose(sample: AttachLineSample, yaw_rad: float) -> RigidTransform:
    from environment.candidates import CandidatePoint

    candidate = CandidatePoint(
        foot_name="foot1" if sample.xyz_m[1] > 0.03 else "foot2",
        point_id=sample.segment_id,
        region_id=sample.segment_id,
        xyz_m=sample.xyz_m,
        normal=sample.normal,
        uv_m=sample.uv_m,
        surface_name="surface1" if sample.xyz_m[1] > 0.03 else "surface2",
    )
    return AttachmentPoseBuilder().build(
        candidate,
        preferred_y_reference_world=np.array([0.0, 0.0, 1.0]),
        yaw_rad=float(yaw_rad),
    )


class MorphologyTaskEvaluator:
    """对一个MorphologyModel执行position、normal、碰撞和轨迹分层评价。"""

    def __init__(self, model: MorphologyModel, *, settings: DesignEvaluationSettings | None = None) -> None:
        self.model = model
        self.settings = settings or DesignEvaluationSettings()
        self.rng = np.random.default_rng(self.settings.random_seed)

    def _start_q(self, task: TaskSpec) -> np.ndarray:
        baseline = np.asarray(task.start_q, dtype=float)
        if self.model.spec.per_side_dof == 4:
            return baseline.copy()
        left = baseline[[3, 2, 1, 0]][list(self.model.spec.left_active_indices)]
        right = baseline[[4, 5, 6, 7]][list(self.model.spec.right_active_indices)]
        return np.concatenate((left, right))

    def _state(self, task: TaskSpec, q: np.ndarray):
        return self.model.world_state_for_support(
            q,
            task.support_endpoint,
            _pose_matrix(task.support_pose),
        )

    def _seeds(self, task: TaskSpec) -> list[np.ndarray]:
        lower = self.model.spec.lower_limits
        upper = self.model.spec.upper_limits
        start = np.clip(self._start_q(task), lower, upper)
        seeds = [start]
        for _ in range(max(0, self.settings.seed_count - 1)):
            seeds.append(self.rng.uniform(lower, upper))
        return seeds

    def _position_attempt(self, task: TaskSpec, sample: AttachLineSample) -> IKAttempt:
        lower = self.model.spec.lower_limits
        upper = self.model.spec.upper_limits
        target = np.asarray(sample.xyz_m, dtype=float)
        best: IKAttempt | None = None
        for seed in self._seeds(task):
            started = time.perf_counter()
            result = least_squares(
                lambda q: self._state(task, q).suction_pose(task.moving_endpoint)[:3, 3] - target,
                x0=np.clip(seed, lower, upper),
                bounds=(lower, upper),
                max_nfev=self.settings.local_max_nfev,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            q = np.clip(result.x, lower, upper)
            state = self._state(task, q)
            moving = state.suction_pose(task.moving_endpoint)
            attempt = IKAttempt(
                sample.segment_id,
                sample.s_m,
                0.0,
                q.copy(),
                moving[:3, 3].copy(),
                float(np.linalg.norm(moving[:3, 3] - target)),
                float("nan"),
                False,
                "POSITION_WORKSPACE",
            )
            if best is None or attempt.position_error_m < best.position_error_m:
                best = attempt
            _ = started
        assert best is not None
        return IKAttempt(
            **{**best.__dict__, "success": best.position_error_m <= self.settings.position_tolerance_m, "failure_type": "" if best.position_error_m <= self.settings.position_tolerance_m else "POSITION_WORKSPACE"}
        )

    def _normal_attempts(self, task: TaskSpec, sample: AttachLineSample) -> list[IKAttempt]:
        lower = self.model.spec.lower_limits
        upper = self.model.spec.upper_limits
        desired_z = -np.asarray(sample.normal, dtype=float)
        attempts: list[IKAttempt] = []
        yaw_values = np.linspace(0.0, 2.0 * math.pi, max(1, self.settings.yaw_samples), endpoint=False)
        seeds = self._seeds(task)
        for yaw in yaw_values:
            for seed in seeds:
                def residual(q: np.ndarray) -> np.ndarray:
                    pose = self._state(task, q).suction_pose(task.moving_endpoint)
                    return np.concatenate((pose[:3, 3] - sample.xyz_m, np.cross(pose[:3, 2], desired_z)))

                result = least_squares(
                    residual,
                    x0=np.clip(seed, lower, upper),
                    bounds=(lower, upper),
                    max_nfev=self.settings.normal_max_nfev,
                    xtol=1e-9,
                    ftol=1e-9,
                    gtol=1e-9,
                )
                q = np.clip(result.x, lower, upper)
                pose = self._state(task, q).suction_pose(task.moving_endpoint)
                position_error = float(np.linalg.norm(pose[:3, 3] - sample.xyz_m))
                normal_error = math.degrees(angle_between_vectors_rad(pose[:3, 2], desired_z))
                valid = position_error <= self.settings.position_tolerance_m and normal_error <= self.settings.normal_tolerance_deg
                attempt = IKAttempt(
                    sample.segment_id,
                    sample.s_m,
                    float(yaw),
                    q.copy(),
                    pose[:3, 3].copy(),
                    position_error,
                    normal_error,
                    valid,
                    "" if valid else ("POSITION_WORKSPACE" if position_error > self.settings.position_tolerance_m else "NORMAL_WORKSPACE"),
                )
                attempts.append(attempt)
        return attempts

    def _normal_attempt(self, task: TaskSpec, sample: AttachLineSample) -> IKAttempt:
        attempts = self._normal_attempts(task, sample)
        if not attempts:
            raise RuntimeError("normal IK没有产生结果")
        successful = [attempt for attempt in attempts if attempt.success]
        return min(successful or attempts, key=self._attempt_score)

    @staticmethod
    def _attempt_score(attempt: IKAttempt) -> float:
        normal = 0.0 if not np.isfinite(attempt.normal_error_deg) else attempt.normal_error_deg / 180.0
        return attempt.position_error_m + 0.01 * normal

    def evaluate_task(
        self,
        task: TaskSpec,
        *,
        target_limit: int | None = None,
        collision: bool | None = None,
        trajectory: bool | None = None,
    ) -> TaskEvaluation:
        started = time.perf_counter()
        use_collision = self.settings.run_collision if collision is None else collision
        use_trajectory = self.settings.run_trajectory if trajectory is None else trajectory
        targets = task.targets if target_limit is None else task.targets[:target_limit]
        world = None
        if use_collision or use_trajectory:
            from environment.paths import ProjectPaths

            world = MorphologyCollisionWorld(ProjectPaths.from_repo_root(), self.model, gui=False)
            world.__enter__()
        results: list[TargetEvaluation] = []
        try:
            for sample in targets:
                position_best = self._position_attempt(task, sample)
                normal_candidates = self._normal_attempts(task, sample)
                successful_normal = [attempt for attempt in normal_candidates if attempt.success]
                normal_best = min(successful_normal or normal_candidates, key=self._attempt_score)
                goal_valid = False
                goal_collision = None
                straight_success = False
                straight_failure = None
                rrt_success = False
                rrt_path: tuple[np.ndarray, ...] = ()
                rrt_stats: list[dict[str, object]] = []
                minimum_clearance = None
                failure_type = normal_best.failure_type

                if normal_best.success and use_collision and world is not None:
                    collision_candidates = []
                    for candidate in successful_normal:
                        state = self._state(task, candidate.q)
                        world.update(state)
                        candidate_collision = world.check(
                            allowed_endpoint_positions={
                                task.support_endpoint: task.support_xyz,
                                task.moving_endpoint: sample.xyz_m,
                            },
                            allowed_contact_radius_m=0.09,
                        )
                        collision_candidates.append((candidate, candidate_collision))
                        if candidate_collision.ok:
                            normal_best = candidate
                            goal_collision = candidate_collision
                            goal_valid = True
                            minimum_clearance = candidate_collision.minimum_clearance_m
                            break
                    if not goal_valid and collision_candidates:
                        normal_best, goal_collision = min(
                            collision_candidates,
                            key=lambda item: item[1].minimum_clearance_m,
                        )
                        minimum_clearance = goal_collision.minimum_clearance_m
                        failure_type = "CENTRAL_BODY_COLLISION" if goal_collision.critical_link == "central_body" else (goal_collision.kind or "TOWER_COLLISION")
                elif normal_best.success:
                    goal_valid = True

                if goal_valid and use_trajectory:
                    start_q = self._start_q(task)
                    straight_path = self._interpolate(start_q, normal_best.q)
                    straight_success, straight_failure, straight_clearance = self._check_path(
                        task, sample, straight_path, world
                    )
                    minimum_clearance = self._min_optional(minimum_clearance, straight_clearance)
                    if not straight_success and world is not None:
                        for seed_index in range(self.settings.rrt_seed_count):
                            planner = RRTConnect(
                                self.model.spec.lower_limits,
                                self.model.spec.upper_limits,
                                step_size_rad=self.settings.rrt_step_size_rad,
                                max_iterations=self.settings.rrt_max_iterations,
                                edge_resolution_rad=self.settings.rrt_edge_resolution_rad,
                                random_seed=self.settings.random_seed + seed_index,
                            )
                            path = planner.plan(
                                start_q,
                                normal_best.q,
                                is_state_valid=lambda q: self._state_valid(task, sample, q, world, allow_goal=False, allow_start=True),
                                is_segment_valid=lambda a, b: self._segment_valid(task, sample, a, b, world),
                            )
                            stats = planner.last_stats
                            rrt_stats.append({"seed": planner.random_seed, "iterations": stats.iterations, "tree_nodes": stats.tree_nodes, "success": path is not None})
                            if path is not None:
                                rrt_success = True
                                rrt_path = tuple(np.asarray(item, dtype=float) for item in path)
                                break
                        if not rrt_success:
                            failure_type = "NO_COLLISION_FREE_PATH"
                results.append(
                    TargetEvaluation(
                        sample=sample,
                        position_best=position_best,
                        normal_best=normal_best,
                        normal_candidates=tuple(normal_candidates),
                        goal_valid=goal_valid,
                        goal_collision=goal_collision,
                        straight_success=straight_success,
                        straight_first_failure=straight_failure,
                        rrt_success=rrt_success,
                        rrt_path=rrt_path,
                        rrt_stats=tuple(rrt_stats),
                        minimum_clearance_m=minimum_clearance,
                        failure_type=failure_type,
                    )
                )
        finally:
            if world is not None:
                world.__exit__(None, None, None)

        successful = [index for index, result in enumerate(results) if result.goal_valid and (not use_trajectory or result.straight_success or result.rrt_success)]
        if successful:
            best_index = max(successful, key=lambda index: results[index].sample.xyz_m[2])
            failure = ""
            success = True
        else:
            best_index = None
            success = False
            failure = self._summarize_failure(results)
        return TaskEvaluation(
            task_id=task.task_id,
            morphology_name=self.model.spec.name,
            target_results=tuple(results),
            success=success,
            best_target_index=best_index,
            failure_type=failure,
            runtime_s=time.perf_counter() - started,
        )

    def _state_valid(self, task, sample, q, world, *, allow_goal: bool, allow_start: bool) -> bool:
        state = self._state(task, q)
        world.update(state)
        positions = {task.support_endpoint: task.support_xyz}
        if allow_goal:
            positions[task.moving_endpoint] = sample.xyz_m
        if allow_start:
            positions[task.moving_endpoint] = task.moving_start_xyz
        return world.check(allowed_endpoint_positions=positions, allowed_contact_radius_m=0.09).ok

    def _segment_valid(self, task, sample, first, second, world) -> bool:
        count = max(2, int(np.linalg.norm(second - first) / self.settings.rrt_edge_resolution_rad) + 1)
        for index, q in enumerate(np.linspace(first, second, count)):
            if not self._state_valid(task, sample, q, world, allow_goal=index == count - 1, allow_start=index == 0):
                return False
        return True

    def _check_path(self, task, sample, path, world):
        if world is None:
            return True, None, None
        minimum = float("inf")
        for index, q in enumerate(path):
            if not self._state_valid(task, sample, q, world, allow_goal=index == len(path) - 1, allow_start=index == 0):
                state = self._state(task, q)
                world.update(state)
                report = world.check(allowed_endpoint_positions={task.support_endpoint: task.support_xyz}, allowed_contact_radius_m=0.09)
                return False, report.critical_link or report.kind or "COLLISION", minimum
            state = self._state(task, q)
            world.update(state)
            report = world.check(allowed_endpoint_positions={task.support_endpoint: task.support_xyz, task.moving_endpoint: sample.xyz_m}, allowed_contact_radius_m=0.09)
            minimum = min(minimum, report.minimum_clearance_m)
        return True, None, minimum

    @staticmethod
    def _interpolate(first: np.ndarray, second: np.ndarray, count: int = 30) -> tuple[np.ndarray, ...]:
        return tuple(np.linspace(first, second, count))

    @staticmethod
    def _min_optional(first, second):
        values = [value for value in (first, second) if value is not None and np.isfinite(value)]
        return min(values) if values else None

    @staticmethod
    def _summarize_failure(results: list[TargetEvaluation]) -> str:
        if not results:
            return "NO_TARGET"
        if all(not result.normal_best.success for result in results):
            return min((result.normal_best.failure_type for result in results), default="NORMAL_WORKSPACE")
        if any(result.goal_valid for result in results):
            return "NO_COLLISION_FREE_PATH"
        return next((result.failure_type for result in results if result.failure_type), "GOAL_COLLISION")
