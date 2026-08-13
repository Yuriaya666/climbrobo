"""关节空间双向RRT-Connect。

该模块只负责通用的状态空间搜索；节点合法性和边碰撞检查由调用方提供，
因此不会绕开项目统一的碰撞规则。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


StateValidity = Callable[[np.ndarray], bool]
SegmentValidity = Callable[[np.ndarray, np.ndarray], bool]


@dataclass
class _Tree:
    states: list[np.ndarray]
    parents: list[int]


@dataclass(frozen=True)
class RRTStats:
    """一次RRT运行的轻量统计，不改变原有plan返回值。"""

    iterations: int = 0
    tree_nodes: int = 0
    start_valid: bool = False
    goal_valid: bool = False


class RRTConnect:
    """在有限关节范围内搜索连接起点和终点的无碰撞路径。"""

    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        *,
        step_size_rad: float = 0.18,
        max_iterations: int = 2500,
        goal_bias: float = 0.15,
        edge_resolution_rad: float = 0.06,
        random_seed: int = 20260812,
    ) -> None:
        self.lower = np.asarray(lower_limits, dtype=float)
        self.upper = np.asarray(upper_limits, dtype=float)
        if self.lower.shape != self.upper.shape or self.lower.ndim != 1:
            raise ValueError("关节上下限shape不一致")
        if np.any(self.upper <= self.lower):
            raise ValueError("关节上限必须大于下限")
        self.step_size_rad = float(step_size_rad)
        self.max_iterations = int(max_iterations)
        self.goal_bias = float(goal_bias)
        self.edge_resolution_rad = float(edge_resolution_rad)
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = int(random_seed)
        self.last_stats = RRTStats()

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        is_state_valid: StateValidity,
        is_segment_valid: SegmentValidity,
    ) -> list[np.ndarray] | None:
        start = self._clip_state(start)
        goal = self._clip_state(goal)
        start_valid = bool(is_state_valid(start))
        goal_valid = bool(is_state_valid(goal))
        if not start_valid or not goal_valid:
            self.last_stats = RRTStats(
                iterations=0,
                tree_nodes=0,
                start_valid=start_valid,
                goal_valid=goal_valid,
            )
            return None
        tree_a = _Tree([start.copy()], [-1])
        tree_b = _Tree([goal.copy()], [-1])

        for iteration in range(self.max_iterations):
            sample = goal if self.rng.random() < self.goal_bias else self.rng.uniform(self.lower, self.upper)
            index_a, reached_a = self._extend(tree_a, sample, is_state_valid, is_segment_valid)
            if index_a is None:
                tree_a, tree_b = tree_b, tree_a
                continue
            bridge_index, reached_b = self._connect(
                tree_b,
                tree_a.states[index_a],
                is_state_valid,
                is_segment_valid,
            )
            if bridge_index is not None and reached_b:
                path_a = self._path_to_root(tree_a, index_a)
                path_b = self._path_to_root(tree_b, bridge_index)
                # 两棵树的根方向不同，第二段需要反转后拼接。
                path = path_a + list(reversed(path_b))
                # 树在迭代中会交换，连接结果可能是goal到start，统一成start到goal。
                if np.linalg.norm(path[0] - start) > np.linalg.norm(path[-1] - start):
                    path.reverse()
                if np.linalg.norm(path[-1] - path[0]) < 1e-10:
                    self.last_stats = RRTStats(
                        iterations=iteration + 1,
                        tree_nodes=len(tree_a.states) + len(tree_b.states),
                        start_valid=True,
                        goal_valid=True,
                    )
                    return [path[0].copy()]
                self.last_stats = RRTStats(
                    iterations=iteration + 1,
                    tree_nodes=len(tree_a.states) + len(tree_b.states),
                    start_valid=True,
                    goal_valid=True,
                )
                return self._deduplicate(path)
            tree_a, tree_b = tree_b, tree_a
        self.last_stats = RRTStats(
            iterations=self.max_iterations,
            tree_nodes=len(tree_a.states) + len(tree_b.states),
            start_valid=True,
            goal_valid=True,
        )
        return None

    def _extend(self, tree: _Tree, target: np.ndarray, is_state_valid: StateValidity,
                is_segment_valid: SegmentValidity) -> tuple[int | None, bool]:
        nearest = self._nearest(tree, target)
        candidate = self._steer(tree.states[nearest], target)
        if np.linalg.norm(candidate - tree.states[nearest]) < 1e-10:
            return nearest, True
        if not is_segment_valid(tree.states[nearest], candidate) or not is_state_valid(candidate):
            return None, False
        tree.states.append(candidate)
        tree.parents.append(nearest)
        reached = np.linalg.norm(candidate - target) <= self.step_size_rad
        return len(tree.states) - 1, reached

    def _connect(self, tree: _Tree, target: np.ndarray, is_state_valid: StateValidity,
                 is_segment_valid: SegmentValidity) -> tuple[int | None, bool]:
        latest = None
        while True:
            latest, reached = self._extend(tree, target, is_state_valid, is_segment_valid)
            if latest is None:
                return None, False
            if reached:
                return latest, True

    def _nearest(self, tree: _Tree, target: np.ndarray) -> int:
        distances = [self._distance(state, target) for state in tree.states]
        return int(np.argmin(distances))

    def _steer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - source
        distance = float(np.linalg.norm(delta))
        if distance <= self.step_size_rad:
            return target.copy()
        return source + delta * (self.step_size_rad / distance)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        scale = np.maximum(self.upper - self.lower, 1e-6)
        return float(np.linalg.norm((a - b) / scale))

    def _clip_state(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=float)
        if value.shape != self.lower.shape:
            raise ValueError("RRT状态shape不正确")
        return np.clip(value, self.lower, self.upper)

    def _path_to_root(self, tree: _Tree, index: int) -> list[np.ndarray]:
        path = []
        while index >= 0:
            path.append(tree.states[index].copy())
            index = tree.parents[index]
        return list(reversed(path))

    @staticmethod
    def _deduplicate(path: list[np.ndarray]) -> list[np.ndarray]:
        result = [path[0]]
        for state in path[1:]:
            if np.linalg.norm(state - result[-1]) > 1e-10:
                result.append(state)
        return result
