"""对称6R删对拓扑的枚举与统一搜索入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from environment.design.length_optimizer import DesignSearchResult, LengthSearchConfig, optimize_lengths
from environment.design.morphology import BaselineGeometry
from environment.design.task_suite import TaskSpec


@dataclass(frozen=True)
class TopologySearchResult:
    candidates: tuple[DesignSearchResult, ...]

    @property
    def best(self) -> DesignSearchResult:
        if not self.candidates:
            raise ValueError("没有拓扑搜索结果")
        return min(
            self.candidates,
            key=lambda result: (
                -result.task_success_count,
                -result.position_success_count,
                result.worst_position_error_m,
            ),
        )


def enumerate_symmetric_6r(
    geometry: BaselineGeometry,
    tasks: tuple[TaskSpec, ...],
    *,
    config: LengthSearchConfig,
    output_dir: Path | None = None,
) -> TopologySearchResult:
    """只枚举当前4R侧链中的四种镜像删一对关节拓扑。"""

    results = []
    for remove_pair_index in range(4):
        results.append(
            optimize_lengths(
                geometry,
                tasks,
                kind="6r",
                remove_pair_index=remove_pair_index,
                config=config,
                output_dir=output_dir,
            )
        )
    return TopologySearchResult(tuple(results))

