"""规划诊断记录和CSV导出。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IKDiagnostic:
    segment_id: int
    s_m: float
    xyz_m: np.ndarray
    yaw_rad: float
    vertical_progress_m: float
    position_error_m: float
    normal_error_deg: float
    failure_reason: str


def write_ik_diagnostics(path: Path, rows: list[IKDiagnostic]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "segment_id", "s_m", "x_m", "y_m", "z_m", "yaw_rad",
                "vertical_progress_m", "position_error_m", "normal_error_deg",
                "failure_reason",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.segment_id, row.s_m, *np.asarray(row.xyz_m, dtype=float).tolist(),
                    row.yaw_rad, row.vertical_progress_m, row.position_error_m,
                    row.normal_error_deg, row.failure_reason,
                ]
            )


@dataclass(frozen=True)
class PlanningDiagnostic:
    segment_id: int
    s_m: float
    xyz_m: np.ndarray
    yaw_rad: float
    vertical_progress_m: float
    position_error_m: float
    normal_error_deg: float
    failure_reason: str
    trajectory_method: str
    planning_time_s: float


def write_planning_diagnostics(path: Path, rows: list[PlanningDiagnostic]) -> None:
    """保存直线/RRT阶段的每个终点尝试。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "segment_id", "s_m", "x_m", "y_m", "z_m", "yaw_rad",
                "vertical_progress_m", "IK_position_error_m", "IK_normal_error_deg",
                "failure_reason", "trajectory_method", "planning_time_s",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.segment_id, row.s_m, *np.asarray(row.xyz_m, dtype=float).tolist(),
                    row.yaw_rad, row.vertical_progress_m, row.position_error_m,
                    row.normal_error_deg, row.failure_reason, row.trajectory_method,
                    row.planning_time_s,
                ]
            )


def append_ik_diagnostics(path: Path, rows: list[IKDiagnostic]) -> None:
    """追加IK诊断行，文件不存在时先写表头。"""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                [
                    "segment_id", "s_m", "x_m", "y_m", "z_m", "yaw_rad",
                    "vertical_progress_m", "position_error_m", "normal_error_deg",
                    "failure_reason",
                ]
            )
        for row in rows:
            writer.writerow(
                [
                    row.segment_id, row.s_m, *np.asarray(row.xyz_m, dtype=float).tolist(),
                    row.yaw_rad, row.vertical_progress_m, row.position_error_m,
                    row.normal_error_deg, row.failure_reason,
                ]
            )
