"""连续可附着中心线的数据结构与读写工具。

中心线用高分辨率折线保存，但规划时使用弧长 ``s`` 作为连续变量。
文件采用拼接数组加offset，避免NumPy object array带来的兼容性问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environment.transforms import normalize
from environment.attachment_semantics import (
    SURFACE_TO_LEGACY_FOOT,
    canonical_surface_name,
)


@dataclass(frozen=True)
class AttachLineSample:
    """中心线上一个由弧长插值得到的附着位置。"""

    segment_id: int
    s_m: float
    xyz_m: np.ndarray
    normal: np.ndarray
    uv_m: np.ndarray


@dataclass(frozen=True)
class AttachLineSet:
    """一个附着面的若干连续可附着线段。

    ``foot_name``保留用于读取旧NPZ；规划语义使用独立的``surface_name``。
    """

    foot_name: str
    segment_ids: np.ndarray
    offsets: np.ndarray
    polyline_xyz_m: np.ndarray
    polyline_normal: np.ndarray
    polyline_uv_m: np.ndarray
    polyline_s_m: np.ndarray
    unit: str
    coordinate_frame: str
    source_path: Path | None = None
    surface_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "segment_ids", np.asarray(self.segment_ids, dtype=np.int32))
        object.__setattr__(self, "offsets", np.asarray(self.offsets, dtype=np.int64))
        object.__setattr__(self, "polyline_xyz_m", np.asarray(self.polyline_xyz_m, dtype=float))
        object.__setattr__(self, "polyline_normal", np.asarray(self.polyline_normal, dtype=float))
        object.__setattr__(self, "polyline_uv_m", np.asarray(self.polyline_uv_m, dtype=float))
        object.__setattr__(self, "polyline_s_m", np.asarray(self.polyline_s_m, dtype=float))
        surface = self.surface_name or self.foot_name
        object.__setattr__(self, "surface_name", canonical_surface_name(surface))
        self.validate()

    @property
    def segment_count(self) -> int:
        return int(len(self.segment_ids))

    def validate(self) -> None:
        if self.unit != "m":
            raise ValueError(f"attach line单位必须是m，实际为{self.unit!r}")
        if self.coordinate_frame != "tower_stl_global":
            raise ValueError(
                "attach line坐标系必须是tower_stl_global，"
                f"实际为{self.coordinate_frame!r}"
            )
        if self.segment_ids.ndim != 1:
            raise ValueError("segment_ids必须是一维数组")
        if self.offsets.shape != (len(self.segment_ids) + 1,):
            raise ValueError("offsets长度必须等于segment数量加1")
        if len(self.segment_ids) == 0 or self.offsets[0] != 0:
            raise ValueError("至少需要一个segment，且offsets必须从0开始")
        if np.any(np.diff(self.offsets) < 2):
            raise ValueError("每个segment至少需要两个折线点")
        point_count = len(self.polyline_xyz_m)
        if self.offsets[-1] != point_count:
            raise ValueError("offsets末值必须等于折线点总数")
        if self.polyline_xyz_m.shape != (point_count, 3):
            raise ValueError("polyline_xyz_m必须是(N,3)")
        if self.polyline_normal.shape != (point_count, 3):
            raise ValueError("polyline_normal必须是(N,3)")
        if self.polyline_uv_m.shape != (point_count, 2):
            raise ValueError("polyline_uv_m必须是(N,2)")
        if self.polyline_s_m.shape != (point_count,):
            raise ValueError("polyline_s_m必须是一维数组")
        for segment_index in range(self.segment_count):
            start, end = self._slice(segment_index)
            s_values = self.polyline_s_m[start:end]
            if abs(float(s_values[0])) > 1e-9 or np.any(np.diff(s_values) < -1e-10):
                raise ValueError("每个segment的弧长必须从0开始单调递增")
            normal_norms = np.linalg.norm(self.polyline_normal[start:end], axis=1)
            if np.any(normal_norms < 1e-12):
                raise ValueError("中心线中不能存在零法向")

    def _segment_index(self, segment_id: int) -> int:
        matches = np.flatnonzero(self.segment_ids == int(segment_id))
        if len(matches) != 1:
            raise KeyError(f"找不到唯一segment_id={segment_id}")
        return int(matches[0])

    def _slice(self, segment_index: int) -> tuple[int, int]:
        return int(self.offsets[segment_index]), int(self.offsets[segment_index + 1])

    def segment_length_m(self, segment_id: int) -> float:
        index = self._segment_index(segment_id)
        start, end = self._slice(index)
        return float(self.polyline_s_m[end - 1] - self.polyline_s_m[start])

    def evaluate(self, segment_id: int, s_m: float) -> AttachLineSample:
        """按segment和弧长插值位置、法向和UV。"""

        index = self._segment_index(segment_id)
        start, end = self._slice(index)
        s_values = self.polyline_s_m[start:end]
        s_value = float(np.clip(s_m, s_values[0], s_values[-1]))
        right = int(np.searchsorted(s_values, s_value, side="right"))
        right = min(max(right, 1), len(s_values) - 1)
        left = right - 1
        denominator = float(s_values[right] - s_values[left])
        alpha = 0.0 if denominator < 1e-12 else (s_value - s_values[left]) / denominator

        xyz = (1.0 - alpha) * self.polyline_xyz_m[start + left] + alpha * self.polyline_xyz_m[start + right]
        normal = normalize(
            (1.0 - alpha) * self.polyline_normal[start + left]
            + alpha * self.polyline_normal[start + right],
            name="插值后的中心线法向",
        )
        uv = (1.0 - alpha) * self.polyline_uv_m[start + left] + alpha * self.polyline_uv_m[start + right]
        return AttachLineSample(
            segment_id=int(segment_id),
            s_m=s_value,
            xyz_m=xyz,
            normal=normal,
            uv_m=uv,
        )

    def sample_uniform(self, segment_id: int, spacing_m: float) -> list[AttachLineSample]:
        """按给定弧长间隔生成调试或粗搜索样本。"""

        if spacing_m <= 0.0:
            raise ValueError("spacing_m必须大于0")
        length = self.segment_length_m(segment_id)
        values = np.arange(0.0, length, spacing_m, dtype=float)
        if len(values) == 0 or values[-1] < length:
            values = np.append(values, length)
        return [self.evaluate(segment_id, float(value)) for value in values]

    def save_npz(self, path: Path) -> None:
        """保存为拼接数组格式的NPZ。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            segment_ids=self.segment_ids,
            offsets=self.offsets,
            polyline_xyz_m=self.polyline_xyz_m,
            polyline_normal=self.polyline_normal,
            polyline_uv_m=self.polyline_uv_m,
            polyline_s_m=self.polyline_s_m,
            unit=np.asarray(self.unit),
            coordinate_frame=np.asarray(self.coordinate_frame),
            foot_name=np.asarray(self.foot_name),
            surface_name=np.asarray(self.surface_name),
        )

    @classmethod
    def load_npz(cls, path: Path, expected_surface_name: str | None = None) -> "AttachLineSet":
        if not path.exists():
            raise FileNotFoundError(f"找不到连续附着线文件：{path}（新连续落点数据）")
        data = np.load(path, allow_pickle=False)
        required = {
            "segment_ids", "offsets", "polyline_xyz_m", "polyline_normal",
            "polyline_uv_m", "polyline_s_m", "unit", "coordinate_frame",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}缺少字段：{missing}")
        foot_name = str(data["foot_name"].item()) if "foot_name" in data else ""
        surface_name = str(data["surface_name"].item()) if "surface_name" in data else foot_name
        surface_name = canonical_surface_name(surface_name)
        if not foot_name:
            foot_name = SURFACE_TO_LEGACY_FOOT[surface_name]
        if expected_surface_name is not None:
            expected = canonical_surface_name(expected_surface_name)
            if surface_name != expected:
                raise ValueError(f"{path}应对应{expected}，实际为{surface_name}")
        return cls(
            foot_name=foot_name,
            segment_ids=data["segment_ids"],
            offsets=data["offsets"],
            polyline_xyz_m=data["polyline_xyz_m"],
            polyline_normal=data["polyline_normal"],
            polyline_uv_m=data["polyline_uv_m"],
            polyline_s_m=data["polyline_s_m"],
            unit=str(data["unit"].item()),
            coordinate_frame=str(data["coordinate_frame"].item()),
            source_path=path,
            surface_name=surface_name,
        )
