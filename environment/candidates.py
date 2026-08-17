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
class CandidatePoint:
    """一个候选附着点，坐标已经在铁塔全局坐标系中。"""

    foot_name: str
    point_id: int
    region_id: int
    xyz_m: np.ndarray
    normal: np.ndarray
    uv_m: np.ndarray
    surface_name: str = ""

    def __post_init__(self) -> None:
        surface = self.surface_name or self.foot_name
        object.__setattr__(self, "surface_name", canonical_surface_name(surface))


@dataclass(frozen=True)
class CandidateSet:
    """一个附着表面对应的一组候选点。"""

    foot_name: str
    point_id: np.ndarray
    region_id: np.ndarray
    xyz_m: np.ndarray
    normal: np.ndarray
    uv_m: np.ndarray
    unit: str
    coordinate_frame: str
    source_path: Path
    surface_name: str = ""

    def __post_init__(self) -> None:
        surface = self.surface_name or self.foot_name
        object.__setattr__(self, "surface_name", canonical_surface_name(surface))

    @classmethod
    def load_npz(
        cls,
        path: Path,
        expected_foot_name: str | None = None,
        *,
        expected_surface_name: str | None = None,
    ) -> "CandidateSet":
        if not path.exists():
            raise FileNotFoundError(f"找不到候选点文件：{path}（候选点NPZ数据）")

        data = np.load(path, allow_pickle=False)
        required_keys = {
            "point_id",
            "xyz_m",
            "normal",
            "uv_m",
            "region_id",
            "unit",
            "coordinate_frame",
        }
        missing = sorted(required_keys - set(data.files))
        if missing:
            raise ValueError(f"{path}缺少字段：{missing}")

        foot_name = str(data["foot_name"].item()) if "foot_name" in data else ""
        surface_name = str(data["surface_name"].item()) if "surface_name" in data else foot_name
        surface_name = canonical_surface_name(surface_name)
        if not foot_name:
            foot_name = SURFACE_TO_LEGACY_FOOT[surface_name]
        unit = str(data["unit"].item())
        coordinate_frame = str(data["coordinate_frame"].item())

        result = cls(
            foot_name=foot_name,
            point_id=np.asarray(data["point_id"], dtype=np.int32),
            region_id=np.asarray(data["region_id"], dtype=np.int32),
            xyz_m=np.asarray(data["xyz_m"], dtype=float),
            normal=np.asarray(data["normal"], dtype=float),
            uv_m=np.asarray(data["uv_m"], dtype=float),
            unit=unit,
            coordinate_frame=coordinate_frame,
            source_path=path,
            surface_name=surface_name,
        )
        result.validate(
            expected_foot_name=expected_foot_name,
            expected_surface_name=expected_surface_name,
        )
        return result

    def validate(
        self,
        expected_foot_name: str | None = None,
        *,
        expected_surface_name: str | None = None,
    ) -> None:
        if expected_foot_name is not None and self.foot_name and self.foot_name != expected_foot_name:
            raise ValueError(
                f"{self.source_path}的foot_name应为{expected_foot_name}，实际为{self.foot_name}"
            )
        if expected_surface_name is not None and self.surface_name != canonical_surface_name(expected_surface_name):
            raise ValueError(
                f"{self.source_path}的surface_name应为{expected_surface_name}，实际为{self.surface_name}"
            )
        if self.unit != "m":
            raise ValueError(f"{self.source_path}的单位应为m，实际为{self.unit}")
        if self.coordinate_frame != "tower_stl_global":
            raise ValueError(
                f"{self.source_path}的坐标系应为tower_stl_global，实际为{self.coordinate_frame}"
            )

        point_count = len(self.point_id)
        expected_shapes = {
            "region_id": self.region_id.shape == (point_count,),
            "xyz_m": self.xyz_m.shape == (point_count, 3),
            "normal": self.normal.shape == (point_count, 3),
            "uv_m": self.uv_m.shape == (point_count, 2),
        }
        bad_shapes = [name for name, ok in expected_shapes.items() if not ok]
        if bad_shapes:
            raise ValueError(f"{self.source_path}字段shape不正确：{bad_shapes}")

        if point_count == 0:
            raise ValueError(f"{self.source_path}中没有候选点")

        normal_norms = np.linalg.norm(self.normal, axis=1)
        if np.any(normal_norms < 1e-12):
            raise ValueError(f"{self.source_path}存在零法向")

    def candidate_at_index(self, index: int) -> CandidatePoint:
        return CandidatePoint(
            foot_name=self.foot_name,
            point_id=int(self.point_id[index]),
            region_id=int(self.region_id[index]),
            xyz_m=np.asarray(self.xyz_m[index], dtype=float),
            normal=normalize(self.normal[index], name=f"{self.foot_name} normal"),
            uv_m=np.asarray(self.uv_m[index], dtype=float),
            surface_name=self.surface_name,
        )

    def sorted_indices_from_bottom(self) -> np.ndarray:
        """按z从低到高排序，同高度时按point_id稳定排序。"""

        return np.lexsort((self.point_id, self.xyz_m[:, 2]))

    def select_from_bottom(self, rank_from_bottom: int) -> CandidatePoint:
        """
        选择从下往上第rank_from_bottom个点。

        这里使用符合中文习惯的1-based编号：1表示最低的候选点。
        """

        if rank_from_bottom < 1:
            raise ValueError("rank_from_bottom必须从1开始")
        order = self.sorted_indices_from_bottom()
        if rank_from_bottom > len(order):
            raise ValueError(
                f"{self.foot_name}只有{len(order)}个点，不能选择第{rank_from_bottom}个"
            )
        return self.candidate_at_index(int(order[rank_from_bottom - 1]))

    def indices_by_distance_from(
        self,
        origin_xyz_m: np.ndarray,
        *,
        descending: bool = True,
    ) -> np.ndarray:
        """按候选点到origin_xyz_m的三维欧氏距离排序。"""

        origin = np.asarray(origin_xyz_m, dtype=float)
        distances = np.linalg.norm(self.xyz_m - origin[None, :], axis=1)
        order = np.argsort(distances)
        if descending:
            order = order[::-1]
        return order

    def distance_to(self, index: int, origin_xyz_m: np.ndarray) -> float:
        origin = np.asarray(origin_xyz_m, dtype=float)
        return float(np.linalg.norm(self.xyz_m[index] - origin))
