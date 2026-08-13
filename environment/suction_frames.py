from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from environment.transforms import (
    RigidTransform,
    build_frame_from_z_and_y_reference,
)


@dataclass(frozen=True)
class SuctionFrame:
    """一个吸盘功能坐标系相对于URDF link坐标系的定义。"""

    name: str
    link_name: str
    position: np.ndarray
    normal: np.ndarray
    y_reference: np.ndarray
    transform_link_to_suction: RigidTransform

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "SuctionFrame":
        for field_name in ("link", "position", "normal", "y_reference"):
            if field_name not in data:
                raise ValueError(f"{name}缺少字段：{field_name}")

        position = np.asarray(data["position"], dtype=float)
        normal = np.asarray(data["normal"], dtype=float)
        y_reference = np.asarray(data["y_reference"], dtype=float)
        if position.shape != (3,) or normal.shape != (3,) or y_reference.shape != (3,):
            raise ValueError(f"{name}的position、normal、y_reference都必须是长度为3的数组")

        # YAML里的normal就是吸盘功能坐标系Z轴在link坐标系下的方向。
        rotation = build_frame_from_z_and_y_reference(
            z_axis=normal,
            y_reference=y_reference,
        )
        transform = RigidTransform.from_rotation_matrix(position, rotation)
        return cls(
            name=name,
            link_name=str(data["link"]),
            position=position,
            normal=normal,
            y_reference=y_reference,
            transform_link_to_suction=transform,
        )


@dataclass(frozen=True)
class SuctionFrameSet:
    """规划阶段需要的两个吸盘坐标系。"""

    base_end: SuctionFrame
    l8_end: SuctionFrame
    source_path: Path
    units: str

    @classmethod
    def load(cls, path: Path) -> "SuctionFrameSet":
        if not path.exists():
            raise FileNotFoundError(f"找不到吸盘配置文件：{path}（吸盘功能坐标系配置）")

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("吸盘配置必须是YAML字典")

        units = str(config.get("units", ""))
        if units not in {"meter", "m"}:
            raise ValueError(f"吸盘配置单位必须是meter或m，实际是：{units!r}")

        frames = config.get("suction_frames")
        if not isinstance(frames, dict):
            raise ValueError("吸盘配置缺少suction_frames字典")

        for required_name in ("base_end", "l8_end"):
            if required_name not in frames:
                raise ValueError(f"吸盘配置缺少{required_name}")

        return cls(
            base_end=SuctionFrame.from_mapping("base_end", frames["base_end"]),
            l8_end=SuctionFrame.from_mapping("l8_end", frames["l8_end"]),
            source_path=path,
            units=units,
        )
