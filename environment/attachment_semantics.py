"""机器人物理端与铁塔附着面的兼容语义。"""

from __future__ import annotations


# 旧版文件名仍然使用foot1/foot2；它们现在只表示历史surface数据文件。
LEGACY_FOOT_TO_SURFACE = {
    "foot1": "surface1",
    "foot2": "surface2",
}
SURFACE_TO_LEGACY_FOOT = {value: key for key, value in LEGACY_FOOT_TO_SURFACE.items()}
LEGACY_FOOT_TO_FRAME = {
    "foot1": "base_end",
    "foot2": "l8_end",
}
FRAME_TO_LEGACY_FOOT = {value: key for key, value in LEGACY_FOOT_TO_FRAME.items()}


def canonical_surface_name(name: str) -> str:
    """把旧foot文件标签或新surface名称统一成surface1/surface2。"""

    value = str(name)
    if value in LEGACY_FOOT_TO_SURFACE:
        return LEGACY_FOOT_TO_SURFACE[value]
    if value in SURFACE_TO_LEGACY_FOOT:
        return value
    raise ValueError(f"未知附着面名称：{name!r}，当前仅支持surface1/surface2")


def legacy_surface_file_name(surface_name: str) -> str:
    """返回现有NPZ文件使用的foot1/foot2兼容文件名。"""

    surface = canonical_surface_name(surface_name)
    return SURFACE_TO_LEGACY_FOOT[surface]


def canonical_frame_name(name: str) -> str:
    """把旧foot标签转换为真实机器人端名称。"""

    value = str(name)
    if value in FRAME_TO_LEGACY_FOOT:
        return value
    if value in LEGACY_FOOT_TO_FRAME:
        return LEGACY_FOOT_TO_FRAME[value]
    raise ValueError(f"未知机器人物理端：{name!r}，当前仅支持base_end/l8_end")


def legacy_foot_name_for_frame(frame_name: str) -> str:
    """把真实物理端转换为旧轨迹字段使用的兼容标签。"""

    frame = canonical_frame_name(frame_name)
    return FRAME_TO_LEGACY_FOOT[frame]
