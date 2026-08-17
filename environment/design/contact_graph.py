"""连续attach line之上的粗粒度接触状态图数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.attach_lines import AttachLineSample, AttachLineSet


@dataclass(frozen=True)
class ContactNode:
    node_id: str
    surface_name: str
    segment_id: int
    s_m: float
    sample: AttachLineSample


@dataclass(frozen=True)
class ContactEdge:
    source_node_id: str
    target_node_id: str
    support_endpoint: str
    moving_endpoint: str
    status: str
    failure_type: str
    minimum_clearance_m: float | None = None


def coarse_contact_nodes(
    lines: AttachLineSet,
    *,
    spacing_m: float = 0.5,
    min_z_m: float | None = None,
    max_z_m: float | None = None,
) -> tuple[ContactNode, ...]:
    """用合理间隔对连续中心线采样，供全局拓扑图使用。"""

    nodes: list[ContactNode] = []
    for segment_id in lines.segment_ids:
        for sample in lines.sample_uniform(int(segment_id), spacing_m):
            z = float(sample.xyz_m[2])
            if min_z_m is not None and z < min_z_m:
                continue
            if max_z_m is not None and z > max_z_m:
                continue
            nodes.append(
                ContactNode(
                    node_id=f"{lines.surface_name}:{sample.segment_id}:{sample.s_m:.6f}",
                    surface_name=lines.surface_name,
                    segment_id=sample.segment_id,
                    s_m=sample.s_m,
                    sample=sample,
                )
            )
    return tuple(nodes)


def first_upward_node(
    nodes: tuple[ContactNode, ...],
    *,
    current_z_m: float,
    min_progress_m: float = 0.02,
) -> ContactNode | None:
    candidates = [node for node in nodes if node.sample.xyz_m[2] > current_z_m + min_progress_m]
    if not candidates:
        return None
    return min(candidates, key=lambda node: (node.sample.xyz_m[2], node.segment_id, node.s_m))


def graph_summary(nodes: tuple[ContactNode, ...]) -> dict[str, object]:
    if not nodes:
        return {"node_count": 0, "min_z_m": None, "max_z_m": None}
    return {
        "node_count": len(nodes),
        "min_z_m": float(min(node.sample.xyz_m[2] for node in nodes)),
        "max_z_m": float(max(node.sample.xyz_m[2] for node in nodes)),
    }

