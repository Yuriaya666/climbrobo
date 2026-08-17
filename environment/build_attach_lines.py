"""从两个附着面STL构建连续可附着中心线。

选面、局部坐标和收缩距离沿用旧脚本已验证的约定；本模块只把收缩区域
转换为连续中心线，不修改任何原始STL或旧候选文件。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from environment.attach_lines import AttachLineSet
from environment.attachment_semantics import canonical_surface_name
from environment.paths import ProjectPaths
from environment.transforms import normalize


@dataclass(frozen=True)
class SurfaceDefinition:
    face_indices: np.ndarray
    center: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class LocalSurfaceFrame:
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray


def load_surface_mesh(path: Path) -> trimesh.Trimesh:
    if not path.exists():
        raise FileNotFoundError(f"找不到附着面STL：{path}（连续附着线输入模型）")
    mesh = trimesh.load_mesh(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"{path}没有有效三角网格")
    return mesh


def find_two_main_faces(mesh: trimesh.Trimesh) -> dict[str, object]:
    """复用旧脚本的主平行面识别逻辑。"""

    face_normals = np.asarray(mesh.face_normals, dtype=float)
    face_areas = np.asarray(mesh.area_faces, dtype=float)
    face_centers = np.asarray(mesh.triangles_center, dtype=float)
    cosine_threshold = float(np.cos(np.deg2rad(3.0)))
    sorted_faces = np.argsort(face_areas)[::-1]
    best_axis = None
    best_parallel_area = -1.0
    for face_index in sorted_faces[: min(500, len(sorted_faces))]:
        candidate_axis = normalize(face_normals[face_index], name="三角面法向")
        alignment = np.abs(face_normals @ candidate_axis)
        parallel_area = float(face_areas[alignment >= cosine_threshold].sum())
        if parallel_area > best_parallel_area:
            best_parallel_area = parallel_area
            best_axis = candidate_axis
    if best_axis is None:
        raise RuntimeError("没有找到附着面主方向")

    alignment = np.abs(face_normals @ best_axis)
    parallel_faces = np.where(alignment >= cosine_threshold)[0]
    positions = face_centers[parallel_faces] @ best_axis
    middle = 0.5 * (float(positions.min()) + float(positions.max()))
    face_a = parallel_faces[positions <= middle]
    face_b = parallel_faces[positions > middle]
    if len(face_a) == 0 or len(face_b) == 0:
        raise RuntimeError("主平行面分组为空")
    center_a = np.average(face_centers[face_a], axis=0, weights=face_areas[face_a])
    center_b = np.average(face_centers[face_b], axis=0, weights=face_areas[face_b])
    middle_center = 0.5 * (center_a + center_b)
    return {
        "face_a_indices": face_a,
        "face_b_indices": face_b,
        "center_a": center_a,
        "center_b": center_b,
        "normal_a": normalize(center_a - middle_center, name="A面外法向"),
        "normal_b": normalize(center_b - middle_center, name="B面外法向"),
    }


def selected_surfaces(meshes: dict[str, trimesh.Trimesh]) -> dict[str, SurfaceDefinition]:
    """保留旧代码的foot1/foot2附着面及法向选择。"""

    data = {name: find_two_main_faces(mesh) for name, mesh in meshes.items()}
    return {
        "foot1": SurfaceDefinition(
            face_indices=np.asarray(data["foot1"]["face_b_indices"], dtype=np.int64),
            center=np.asarray(data["foot1"]["center_b"], dtype=float),
            # foot1附着面外法向取A面方向
            normal=np.asarray(data["foot1"]["normal_a"], dtype=float),
        ),
        "foot2": SurfaceDefinition(
            face_indices=np.asarray(data["foot2"]["face_a_indices"], dtype=np.int64),
            center=np.asarray(data["foot2"]["center_a"], dtype=float),
            # foot2附着面外法向取B面方向
            normal=np.asarray(data["foot2"]["normal_b"], dtype=float),
        ),
    }


def build_local_frame(surface: SurfaceDefinition, mesh: trimesh.Trimesh) -> LocalSurfaceFrame:
    triangles = mesh.triangles[surface.face_indices]
    points = np.unique(np.round(triangles.reshape(-1, 3), decimals=10), axis=0)
    normal = normalize(surface.normal, name="附着面法向")
    relative = points - surface.center
    projected = points - (relative @ normal)[:, None] * normal[None, :]
    _, _, vh = np.linalg.svd(projected - surface.center, full_matrices=False)
    u_axis = normalize(vh[0], name="附着面U轴")
    if abs(float(np.dot(u_axis, normal))) > 1e-3:
        u_axis = normalize(vh[1], name="附着面U轴")
    v_axis = normalize(np.cross(normal, u_axis), name="附着面V轴")
    u_axis = normalize(np.cross(v_axis, normal), name="附着面U轴")
    return LocalSurfaceFrame(surface.center, u_axis, v_axis, normal)


def xyz_to_uv(points_xyz: np.ndarray, frame: LocalSurfaceFrame) -> np.ndarray:
    relative = np.asarray(points_xyz, dtype=float) - frame.origin
    return np.column_stack((relative @ frame.u_axis, relative @ frame.v_axis))


def build_uv_region(mesh: trimesh.Trimesh, surface: SurfaceDefinition, frame: LocalSurfaceFrame):
    triangles_xyz = mesh.triangles[surface.face_indices]
    points_xyz = triangles_xyz.reshape(-1, 3)
    points_xyz = points_xyz - ((points_xyz - frame.origin) @ frame.normal)[:, None] * frame.normal[None, :]
    triangles_uv = xyz_to_uv(points_xyz, frame).reshape(-1, 3, 2)
    polygons = [Polygon(triangle) for triangle in triangles_uv if Polygon(triangle).area > 1e-12]
    region = unary_union(polygons).buffer(0)
    if region.is_empty:
        raise RuntimeError("二维附着区域合并后为空")
    return region


def polygon_list(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        result: list[Polygon] = []
        for item in geometry.geoms:
            result.extend(polygon_list(item))
        return result
    return []


def _intervals_at_u(polygon: Polygon, u_value: float) -> list[tuple[float, float]]:
    min_u, min_v, max_u, max_v = polygon.bounds
    line = LineString([(u_value, min_v - 1e-8), (u_value, max_v + 1e-8)])
    intersection = polygon.intersection(line)
    geometries = []
    if isinstance(intersection, LineString):
        geometries = [intersection]
    elif isinstance(intersection, MultiLineString):
        geometries = list(intersection.geoms)
    elif isinstance(intersection, GeometryCollection):
        geometries = [item for item in intersection.geoms if isinstance(item, LineString)]
    intervals = []
    for geometry in geometries:
        values = np.asarray(geometry.coords, dtype=float)[:, 1]
        if len(values) >= 2 and abs(float(values[-1] - values[0])) > 1e-10:
            intervals.append((float(values.min()), float(values.max())))
    return sorted(intervals)


def _centerline_polylines(polygon: Polygon, sample_spacing_m: float) -> list[np.ndarray]:
    min_u, _, max_u, _ = polygon.bounds
    if max_u - min_u < 1e-10:
        return []
    u_values = np.arange(min_u, max_u + 0.5 * sample_spacing_m, sample_spacing_m)
    if u_values[-1] < max_u:
        u_values = np.append(u_values, max_u)
    runs: list[list[list[float]]] = []
    active: list[tuple[float, list[float]]] = []
    for u_value in u_values:
        intervals = _intervals_at_u(polygon, float(u_value))
        next_active: list[tuple[float, list[float]]] = []
        used = set()
        for previous_index, (previous_mid, points) in enumerate(active):
            candidates = [
                (index, interval)
                for index, interval in enumerate(intervals)
                if index not in used
                and interval[0] - 2.0 * sample_spacing_m <= previous_mid <= interval[1] + 2.0 * sample_spacing_m
            ]
            if not candidates:
                if len(points) >= 2:
                    runs.append(points)
                continue
            index, interval = min(candidates, key=lambda item: abs(np.mean(item[1]) - previous_mid))
            used.add(index)
            midpoint = float(np.mean(interval))
            points.append([float(u_value), midpoint])
            next_active.append((midpoint, points))
        for index, interval in enumerate(intervals):
            if index not in used:
                next_active.append((float(np.mean(interval)), [[float(u_value), float(np.mean(interval))]]))
        active = next_active
    for _, points in active:
        if len(points) >= 2:
            runs.append(points)
    return [np.asarray(points, dtype=float) for points in runs]


def _polyline_to_arrays(polyline_uv: np.ndarray, frame: LocalSurfaceFrame):
    xyz = frame.origin[None, :] + polyline_uv[:, 0, None] * frame.u_axis[None, :] + polyline_uv[:, 1, None] * frame.v_axis[None, :]
    normal = np.repeat(frame.normal[None, :], len(polyline_uv), axis=0)
    segment_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s_values = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return xyz, normal, s_values


def build_attach_line_set(
    mesh: trimesh.Trimesh,
    surface: SurfaceDefinition,
    *,
    foot_name: str,
    surface_name: str | None = None,
    sample_spacing_m: float = 0.01,
    shrink_distance_m: float = 0.062,
) -> AttachLineSet:
    if sample_spacing_m <= 0.0 or shrink_distance_m < 0.0:
        raise ValueError("sample_spacing_m必须大于0，shrink_distance_m不能小于0")
    frame = build_local_frame(surface, mesh)
    original_region = build_uv_region(mesh, surface, frame)
    shrunk_region = original_region.buffer(-shrink_distance_m).buffer(0)
    if shrunk_region.is_empty:
        raise RuntimeError(f"{foot_name}收缩后没有可附着区域")

    uv_lines = []
    for polygon in polygon_list(shrunk_region):
        uv_lines.extend(_centerline_polylines(polygon, sample_spacing_m))
    if not uv_lines:
        raise RuntimeError(f"{foot_name}没有生成连续中心线")

    xyz_parts, normal_parts, uv_parts, s_parts = [], [], [], []
    offsets = [0]
    for uv_line in uv_lines:
        xyz, normal, s_values = _polyline_to_arrays(uv_line, frame)
        xyz_parts.append(xyz)
        normal_parts.append(normal)
        uv_parts.append(uv_line)
        s_parts.append(s_values)
        offsets.append(offsets[-1] + len(uv_line))
    return AttachLineSet(
        foot_name=foot_name,
        segment_ids=np.arange(len(uv_lines), dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        polyline_xyz_m=np.concatenate(xyz_parts, axis=0),
        polyline_normal=np.concatenate(normal_parts, axis=0),
        polyline_uv_m=np.concatenate(uv_parts, axis=0),
        polyline_s_m=np.concatenate(s_parts, axis=0),
        unit="m",
        coordinate_frame="tower_stl_global",
        surface_name=surface_name or canonical_surface_name(foot_name),
    )


def build_all_attach_lines(paths: ProjectPaths, *, sample_spacing_m: float = 0.01) -> dict[str, AttachLineSet]:
    meshes = {
        "foot1": load_surface_mesh(paths.repo_root / "models" / "Adhension1.STL"),
        "foot2": load_surface_mesh(paths.repo_root / "models" / "Adhension2.STL"),
    }
    surfaces = selected_surfaces(meshes)
    result = {}
    for foot_name, mesh in meshes.items():
        result[foot_name] = build_attach_line_set(
            mesh,
            surfaces[foot_name],
            foot_name=foot_name,
            sample_spacing_m=sample_spacing_m,
        )
    return result


def validate_against_discrete_candidates(paths: ProjectPaths, lines: dict[str, AttachLineSet]) -> None:
    """检查旧离散候选是否落在对应连续中心线附近。"""

    for foot_name, line_set in lines.items():
        data = np.load(paths.candidate_npz(foot_name), allow_pickle=False)
        candidates = np.asarray(data["xyz_m"], dtype=float)
        distances = []
        for point in candidates:
            best = float("inf")
            for segment_id in line_set.segment_ids:
                start = int(line_set.offsets[int(segment_id)])
                end = int(line_set.offsets[int(segment_id) + 1])
                best = min(best, float(np.min(np.linalg.norm(line_set.polyline_xyz_m[start:end] - point, axis=1))))
            distances.append(best)
        distances_array = np.asarray(distances, dtype=float)
        print(
            f"{foot_name}旧候选到连续中心线距离：最大={distances_array.max():.6f} m，"
            f"中位数={np.median(distances_array):.6f} m"
        )


def visualize_attach_lines(paths: ProjectPaths, lines: dict[str, AttachLineSet]) -> None:
    """在PyBullet GUI中显示两组中心线和少量弧长采样点。"""

    import time
    import pybullet as p
    from environment.scene import PyBulletScene

    with PyBulletScene(paths, gui=True) as scene:
        scene.load_tower()
        colors = {"foot1": [1.0, 0.2, 0.1], "foot2": [0.1, 0.8, 1.0]}
        for foot_name, line_set in lines.items():
            color = colors[foot_name]
            for segment_id in line_set.segment_ids:
                start = int(line_set.offsets[int(segment_id)])
                end = int(line_set.offsets[int(segment_id) + 1])
                scene.draw_polyline(line_set.polyline_xyz_m[start:end], color, radius=0.0008)
                for sample in line_set.sample_uniform(int(segment_id), 1.0):
                    scene.draw_point(sample.xyz_m, color, size=2.5)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=35.0,
            cameraPitch=-20.0,
            cameraTargetPosition=[0.0, 0.0, 19.0],
        )
        print("GUI已显示foot1红色、foot2蓝色连续中心线；按Ctrl+C退出。", flush=True)
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="构建并检查连续附着中心线")
    parser.add_argument("--gui", action="store_true", help="用PyBullet显示中心线")
    args = parser.parse_args()
    paths = ProjectPaths.from_repo_root()
    lines = build_all_attach_lines(paths)
    validate_against_discrete_candidates(paths, lines)
    for foot_name, line_set in lines.items():
        output = paths.candidate_dir / f"{foot_name}_attach_lines.npz"
        line_set.save_npz(output)
        print(
            f"{foot_name}连续附着线：segments={line_set.segment_count}, "
            f"折线点={len(line_set.polyline_xyz_m)}, 输出={output}（连续落点几何数据）"
        )
    if args.gui:
        visualize_attach_lines(paths, lines)


if __name__ == "__main__":
    main()
