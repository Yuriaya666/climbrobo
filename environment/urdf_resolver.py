from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import TracebackType


class ResolvedUrdf:
    """
    为PyBullet生成临时URDF，把package://mesh路径改成绝对路径。

    原始URDF不做任何修改，临时文件在上下文退出时删除。
    """

    def __init__(self, source_urdf: Path, mesh_dir: Path) -> None:
        self.source_urdf = source_urdf
        self.mesh_dir = mesh_dir
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        if not self.source_urdf.exists():
            raise FileNotFoundError(f"找不到URDF：{self.source_urdf}（机器人URDF主文件）")
        if not self.mesh_dir.exists():
            raise FileNotFoundError(f"找不到mesh目录：{self.mesh_dir}（机器人mesh目录）")

        self._temporary_directory = tempfile.TemporaryDirectory(prefix="climbrobo_urdf_")
        output_path = Path(self._temporary_directory.name) / self.source_urdf.name

        tree = ET.parse(self.source_urdf)
        root = tree.getroot()
        for mesh in root.iter("mesh"):
            filename = mesh.attrib.get("filename")
            if not filename:
                continue
            mesh.attrib["filename"] = str(self._resolve_mesh_filename(filename))

        try:
            ET.indent(tree, space="  ")
        except AttributeError:
            pass
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        self.path = output_path
        return output_path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
        self.path = None

    def _resolve_mesh_filename(self, filename: str) -> Path:
        if filename.startswith("package://"):
            # 现有URDF格式是 package://包名/meshes/L1.STL。
            relative = filename.removeprefix("package://")
            parts = Path(relative).parts
            mesh_name = parts[-1]
            resolved = self.mesh_dir / mesh_name
        else:
            candidate = Path(filename)
            if candidate.is_absolute():
                resolved = candidate
            else:
                resolved = self.source_urdf.parent / candidate

        if not resolved.exists():
            raise FileNotFoundError(f"URDF引用的mesh不存在：{resolved}（机器人mesh文件）")
        return resolved.resolve()
