import trimesh
from pathlib import Path

base_dir = Path(__file__).resolve().parent

mesh = trimesh.load_mesh(
    base_dir / "Tower.STL",
    force="mesh"
)

print("原始三角面：", len(mesh.faces))

# 40万面先降到约10万面
visual_mesh = mesh.simplify_quadric_decimation(face_count=100000)

visual_mesh.export(base_dir / "Tower_visual.stl")

print("显示版三角面：", len(visual_mesh.faces))