import trimesh
import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "Tower.STL"
output_path = BASE_DIR / "tower_work.stl"
meta_path = BASE_DIR / "tower_work_transform.json"

mesh = trimesh.load_mesh(input_path, force="mesh")

# 原始单位假设是 mm，转成 m
scale = 0.001
mesh.apply_scale(scale)

# 计算包围盒
bounds = mesh.bounds
min_xyz = bounds[0]
max_xyz = bounds[1]
center = (min_xyz + max_xyz) / 2

# 把模型中心移到原点附近，底部放到 z=0
translation = np.array([
    -center[0],
    -center[1],
    -min_xyz[2]
])

mesh.apply_translation(translation)

mesh.export(output_path)

meta = {
    "input": str(input_path),
    "output": str(output_path),
    "scale": scale,
    "translation_after_scale": translation.tolist()
}

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print("saved:", output_path)
print("bounds after:", mesh.bounds)
print("transform saved:", meta_path)