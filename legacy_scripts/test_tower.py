import pybullet as p
import time
from pathlib import Path
import math

p.connect(p.GUI)

BASE_DIR = Path(__file__).resolve().parent
stl_path = BASE_DIR / "Tower.STL"

scale = [0.001, 0.001, 0.001]

# 用 TOWER.STL 创建碰撞体
collision_shape = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName=str(stl_path),
    meshScale=scale,
    flags=p.GEOM_FORCE_CONCAVE_TRIMESH
)

# 同一个 TOWER.STL 用来显示
visual_shape = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName=str(stl_path),
    meshScale=scale,
    rgbaColor=[0.7, 0.7, 0.7, 0.8]
)

tower_pos = [1.368, 0.842, -6.211]
tower_rpy_deg = [-36.0, -51.158, 0.0]
tower_quat = p.getQuaternionFromEuler([math.radians(v) for v in tower_rpy_deg])

tower_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=collision_shape,
    baseVisualShapeIndex=visual_shape,
    basePosition=tower_pos,
    baseOrientation=tower_quat
)

# 打开线框，看三角网格
p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)

p.resetDebugVisualizerCamera(
    cameraDistance=8,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 2]
)

print("TOWER.STL loaded as both visual and collision mesh.")

while True:
    p.stepSimulation()
    time.sleep(1 / 240)