import math
import time
from pathlib import Path
import numpy as np

import pybullet as p
import pybullet_data


# =========================================================
# 1. 文件路径
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# 原始铁塔：碰撞检测使用
tower_collision_path = BASE_DIR / "Tower.STL"

# 简化铁塔：GUI 显示使用
tower_visual_path = BASE_DIR / "Tower_visual.STL"

# 机器人 URDF
robot_path = (
    BASE_DIR
    / "00样机（更换电机）-x12.SLDASM"
    / "urdf"
    / "00样机（更换电机）-x12.SLDASM.urdf"
)

if not tower_collision_path.exists():
    raise FileNotFoundError(f"找不到铁塔碰撞模型：{tower_collision_path}")

if not tower_visual_path.exists():
    raise FileNotFoundError(f"找不到铁塔显示模型：{tower_visual_path}")

if not robot_path.exists():
    raise FileNotFoundError(f"找不到机器人 URDF：{robot_path}")


# =========================================================
# 2. 启动 PyBullet
# =========================================================

client_id = p.connect(p.GUI)

if client_id < 0:
    raise RuntimeError("PyBullet GUI 启动失败")

p.setGravity(0, 0, -9.81)

# 加载模型时暂时关闭渲染
p.configureDebugVisualizer(
    p.COV_ENABLE_RENDERING,
    0
)


# =========================================================
# 3. 加载地板
# =========================================================

plane_path = Path(pybullet_data.getDataPath()) / "plane.urdf"

plane_id = p.loadURDF(
    fileName=str(plane_path),
    basePosition=[0, 0, 0]  
)


# =========================================================
# 4. 加载机器人
# =========================================================

initial_robot_position = [-0.5, -0.5, 1]
initial_robot_euler = [0.0, 0.0, 0.0]

robot_id = p.loadURDF(
    fileName=str(robot_path),
    basePosition=initial_robot_position,
    baseOrientation=p.getQuaternionFromEuler(
        initial_robot_euler
    ),
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION
)


# =========================================================
# 5. 加载铁塔
# =========================================================

# 原始高精度 STL 作为碰撞体
tower_collision = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName=str(tower_collision_path),
    flags=p.GEOM_FORCE_CONCAVE_TRIMESH
)

# 简化 STL 仅用于显示
tower_visual = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName=str(tower_visual_path),
    rgbaColor=[0.65, 0.65, 0.65, 1.0]
)

tower_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=tower_collision,
    baseVisualShapeIndex=tower_visual,
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1]
)


# =========================================================
# 6. 创建机器人整体位姿滑块
# =========================================================

# 根据铁塔实际尺寸，可修改滑块范围
base_x_slider = p.addUserDebugParameter(
    paramName="Base X",
    rangeMin=-5.0,
    rangeMax=5.0,
    startValue=initial_robot_position[0]
)

base_y_slider = p.addUserDebugParameter(
    paramName="Base Y",
    rangeMin=-5.0,
    rangeMax=5.0,
    startValue=initial_robot_position[1]
)

base_z_slider = p.addUserDebugParameter(
    paramName="Base Z",
    rangeMin=0.0,
    rangeMax=10.0,
    startValue=initial_robot_position[2]
)

base_roll_slider = p.addUserDebugParameter(
    paramName="Base Roll",
    rangeMin=-math.pi,
    rangeMax=math.pi,
    startValue=initial_robot_euler[0]
)

base_pitch_slider = p.addUserDebugParameter(
    paramName="Base Pitch",
    rangeMin=-math.pi,
    rangeMax=math.pi,
    startValue=initial_robot_euler[1]
)

base_yaw_slider = p.addUserDebugParameter(
    paramName="Base Yaw",
    rangeMin=-math.pi,
    rangeMax=math.pi,
    startValue=initial_robot_euler[2]
)


# =========================================================
# 7. 创建机器人关节滑块
# =========================================================

joint_sliders = []

for joint_id in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(
        robot_id,
        joint_id
    )

    joint_name = joint_info[1].decode(
        "utf-8",
        errors="ignore"
    )

    joint_type = joint_info[2]
    lower_limit = joint_info[8]
    upper_limit = joint_info[9]

    # 只为旋转关节和移动关节创建滑块
    if joint_type not in (
        p.JOINT_REVOLUTE,
        p.JOINT_PRISMATIC
    ):
        continue

    # URDF 中没有正常限位时使用默认范围
    if lower_limit >= upper_limit:
        if joint_type == p.JOINT_REVOLUTE:
            lower_limit = -math.pi
            upper_limit = math.pi
        else:
            lower_limit = -1.0
            upper_limit = 1.0

    current_position = p.getJointState(
        robot_id,
        joint_id
    )[0]

    slider_id = p.addUserDebugParameter(
        paramName=f"Joint {joint_id}: {joint_name}",
        rangeMin=lower_limit,
        rangeMax=upper_limit,
        startValue=current_position
    )

    joint_sliders.append(
        (joint_id, slider_id)
    )


def load_candidate_set(npz_path: str | Path) -> dict:
    """
    读取单只脚的候选点文件。
    """

    data = np.load(npz_path)

    return {
        "xyz": np.asarray(
            data["xyz_m"],
            dtype=float,
        ),
        "normal": np.asarray(
            data["normal"],
            dtype=float,
        ),
        "point_id": np.asarray(
            data["point_id"],
            dtype=int,
        ),
    }


def draw_candidate_set(
    points_xyz: np.ndarray,
    normals: np.ndarray,
    point_color: list[float],
    point_size: float = 6.0,
    visible_offset: float = 0.002,
    normal_length: float = 0.08,
    normal_stride: int = 20,
) -> list[int]:
    """
    在PyBullet中绘制一组候选点及部分法向。

    visible_offset:
        为避免候选点被铁塔表面遮住，
        沿法向向外偏移的显示距离，单位m。
        只影响显示，不改变原始数据。

    normal_stride:
        每隔多少个候选点绘制一根法向线。
    """

    points_xyz = np.asarray(
        points_xyz,
        dtype=float,
    )

    normals = np.asarray(
        normals,
        dtype=float,
    )

    # 仅用于显示，沿法向向外偏移2 mm
    visible_points = (
        points_xyz
        + visible_offset * normals
    )

    colors = np.repeat(
        np.asarray(
            point_color,
            dtype=float,
        )[None, :],
        repeats=len(visible_points),
        axis=0,
    )

    debug_ids = []

    # 一次性批量绘制所有候选点
    point_debug_id = p.addUserDebugPoints(
        pointPositions=visible_points.tolist(),
        pointColorsRGB=colors.tolist(),
        pointSize=point_size,
        lifeTime=0,
    )

    debug_ids.append(point_debug_id)

    # 稀疏绘制法向，避免画面过密
    for index in range(
        0,
        len(points_xyz),
        normal_stride,
    ):

        point = points_xyz[index]
        normal = normals[index]

        line_start = point
        line_end = (
            point
            + normal_length * normal
        )

        line_debug_id = p.addUserDebugLine(
            lineFromXYZ=line_start.tolist(),
            lineToXYZ=line_end.tolist(),
            lineColorRGB=[0.0, 1.0, 0.0],
            lineWidth=2.0,
            lifeTime=0,
        )

        debug_ids.append(line_debug_id)

    return debug_ids


# ---------------------------------------------------------
# 读取候选点
# ---------------------------------------------------------

foot1_data = load_candidate_set(
    "candidate_output/foot1_candidates.npz"
)

foot2_data = load_candidate_set(
    "candidate_output/foot2_candidates.npz"
)


foot1_points = foot1_data["xyz"]
foot1_normals = foot1_data["normal"]

foot2_points = foot2_data["xyz"]
foot2_normals = foot2_data["normal"]


print("foot1候选点数量：", len(foot1_points))
print("foot2候选点数量：", len(foot2_points))


# ---------------------------------------------------------
# 绘制候选点
# ---------------------------------------------------------

foot1_debug_ids = draw_candidate_set(
    points_xyz=foot1_points,
    normals=foot1_normals,
    point_color=[1.0, 0.0, 0.0],  # 红色
    point_size=7.0,
    visible_offset=0.002,
    normal_length=0.08,
    normal_stride=20,
)

foot2_debug_ids = draw_candidate_set(
    points_xyz=foot2_points,
    normals=foot2_normals,
    point_color=[0.0, 0.3, 1.0],  # 蓝色
    point_size=7.0,
    visible_offset=0.002,
    normal_length=0.08,
    normal_stride=20,
)



# =========================================================
# 8. 设置相机并恢复渲染
# =========================================================

p.resetDebugVisualizerCamera(
    cameraDistance=8,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 2]
)

p.configureDebugVisualizer(
    p.COV_ENABLE_RENDERING,
    1
)

print("模型加载完成")
print("机器人可动关节数量：", len(joint_sliders))
print("可通过 Base X/Y/Z 调整机器人位置")
print("可通过 Base Roll/Pitch/Yaw 调整机器人姿态")


# =========================================================
# 9. 主循环
# =========================================================

try:
    while True:

        # -------------------------------------------------
        # 读取机器人整体位置
        # -------------------------------------------------

        base_x = p.readUserDebugParameter(
            base_x_slider
        )

        base_y = p.readUserDebugParameter(
            base_y_slider
        )

        base_z = p.readUserDebugParameter(
            base_z_slider
        )

        # -------------------------------------------------
        # 读取机器人整体姿态
        # -------------------------------------------------

        base_roll = p.readUserDebugParameter(
            base_roll_slider
        )

        base_pitch = p.readUserDebugParameter(
            base_pitch_slider
        )

        base_yaw = p.readUserDebugParameter(
            base_yaw_slider
        )

        base_position = [
            base_x,
            base_y,
            base_z
        ]

        base_orientation = p.getQuaternionFromEuler(
            [
                base_roll,
                base_pitch,
                base_yaw
            ]
        )

        # 实时更新机器人整体位姿
        p.resetBasePositionAndOrientation(
            bodyUniqueId=robot_id,
            posObj=base_position,
            ornObj=base_orientation
        )

        # -------------------------------------------------
        # 更新机器人各关节角
        # -------------------------------------------------

        for joint_id, slider_id in joint_sliders:
            target_position = p.readUserDebugParameter(
                slider_id
            )

            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=100
            )

        p.stepSimulation()
        time.sleep(1 / 240)

except KeyboardInterrupt:
    print("\n退出 PyBullet")

finally:
    p.disconnect()