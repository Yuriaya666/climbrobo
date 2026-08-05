import copy
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pybullet as p


# =========================================================
# 1. 文件路径
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# 铁塔实际碰撞模型
tower_collision_path = BASE_DIR / "Tower.STL"

# 铁塔简化显示模型
# 只用于避免原始高精度网格再次造成 OpenGL 显示丢失
tower_visual_path = BASE_DIR / "Tower_visual.stl"

# 原始机器人 URDF
robot_original_path = (
    BASE_DIR
    / "00样机（更换电机）-x12.SLDASM"
    / "urdf"
    / "00样机（更换电机）-x12.SLDASM.urdf"
)

# 自动生成的“碰撞体可视化 URDF”
robot_debug_path = (
    robot_original_path.parent
    / "_robot_collision_visual_debug.urdf"
)

for path, description in [
    (tower_collision_path, "铁塔碰撞模型"),
    (tower_visual_path, "铁塔简化显示模型"),
    (robot_original_path, "机器人 URDF"),
]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到{description}：{path}"
        )


# =========================================================
# 2. 显示设置
# =========================================================

# 默认不要将高精度 Tower.STL 直接作为 visual，
# 否则可能重新出现机器人模型显示不完整的问题。
SHOW_EXACT_TOWER_COLLISION_VISUAL = False

# 铁塔显示颜色
TOWER_COLOR = [1.0, 0.0, 0.0, 0.28]

# 机器人碰撞体显示颜色
ROBOT_COLLISION_COLOR = [0.0, 1.0, 0.15, 0.70]


# =========================================================
# 3. 将 URDF 碰撞几何复制为可视几何
# =========================================================

def local_tag_name(tag: str) -> str:
    """去除可能存在的 XML 命名空间。"""

    if "}" in tag:
        return tag.split("}", 1)[1]

    return tag


def namespace_prefix(tag: str) -> str:
    """取得 XML 标签可能存在的命名空间前缀。"""

    if tag.startswith("{") and "}" in tag:
        return tag[:tag.index("}") + 1]

    return ""


def build_collision_visual_urdf(
    source_path: Path,
    output_path: Path,
) -> int:
    """
    生成调试 URDF：

    1. 删除原来的 visual；
    2. 保留原来的 collision；
    3. 将每个 collision 复制为 visual；
    4. 将复制出的 visual 设置为绿色。

    这样加载后的机器人仍使用原始碰撞体进行碰撞检测，
    显示出来的几何则对应 URDF 的 collision 定义。
    """

    tree = ET.parse(source_path)
    root = tree.getroot()

    collision_count = 0

    for link in root.iter():
        if local_tag_name(link.tag) != "link":
            continue

        children = list(link)

        visual_elements = [
            child
            for child in children
            if local_tag_name(child.tag) == "visual"
        ]

        collision_elements = [
            child
            for child in children
            if local_tag_name(child.tag) == "collision"
        ]

        # 删除原有 visual，避免显示原机器人外观
        for visual in visual_elements:
            link.remove(visual)

        # 将 collision 复制为 visual
        for shape_index, collision in enumerate(
            collision_elements
        ):
            visual = copy.deepcopy(collision)

            namespace = namespace_prefix(collision.tag)
            visual.tag = f"{namespace}visual"

            visual.set(
                "name",
                f"collision_visual_{shape_index}"
            )

            # collision 中通常没有 material；
            # 添加绿色半透明材质
            material = ET.SubElement(
                visual,
                f"{namespace}material",
                {
                    "name": (
                        f"collision_debug_green_"
                        f"{shape_index}"
                    )
                }
            )

            ET.SubElement(
                material,
                f"{namespace}color",
                {
                    "rgba": (
                        f"{ROBOT_COLLISION_COLOR[0]} "
                        f"{ROBOT_COLLISION_COLOR[1]} "
                        f"{ROBOT_COLLISION_COLOR[2]} "
                        f"{ROBOT_COLLISION_COLOR[3]}"
                    )
                }
            )

            link.append(visual)
            collision_count += 1

    if collision_count == 0:
        raise RuntimeError(
            "机器人 URDF 中没有找到任何 <collision> 标签"
        )

    try:
        ET.indent(
            tree,
            space="  "
        )
    except AttributeError:
        # Python 3.8 及更早版本没有 ET.indent
        pass

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True
    )

    return collision_count


collision_count = build_collision_visual_urdf(
    robot_original_path,
    robot_debug_path
)

print("已生成碰撞体调试 URDF：")
print(robot_debug_path)
print("URDF 碰撞几何数量：", collision_count)


# =========================================================
# 4. 启动 PyBullet
# =========================================================

client_id = p.connect(p.GUI)

if client_id < 0:
    raise RuntimeError("PyBullet GUI 启动失败")

p.resetSimulation()

# 当前程序用于检查碰撞几何，不模拟机器人掉落
p.setGravity(0, 0, 0)

p.setPhysicsEngineParameter(
    enableFileCaching=0
)

p.configureDebugVisualizer(
    p.COV_ENABLE_RENDERING,
    0
)

p.configureDebugVisualizer(
    p.COV_ENABLE_SHADOWS,
    0
)


# =========================================================
# 5. 加载铁塔
# =========================================================

# 实际碰撞检测始终使用原始 Tower.STL
tower_collision_shape = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName=str(tower_collision_path),
    flags=p.GEOM_FORCE_CONCAVE_TRIMESH
)

# 默认用简化模型显示，防止高精度 visual 导致机器人消失
if SHOW_EXACT_TOWER_COLLISION_VISUAL:
    tower_display_path = tower_collision_path
else:
    tower_display_path = tower_visual_path

tower_visual_shape = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName=str(tower_display_path),
    rgbaColor=TOWER_COLOR
)

tower_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=tower_collision_shape,
    baseVisualShapeIndex=tower_visual_shape,
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1]
)


# =========================================================
# 6. 加载碰撞体可视化机器人
# =========================================================

initial_robot_position = [0.0, 0.0, 1.0]
initial_robot_euler = [0.0, 0.0, 0.0]

robot_id = p.loadURDF(
    fileName=str(robot_debug_path),
    basePosition=initial_robot_position,
    baseOrientation=p.getQuaternionFromEuler(
        initial_robot_euler
    ),
    useFixedBase=True,
    flags=(
        p.URDF_USE_SELF_COLLISION
        | p.URDF_MAINTAIN_LINK_ORDER
    )
)

print("robot_id：", robot_id)
print("机器人关节数量：", p.getNumJoints(robot_id))


# =========================================================
# 7. 强制设置机器人碰撞体颜色
# =========================================================

# 某些 mesh 材质可能覆盖 URDF 中指定的颜色，
# 因此加载后再次强制修改颜色。
for link_index in range(
    -1,
    p.getNumJoints(robot_id)
):
    try:
        p.changeVisualShape(
            objectUniqueId=robot_id,
            linkIndex=link_index,
            rgbaColor=ROBOT_COLLISION_COLOR
        )

    except p.error:
        pass


# =========================================================
# 8. 打印各 link 的碰撞体信息
# =========================================================

print("\n========== 机器人碰撞体 ==========")

total_shapes = 0

for link_index in range(
    -1,
    p.getNumJoints(robot_id)
):
    if link_index == -1:
        link_name = "base_link"
    else:
        joint_info = p.getJointInfo(
            robot_id,
            link_index
        )

        link_name = joint_info[12].decode(
            "utf-8",
            errors="ignore"
        )

    shape_data = p.getCollisionShapeData(
        robot_id,
        link_index
    )

    shape_number = len(shape_data)
    total_shapes += shape_number

    print(
        f"link {link_index:2d} | "
        f"{link_name:30s} | "
        f"碰撞体数量：{shape_number}"
    )

print("PyBullet读取到的碰撞体总数：", total_shapes)


# =========================================================
# 9. 创建机器人整体位姿滑块
# =========================================================

base_x_slider = p.addUserDebugParameter(
    "Base X",
    -5.0,
    5.0,
    initial_robot_position[0]
)

base_y_slider = p.addUserDebugParameter(
    "Base Y",
    -5.0,
    5.0,
    initial_robot_position[1]
)

base_z_slider = p.addUserDebugParameter(
    "Base Z",
    0.0,
    10.0,
    initial_robot_position[2]
)

base_roll_slider = p.addUserDebugParameter(
    "Base Roll",
    -math.pi,
    math.pi,
    initial_robot_euler[0]
)

base_pitch_slider = p.addUserDebugParameter(
    "Base Pitch",
    -math.pi,
    math.pi,
    initial_robot_euler[1]
)

base_yaw_slider = p.addUserDebugParameter(
    "Base Yaw",
    -math.pi,
    math.pi,
    initial_robot_euler[2]
)


# =========================================================
# 10. 创建关节滑块
# =========================================================

joint_sliders = []

for joint_index in range(
    p.getNumJoints(robot_id)
):
    joint_info = p.getJointInfo(
        robot_id,
        joint_index
    )

    joint_name = joint_info[1].decode(
        "utf-8",
        errors="ignore"
    )

    joint_type = joint_info[2]
    lower_limit = joint_info[8]
    upper_limit = joint_info[9]

    if joint_type not in (
        p.JOINT_REVOLUTE,
        p.JOINT_PRISMATIC
    ):
        continue

    if lower_limit >= upper_limit:
        if joint_type == p.JOINT_REVOLUTE:
            lower_limit = -math.pi
            upper_limit = math.pi
        else:
            lower_limit = -1.0
            upper_limit = 1.0

    current_position = p.getJointState(
        robot_id,
        joint_index
    )[0]

    slider_id = p.addUserDebugParameter(
        paramName=(
            f"Joint {joint_index}: "
            f"{joint_name}"
        ),
        rangeMin=lower_limit,
        rangeMax=upper_limit,
        startValue=current_position
    )

    joint_sliders.append(
        (joint_index, slider_id)
    )


# =========================================================
# 11. 设置相机并恢复渲染
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

print("\n显示说明：")
print("绿色：机器人 URDF 中定义的碰撞几何")
print("红色：铁塔碰撞区域的显示参考")
print("铁塔实际碰撞检测仍使用原始 Tower.STL")

if SHOW_EXACT_TOWER_COLLISION_VISUAL:
    print("当前显示：原始高精度铁塔碰撞网格")
else:
    print("当前显示：简化铁塔代理，避免 OpenGL 丢失机器人")


# =========================================================
# 12. 主循环
# =========================================================

try:
    while True:

        # -------------------------------------------------
        # 更新机器人整体位姿
        # -------------------------------------------------

        base_position = [
            p.readUserDebugParameter(
                base_x_slider
            ),
            p.readUserDebugParameter(
                base_y_slider
            ),
            p.readUserDebugParameter(
                base_z_slider
            ),
        ]

        base_euler = [
            p.readUserDebugParameter(
                base_roll_slider
            ),
            p.readUserDebugParameter(
                base_pitch_slider
            ),
            p.readUserDebugParameter(
                base_yaw_slider
            ),
        ]

        base_orientation = p.getQuaternionFromEuler(
            base_euler
        )

        p.resetBasePositionAndOrientation(
            bodyUniqueId=robot_id,
            posObj=base_position,
            ornObj=base_orientation
        )

        # -------------------------------------------------
        # 更新机器人关节构型
        # -------------------------------------------------

        for joint_index, slider_id in joint_sliders:
            target_position = p.readUserDebugParameter(
                slider_id
            )

            p.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                targetValue=target_position,
                targetVelocity=0.0
            )

        # 刷新碰撞检测
        p.performCollisionDetection()

        time.sleep(1 / 120)

except KeyboardInterrupt:
    print("\n退出 PyBullet")

finally:
    p.disconnect()

    # 删除自动生成的临时 URDF
    try:
        if robot_debug_path.exists():
            robot_debug_path.unlink()
    except OSError:
        pass