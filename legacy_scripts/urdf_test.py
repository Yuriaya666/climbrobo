from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pybullet as p


# ============================================================
# 1. 用户参数
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

URDF_PATH = (
    BASE_DIR
    / "00样机（更换电机）-x12.SLDASM"
    / "urdf"
    / "00样机（更换电机）-x12.SLDASM.urdf"
)



# 吸盘坐标系相对于对应link坐标系的位置和法向
BASE_CONTACT_LOCAL_POSITION = np.array(
    [0.0, 0.0, 0.0],
    dtype=float,
)

BASE_CONTACT_LOCAL_NORMAL = np.array(
    [1.0, 0.0, -1.0],
    dtype=float,
)

L8_CONTACT_LOCAL_POSITION = np.array(
    [-0.11768, 0.0, -0.10853],
    dtype=float,
)

L8_CONTACT_LOCAL_NORMAL = np.array(
    [-1.0, 0.0, -1.0],
    dtype=float,
)

# 统一使用各link局部坐标系的+Y作为吸盘坐标系Y轴参考方向
CONTACT_Y_REFERENCE = np.array(
    [0.0, 1.0, 0.0],
    dtype=float,
)

AXIS_LENGTH = 0.08
LINE_WIDTH = 4.0

ENABLE_JOINT_SLIDERS = True


# ============================================================
# 2. 基础数学工具
# ============================================================

def normalize(vector: np.ndarray) -> np.ndarray:
    """向量单位化。"""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        raise ValueError("不能单位化零向量。")

    return vector / norm


def rotation_matrix_to_quaternion(
    rotation_matrix: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    将3×3旋转矩阵转换成PyBullet使用的四元数[x, y, z, w]。
    """
    m = np.asarray(rotation_matrix, dtype=float)

    if m.shape != (3, 3):
        raise ValueError("rotation_matrix必须是3×3矩阵。")

    trace = np.trace(m)

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s

    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(
            1.0 + m[0, 0] - m[1, 1] - m[2, 2]
        ) * 2.0

        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s

    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(
            1.0 + m[1, 1] - m[0, 0] - m[2, 2]
        ) * 2.0

        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s

    else:
        s = math.sqrt(
            1.0 + m[2, 2] - m[0, 0] - m[1, 1]
        ) * 2.0

        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quaternion = np.array(
        [qx, qy, qz, qw],
        dtype=float,
    )

    quaternion /= np.linalg.norm(quaternion)

    return tuple(quaternion.tolist())


def build_contact_frame_orientation(
    normal: np.ndarray,
    y_reference: np.ndarray,
) -> tuple[
    tuple[float, float, float, float],
    np.ndarray,
]:
    """
    根据吸盘法向和Y轴参考方向建立吸盘局部坐标系。

    坐标系约定：
        Z轴 = 吸盘法向
        Y轴 = 尽可能接近link局部+Y
        X轴 = Y × Z

    返回：
        quaternion：接触坐标系相对于link坐标系的姿态
        rotation_matrix：三列依次为X、Y、Z轴
    """
    z_axis = normalize(normal)
    y_reference = normalize(y_reference)

    # 防止Y参考方向和Z轴平行
    if abs(np.dot(y_reference, z_axis)) > 0.99:
        raise ValueError(
            "Y轴参考方向与吸盘法向接近平行，无法稳定构造坐标系。"
        )

    # 右手坐标系
    x_axis = normalize(np.cross(y_reference, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))

    rotation_matrix = np.column_stack(
        [
            x_axis,
            y_axis,
            z_axis,
        ]
    )

    quaternion = rotation_matrix_to_quaternion(
        rotation_matrix
    )

    return quaternion, rotation_matrix


# ============================================================
# 3. PyBullet link和位姿工具
# ============================================================

def build_link_index_map(
    body_id: int,
) -> dict[str, int]:
    """建立link名称到PyBullet link index的映射。"""
    link_map = {
        "base_link": -1,
    }

    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(
            body_id,
            joint_index,
        )

        child_link_name = joint_info[12].decode("utf-8")
        link_map[child_link_name] = joint_index

    return link_map


def get_link_frame_world_pose(
    body_id: int,
    link_index: int,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float, float],
]:
    """
    获取URDF link坐标系在世界坐标系中的位姿。

    link_index=-1时表示base_link。
    """
    if link_index == -1:
        return p.getBasePositionAndOrientation(body_id)

    link_state = p.getLinkState(
        body_id,
        link_index,
        computeForwardKinematics=True,
    )

    # [4]和[5]对应URDF link frame的世界位姿
    return link_state[4], link_state[5]


def get_contact_world_pose(
    body_id: int,
    link_index: int,
    contact_local_position,
    contact_local_orientation,
):
    """
    将接触坐标系从link局部坐标转换到世界坐标。
    """
    link_world_position, link_world_orientation = (
        get_link_frame_world_pose(
            body_id,
            link_index,
        )
    )

    contact_world_position, contact_world_orientation = (
        p.multiplyTransforms(
            link_world_position,
            link_world_orientation,
            contact_local_position,
            contact_local_orientation,
        )
    )

    return contact_world_position, contact_world_orientation


# ============================================================
# 4. 调试坐标系绘制
# ============================================================

def quaternion_axes(
    quaternion,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回四元数对应坐标系的世界X、Y、Z单位向量。
    """
    matrix = p.getMatrixFromQuaternion(quaternion)

    rotation_matrix = np.array(
        matrix,
        dtype=float,
    ).reshape(3, 3)

    # 旋转矩阵三列分别是局部X、Y、Z轴在世界坐标中的方向
    x_axis = rotation_matrix[:, 0]
    y_axis = rotation_matrix[:, 1]
    z_axis = rotation_matrix[:, 2]

    return x_axis, y_axis, z_axis


def update_debug_line(
    start_position,
    end_position,
    color,
    old_line_id: int | None,
) -> int:
    """创建或更新一条调试线。"""
    kwargs = {}

    if old_line_id is not None:
        kwargs["replaceItemUniqueId"] = old_line_id

    return p.addUserDebugLine(
        lineFromXYZ=start_position,
        lineToXYZ=end_position,
        lineColorRGB=color,
        lineWidth=LINE_WIDTH,
        lifeTime=0,
        **kwargs,
    )


def update_debug_text(
    text: str,
    position,
    old_text_id: int | None,
) -> int:
    """创建或更新调试文字。"""
    kwargs = {}

    if old_text_id is not None:
        kwargs["replaceItemUniqueId"] = old_text_id

    return p.addUserDebugText(
        text=text,
        textPosition=position,
        textColorRGB=[1.0, 1.0, 1.0],
        textSize=1.3,
        lifeTime=0,
        **kwargs,
    )


def update_coordinate_frame(
    position,
    orientation,
    label: str,
    handles: dict[str, int | None],
) -> dict[str, int | None]:
    """
    在世界坐标系中画一个动态更新的坐标系。

    红色：X轴
    绿色：Y轴
    蓝色：Z轴，即吸盘法向
    """
    origin = np.asarray(position, dtype=float)

    x_axis, y_axis, z_axis = quaternion_axes(
        orientation
    )

    x_end = origin + AXIS_LENGTH * x_axis
    y_end = origin + AXIS_LENGTH * y_axis
    z_end = origin + AXIS_LENGTH * z_axis

    handles["x"] = update_debug_line(
        origin.tolist(),
        x_end.tolist(),
        [1.0, 0.0, 0.0],
        handles.get("x"),
    )

    handles["y"] = update_debug_line(
        origin.tolist(),
        y_end.tolist(),
        [0.0, 1.0, 0.0],
        handles.get("y"),
    )

    handles["z"] = update_debug_line(
        origin.tolist(),
        z_end.tolist(),
        [0.0, 0.0, 1.0],
        handles.get("z"),
    )

    text_position = origin + np.array(
        [0.0, 0.0, AXIS_LENGTH * 0.35]
    )

    handles["text"] = update_debug_text(
        label,
        text_position.tolist(),
        handles.get("text"),
    )

    return handles


# ============================================================
# 5. 可选：关节滑块
# ============================================================

def create_joint_sliders(
    body_id: int,
) -> dict[int, int]:
    slider_map = {}

    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(
            body_id,
            joint_index,
        )

        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]

        if joint_type not in (
            p.JOINT_REVOLUTE,
            p.JOINT_PRISMATIC,
        ):
            continue

        lower_limit = joint_info[8]
        upper_limit = joint_info[9]

        if lower_limit > upper_limit:
            lower_limit = -math.pi
            upper_limit = math.pi

        slider_id = p.addUserDebugParameter(
            paramName=joint_name,
            rangeMin=lower_limit,
            rangeMax=upper_limit,
            startValue=0.0,
        )

        slider_map[joint_index] = slider_id

    return slider_map


def update_joints_from_sliders(
    body_id: int,
    slider_map: dict[int, int],
) -> None:
    for joint_index, slider_id in slider_map.items():
        joint_position = p.readUserDebugParameter(
            slider_id
        )

        # 验证坐标系时采用运动学设置
        p.resetJointState(
            body_id,
            joint_index,
            joint_position,
        )


# ============================================================
# 6. 主程序
# ============================================================

def main() -> None:
    if not Path(URDF_PATH).exists():
        raise FileNotFoundError(
            f"找不到URDF文件：{URDF_PATH}"
        )

    physics_client = p.connect(p.GUI)

    if physics_client < 0:
        raise RuntimeError("无法启动PyBullet GUI。")

    p.resetSimulation()
    p.setGravity(0.0, 0.0, 0.0)
    p.setRealTimeSimulation(0)

    # 验证坐标系阶段使用固定根节点，避免机器人整体漂移。
    # 后续爬行仿真再改为useFixedBase=False。
    robot_id = p.loadURDF(
        str(URDF_PATH),
        basePosition=[0.0, 0.0, 0.5],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    p.resetDebugVisualizerCamera(
        cameraDistance=1.4,
        cameraYaw=45.0,
        cameraPitch=-25.0,
        cameraTargetPosition=[0.1, 0.0, 0.5],
    )

    link_map = build_link_index_map(robot_id)

    print("\nPyBullet link映射：")
    for link_name, link_index in link_map.items():
        print(f"{link_name:>12s} : {link_index}")

    if "L8" not in link_map:
        raise RuntimeError(
            "URDF中没有找到L8，请检查link名称。"
        )

    base_link_index = -1
    l8_link_index = link_map["L8"]

    # 根据圆心、法向和+Y参考方向建立两个吸盘坐标系
    base_contact_local_orientation, base_rotation = (
        build_contact_frame_orientation(
            BASE_CONTACT_LOCAL_NORMAL,
            CONTACT_Y_REFERENCE,
        )
    )

    l8_contact_local_orientation, l8_rotation = (
        build_contact_frame_orientation(
            L8_CONTACT_LOCAL_NORMAL,
            CONTACT_Y_REFERENCE,
        )
    )

    print("\nBase吸盘局部坐标系：")
    print("位置：", BASE_CONTACT_LOCAL_POSITION)
    print("X轴：", base_rotation[:, 0])
    print("Y轴：", base_rotation[:, 1])
    print("Z轴：", base_rotation[:, 2])
    print("四元数[x,y,z,w]：", base_contact_local_orientation)

    print("\nL8吸盘局部坐标系：")
    print("位置：", L8_CONTACT_LOCAL_POSITION)
    print("X轴：", l8_rotation[:, 0])
    print("Y轴：", l8_rotation[:, 1])
    print("Z轴：", l8_rotation[:, 2])
    print("四元数[x,y,z,w]：", l8_contact_local_orientation)

    slider_map = {}

    if ENABLE_JOINT_SLIDERS:
        slider_map = create_joint_sliders(robot_id)

    base_frame_handles = {
        "x": None,
        "y": None,
        "z": None,
        "text": None,
    }

    l8_frame_handles = {
        "x": None,
        "y": None,
        "z": None,
        "text": None,
    }

    try:
        while p.isConnected():
            if ENABLE_JOINT_SLIDERS:
                update_joints_from_sliders(
                    robot_id,
                    slider_map,
                )

            p.stepSimulation()

            # Base端吸盘坐标系世界位姿
            base_contact_world_position, base_contact_world_orientation = (
                get_contact_world_pose(
                    robot_id,
                    base_link_index,
                    BASE_CONTACT_LOCAL_POSITION.tolist(),
                    base_contact_local_orientation,
                )
            )

            # L8端吸盘坐标系世界位姿
            l8_contact_world_position, l8_contact_world_orientation = (
                get_contact_world_pose(
                    robot_id,
                    l8_link_index,
                    L8_CONTACT_LOCAL_POSITION.tolist(),
                    l8_contact_local_orientation,
                )
            )

            base_frame_handles = update_coordinate_frame(
                base_contact_world_position,
                base_contact_world_orientation,
                "Base suction",
                base_frame_handles,
            )

            l8_frame_handles = update_coordinate_frame(
                l8_contact_world_position,
                l8_contact_world_orientation,
                "L8 suction",
                l8_frame_handles,
            )

            time.sleep(1.0 / 240.0)

    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()