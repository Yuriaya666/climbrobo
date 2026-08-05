import pybullet as p
import time
from pathlib import Path

p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

BASE_DIR = Path(__file__).resolve().parent

# 改成你的实际 URDF 路径
robot_urdf = BASE_DIR / "00样机（更换电机）-x12.SLDASM" / "urdf" / "00样机（更换电机）-x12.SLDASM.urdf"

robot_id = p.loadURDF(
    fileName=str(robot_urdf),
    basePosition=[0, 0, 1.0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION
)

# 红色标记球
marker_visual = p.createVisualShape(
    shapeType=p.GEOM_SPHERE,
    radius=0.15,
    rgbaColor=[1, 0, 0, 1]
)

p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=marker_visual,
    basePosition=[0, 0, 1.5]
)

p.addUserDebugText(
    text="ROBOT",
    textPosition=[0, 0, 1.8],
    textColorRGB=[1, 0, 0],
    textSize=2
)

movable_joints = []

for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode()
    joint_type = info[2]
    lower_limit = info[8]
    upper_limit = info[9]

    # 只控制转动关节和移动关节
    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        movable_joints.append(i)

        print(
            "可动关节:",
            i,
            joint_name,
            "limit:",
            lower_limit,
            upper_limit
        )

# 给每个可动关节创建滑块
joint_sliders = []

for joint_id in movable_joints:
    info = p.getJointInfo(robot_id, joint_id)
    joint_name = info[1].decode()
    lower_limit = info[8]
    upper_limit = info[9]

    # 有些 URDF 关节限位可能没写好，这里给个默认范围
    if lower_limit >= upper_limit:
        lower_limit = -3.14
        upper_limit = 3.14

    slider = p.addUserDebugParameter(
        joint_name,
        lower_limit,
        upper_limit,
        0
    )

    joint_sliders.append((joint_id, slider))

# 打印关节信息
print("robot_id:", robot_id)
print("joint num:", p.getNumJoints(robot_id))

for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    print(i, info[1].decode(), "link:", info[12].decode())

# 相机对准机器人
p.resetDebugVisualizerCamera(
    cameraDistance=3,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 1]
)

while True:
    for joint_id, slider in joint_sliders:
        target = p.readUserDebugParameter(slider)

        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=50
        )

    p.stepSimulation()
    time.sleep(1 / 240)