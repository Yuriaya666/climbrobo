from __future__ import annotations

import time

import numpy as np
import pybullet as p

from environment.candidates import CandidateSet
from environment.one_step_planner import AttachmentPoseBuilder
from environment.paths import ProjectPaths
from environment.suction_frames import SuctionFrameSet
from environment.transforms import RigidTransform
from environment.urdf_resolver import ResolvedUrdf


def main() -> None:
    """最小GUI检查：只确认机器人是否能被看见。"""

    paths = ProjectPaths.from_repo_root()
    frames = SuctionFrameSet.load(paths.suction_config)
    foot1 = CandidateSet.load_npz(paths.candidate_npz("foot1"), "foot1")

    support = foot1.select_from_bottom(123)
    pose_builder = AttachmentPoseBuilder()
    support_pose = pose_builder.build(
        support,
        preferred_y_reference_world=np.array([0.0, 0.0, 1.0]),
    )
    base_pose = support_pose.multiply(
        frames.base_end.transform_link_to_suction.inverse()
    )

    p.connect(p.GUI)
    p.resetSimulation()
    p.setGravity(0.0, 0.0, 0.0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    # 加一个透明塔作为位置参照；如果还看不见机器人，可先把这段注释掉。
    tower_visual = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=str(paths.tower_visual_mesh),
        rgbaColor=[0.75, 0.75, 0.75, 0.20],
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=tower_visual,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )

    base_position, base_orientation = base_pose.as_pybullet()
    with ResolvedUrdf(paths.robot_urdf, paths.robot_mesh_dir) as urdf_path:
        robot_id = p.loadURDF(
            fileName=str(urdf_path),
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER,
        )

    # 把机器人整体强制改成亮红色，方便在塔旁边找。
    for link_index in range(-1, p.getNumJoints(robot_id)):
        p.changeVisualShape(
            objectUniqueId=robot_id,
            linkIndex=link_index,
            rgbaColor=[1.0, 0.05, 0.02, 1.0],
        )

    aabb_min, aabb_max = p.getAABB(robot_id)
    robot_center = 0.5 * (np.asarray(aabb_min) + np.asarray(aabb_max))

    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45.0,
        cameraPitch=-25.0,
        cameraTargetPosition=robot_center.tolist(),
    )

    print("robot_id:", robot_id)
    print("robot_aabb:", aabb_min, aabb_max)
    print("camera_target:", robot_center.tolist())
    print("红色模型就是机器人。按 Ctrl+C 退出。")

    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
