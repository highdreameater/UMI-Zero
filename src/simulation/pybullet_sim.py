"""PyBullet physics simulation environment and digital twin renderer."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np

from src.representation.spatial_math import (
    construct_se3_matrix,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    p = None  # type: ignore
    pybullet_data = None  # type: ignore


class PyBulletSim:
    """Manages PyBullet physics simulation, robot manipulation, and digital twin camera rendering."""

    def __init__(self, robot_config: Dict[str, Any], gui: bool = True):
        """Sets up PyBullet simulation context, loads ground plane, table, target object, and robot arm URDF.

        Args:
            robot_config: Robot parameter dictionary or robot configuration root.
            gui: Whether to attempt opening an interactive graphical window (falls back to headless).
        """
        # Extract specific robot configuration if nested
        if "franka_panda" in robot_config:
            self.cfg = robot_config["franka_panda"]
        elif "target_robot" in robot_config:
            target_name = robot_config["target_robot"]
            self.cfg = robot_config.get(target_name, robot_config)
        else:
            self.cfg = robot_config

        self.urdf_path = self.cfg.get("urdf_path", "franka_panda/panda.urdf")
        self.ee_link_name = self.cfg.get("ee_link_name", "panda_hand")
        self.joint_names = self.cfg.get(
            "joint_names", [f"panda_joint{i}" for i in range(1, 8)]
        )
        self.gripper_joint_names = self.cfg.get(
            "gripper_joint_names", ["panda_finger_joint1", "panda_finger_joint2"]
        )
        self.max_gripper_aperture = float(self.cfg.get("max_gripper_aperture", 0.08))
        self.home_configuration = list(
            self.cfg.get("home_configuration", [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        )

        self.client_id: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.table_id: Optional[int] = None
        self.object_id: Optional[int] = None
        self.plane_id: Optional[int] = None

        self.arm_joint_indices: List[int] = []
        self.gripper_joint_indices: List[int] = []
        self.ee_link_idx: int = -1

        self.current_arm_joints: List[float] = list(self.home_configuration)
        self.current_gripper_pos: float = self.max_gripper_aperture

        self.has_pybullet = p is not None
        self._init_simulation(gui)

    def _init_simulation(self, gui: bool) -> None:
        """Connects to PyBullet, loads scene objects and sets joint controllers."""
        if not self.has_pybullet:
            return

        # Attempt GUI first if requested, otherwise headless DIRECT mode
        conn_mode = p.GUI if gui else p.DIRECT
        try:
            self.client_id = p.connect(conn_mode)
        except Exception:
            # Fallback to headless DIRECT
            self.client_id = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setRealTimeSimulation(0, physicsClientId=self.client_id)

        if pybullet_data is not None:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

        # 1. Load ground plane
        try:
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        except Exception:
            pass

        # 2. Load Table
        try:
            self.table_id = p.loadURDF(
                "table/table.urdf",
                basePosition=[0.5, 0.0, -0.65],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self.client_id,
            )
        except Exception:
            pass

        # 3. Load Target Object (Cube 0.05m)
        try:
            col_box = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.025, 0.025, 0.025],
                physicsClientId=self.client_id,
            )
            vis_box = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.025, 0.025, 0.025],
                rgbaColor=[0.2, 0.7, 0.3, 1.0],
                physicsClientId=self.client_id,
            )
            self.object_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=col_box,
                baseVisualShapeIndex=vis_box,
                basePosition=[0.5, 0.0, 0.025],
                physicsClientId=self.client_id,
            )
        except Exception:
            pass

        # 4. Load Robot URDF
        try:
            self.robot_id = p.loadURDF(
                self.urdf_path,
                basePosition=[0.0, 0.0, 0.0],
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
                useFixedBase=True,
                physicsClientId=self.client_id,
            )
        except Exception:
            # Fall back to default panda URDF
            try:
                self.robot_id = p.loadURDF(
                    "franka_panda/panda.urdf",
                    basePosition=[0.0, 0.0, 0.0],
                    useFixedBase=True,
                    physicsClientId=self.client_id,
                )
            except Exception:
                self.robot_id = None

        if self.robot_id is not None:
            self._map_robot_joints()
            self._reset_to_home()

    def _map_robot_joints(self) -> None:
        """Maps joint and link names from URDF to PyBullet indices."""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        name_to_idx = {}
        link_to_idx = {}

        for j in range(num_joints):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            j_name = info[1].decode("utf-8")
            l_name = info[12].decode("utf-8")
            name_to_idx[j_name] = j
            link_to_idx[l_name] = j

        self.arm_joint_indices = [
            name_to_idx[name] for name in self.joint_names if name in name_to_idx
        ]
        self.gripper_joint_indices = [
            name_to_idx[name] for name in self.gripper_joint_names if name in name_to_idx
        ]

        if self.ee_link_name in link_to_idx:
            self.ee_link_idx = link_to_idx[self.ee_link_name]
        else:
            self.ee_link_idx = num_joints - 1

    def _reset_to_home(self) -> None:
        """Sets robot joints to home rest pose immediately."""
        if self.robot_id is None:
            return

        for idx, val in zip(self.arm_joint_indices, self.home_configuration):
            p.resetJointState(self.robot_id, idx, val, physicsClientId=self.client_id)

        for idx in self.gripper_joint_indices:
            p.resetJointState(
                self.robot_id,
                idx,
                self.max_gripper_aperture / 2.0,
                physicsClientId=self.client_id,
            )

    def step(
        self,
        target_arm_joints: List[float],
        target_gripper_pos: float,
        target_object_pose: Optional[np.ndarray] = None,
    ) -> None:
        """Applies position controllers to robot joints and steps physics engine forward.

        Args:
            target_arm_joints: Target joint angles for manipulator arm (radians).
            target_gripper_pos: Target gripper total aperture in meters [0.0, max_aperture].
            target_object_pose: Optional 4x4 pose to sync target virtual object.
        """
        self.current_arm_joints = list(target_arm_joints)
        self.current_gripper_pos = float(target_gripper_pos)

        if not self.has_pybullet or self.robot_id is None or self.client_id is None:
            return

        # Position control for 7-DoF arm
        if self.arm_joint_indices and len(target_arm_joints) >= len(self.arm_joint_indices):
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.arm_joint_indices,
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_arm_joints[:len(self.arm_joint_indices)],
                forces=[200.0] * len(self.arm_joint_indices),
                physicsClientId=self.client_id,
            )

        # Position control for 2-finger parallel gripper
        finger_pos = float(np.clip(target_gripper_pos / 2.0, 0.0, self.max_gripper_aperture / 2.0))
        for g_idx in self.gripper_joint_indices:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=g_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=finger_pos,
                force=50.0,
                physicsClientId=self.client_id,
            )

        # Sync target object pose if provided
        if target_object_pose is not None and self.object_id is not None:
            obj_pos = target_object_pose[:3, 3].tolist()
            obj_rot = target_object_pose[:3, :3]
            obj_orn = rotation_matrix_to_quaternion(obj_rot).tolist()
            p.resetBasePositionAndOrientation(
                self.object_id,
                obj_pos,
                obj_orn,
                physicsClientId=self.client_id,
            )

        # Step physics world forward
        p.stepSimulation(physicsClientId=self.client_id)

    def get_end_effector_pose(self) -> np.ndarray:
        """Returns current simulated end-effector 4x4 transformation matrix in robot frame."""
        if self.has_pybullet and self.robot_id is not None and self.client_id is not None:
            try:
                link_state = p.getLinkState(
                    self.robot_id,
                    self.ee_link_idx,
                    computeForwardKinematics=True,
                    physicsClientId=self.client_id,
                )
                pos = np.array(link_state[4], dtype=np.float64)
                orn_q = np.array(link_state[5], dtype=np.float64)  # [x, y, z, w]
                R = quaternion_to_rotation_matrix(orn_q)
                return construct_se3_matrix(R, pos)
            except Exception:
                pass

        # Fallback FK estimation
        pos = np.array([0.5, 0.0, 0.2], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        return construct_se3_matrix(R, pos)

    def render_camera(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Renders simulated digital twin camera stream as RGB image.

        Args:
            width: Output image width in pixels.
            height: Output image height in pixels.

        Returns:
            RGB image array (height, width, 3) in uint8.
        """
        if self.has_pybullet and self.client_id is not None:
            try:
                cam_target = [0.45, 0.0, 0.15]
                cam_dist = 1.1
                pitch = -30.0
                yaw = 45.0
                roll = 0.0

                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=cam_target,
                    distance=cam_dist,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    upAxisIndex=2,
                    physicsClientId=self.client_id,
                )
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60.0,
                    aspect=float(width) / float(height),
                    nearVal=0.1,
                    farVal=3.0,
                    physicsClientId=self.client_id,
                )

                _, _, rgb, _, _ = p.getCameraImage(
                    width=width,
                    height=height,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    physicsClientId=self.client_id,
                )
                return rgb[:, :, :3].astype(np.uint8)
            except Exception:
                pass

        # Fallback procedural digital twin render if PyBullet renderer unavailable
        return self._render_synthetic_twin(width, height)

    def _render_synthetic_twin(self, width: int, height: int) -> np.ndarray:
        """Synthesizes a digital twin status panel when PyBullet rendering is offline."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = [30, 30, 35]  # Dark sleek background

        # Draw a simulated floor and robot pedestal
        cx = width // 2
        cy = int(height * 0.75)
        img[cy:, :] = [45, 45, 50]

        # Draw pedestal
        img[cy - 80:cy, cx - 25:cx + 25] = [80, 80, 85]

        # Draw schematic 2D kinematic arm
        ee_pose = self.get_end_effector_pose()
        ee_pos = ee_pose[:3, 3]

        ee_u = int(cx + ee_pos[1] * 300)
        ee_v = int(cy - 80 - ee_pos[2] * 250)

        # Draw arm segments
        mid_u = int((cx + ee_u) / 2 - 20)
        mid_v = int((cy - 80 + ee_v) / 2 - 40)

        # Simple coordinate drawings
        img[max(0, mid_v-6):min(height, mid_v+6), max(0, mid_u-6):min(width, mid_u+6)] = [180, 180, 180]
        img[max(0, ee_v-8):min(height, ee_v+8), max(0, ee_u-8):min(width, ee_u+8)] = [0, 200, 255]

        return img

    def close(self) -> None:
        """Terminates simulation session and frees PyBullet resources."""
        if self.has_pybullet and self.client_id is not None:
            try:
                p.disconnect(physicsClientId=self.client_id)
            except Exception:
                pass
            self.client_id = None
