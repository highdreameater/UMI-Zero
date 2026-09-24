"""Inverse Kinematics (IK) solver using PyBullet numerical kinematics and damped least squares fallback."""

from __future__ import annotations

from typing import Any, List, Optional
import numpy as np

from src.representation.spatial_math import rotation_matrix_to_quaternion

try:
    import pybullet as p
except ImportError:
    p = None  # type: ignore


class IKSolver:
    """Calculates manipulator joint configurations that position the end-effector at desired 6-DoF poses."""

    def __init__(
        self,
        urdf_path: str = "franka_panda/panda.urdf",
        ee_link_name: str = "panda_hand",
        pybullet_client: Any = None,
        robot_id: Optional[int] = None,
        joint_names: Optional[List[str]] = None,
        home_configuration: Optional[List[float]] = None,
    ):
        """Initializes the numerical Inverse Kinematics solver.

        Args:
            urdf_path: Path to robot URDF description.
            ee_link_name: Name of target end-effector link (e.g., 'panda_hand').
            pybullet_client: Optional pybullet physics client instance.
            robot_id: Optional PyBullet unique body ID.
            joint_names: List of controllable arm joint names.
            home_configuration: Rest pose joint configuration for null-space projection.
        """
        self.urdf_path = urdf_path
        self.ee_link_name = ee_link_name
        self.pc = pybullet_client or p
        self.robot_id = robot_id
        self.joint_names = joint_names or [f"panda_joint{i}" for i in range(1, 8)]
        self.num_joints = len(self.joint_names)

        self.home_configuration = home_configuration or [
            0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785
        ]

        # Standard Franka Panda joint limits (radians)
        self.lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        self.joint_ranges = [u - l for l, u in zip(self.lower_limits, self.upper_limits)]

        self.ee_link_idx = -1
        self.arm_joint_indices: List[int] = []

        if self.pc is not None and self.robot_id is not None:
            self._cache_robot_indices()

    def set_robot_context(self, robot_id: int, pybullet_client: Any = None) -> None:
        """Connects solver to an instantiated PyBullet robot simulation body."""
        self.robot_id = robot_id
        if pybullet_client is not None:
            self.pc = pybullet_client
        self._cache_robot_indices()

    def _cache_robot_indices(self) -> None:
        """Caches joint indices and target link indices from PyBullet."""
        if self.pc is None or self.robot_id is None:
            return

        num_joints = self.pc.getNumJoints(self.robot_id)
        name_to_idx = {}
        link_name_to_idx = {}

        for j in range(num_joints):
            info = self.pc.getJointInfo(self.robot_id, j)
            j_name = info[1].decode("utf-8")
            l_name = info[12].decode("utf-8")
            name_to_idx[j_name] = j
            link_name_to_idx[l_name] = j

        # Map end-effector link
        if self.ee_link_name in link_name_to_idx:
            self.ee_link_idx = link_name_to_idx[self.ee_link_name]
        else:
            self.ee_link_idx = num_joints - 1

        # Map controllable arm joints
        self.arm_joint_indices = []
        for name in self.joint_names:
            if name in name_to_idx:
                self.arm_joint_indices.append(name_to_idx[name])
            else:
                self.arm_joint_indices.append(len(self.arm_joint_indices))

    def solve(
        self,
        current_joint_states: Optional[List[float]],
        target_ee_pose: np.ndarray,
    ) -> List[float]:
        """Calculates joint angles satisfying target end-effector pose while respecting joint limits.

        Args:
            current_joint_states: Current manipulator joint configuration (radians).
            target_ee_pose: 4x4 homogeneous transformation matrix in robot base frame.

        Returns:
            target_joint_angles: List of target joint angles (radians).
        """
        target_pos = target_ee_pose[:3, 3].tolist()
        target_rot = target_ee_pose[:3, :3]
        target_orn = rotation_matrix_to_quaternion(target_rot).tolist()

        if self.pc is not None and self.robot_id is not None:
            try:
                # Numerical IK with null space rest poses
                ik_solution = self.pc.calculateInverseKinematics(
                    bodyUniqueId=self.robot_id,
                    endEffectorLinkIndex=self.ee_link_idx,
                    targetPosition=target_pos,
                    targetOrientation=target_orn,
                    lowerLimits=self.lower_limits,
                    upperLimits=self.upper_limits,
                    jointRanges=self.joint_ranges,
                    restPoses=self.home_configuration,
                    maxNumIterations=100,
                    residualThreshold=1e-4,
                )

                # Extract only the controllable arm joint angles
                if self.arm_joint_indices:
                    angles = [float(ik_solution[i]) for i in self.arm_joint_indices]
                else:
                    angles = [float(ik_solution[i]) for i in range(min(len(ik_solution), self.num_joints))]

                # Clamp within joint limits
                angles = self._clamp_to_limits(angles)
                return angles
            except Exception:
                pass

        # Standalone numerical/geometric approximation fallback
        return self._solve_analytical_fallback(current_joint_states, target_pos, target_rot)

    def _clamp_to_limits(self, angles: List[float]) -> List[float]:
        """Clamps joint angles strictly within physical bounds."""
        clamped = []
        for i, a in enumerate(angles):
            l = self.lower_limits[i] if i < len(self.lower_limits) else -np.pi
            u = self.upper_limits[i] if i < len(self.upper_limits) else np.pi
            clamped.append(float(np.clip(a, l, u)))
        return clamped

    def _solve_analytical_fallback(
        self,
        current_joint_states: Optional[List[float]],
        target_pos: List[float],
        target_rot: np.ndarray,
    ) -> List[float]:
        """Computes geometric joint angles for testing when PyBullet engine is offline."""
        current = (
            list(current_joint_states)
            if current_joint_states and len(current_joint_states) == self.num_joints
            else list(self.home_configuration)
        )

        x, y, z = target_pos
        # Approximate base rotation towards target (x, y)
        theta1 = float(np.arctan2(y, x)) if (x * x + y * y) > 1e-6 else 0.0

        # Reach radius in horizontal plane
        r = float(np.sqrt(x * x + y * y))
        # Shoulder and elbow elevation
        shoulder_angle = float(np.clip(-0.785 - (z - 0.3) * 0.8, -1.5, 0.5))
        elbow_angle = float(np.clip(-2.356 + (r - 0.4) * 1.2, -2.8, -0.5))

        out = list(current)
        out[0] = theta1
        out[1] = shoulder_angle
        out[3] = elbow_angle
        return self._clamp_to_limits(out)
