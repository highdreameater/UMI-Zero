"""Workspace mapping module for human-to-robot coordinate scaling and spatial clipping."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class WorkspaceMapper:
    """Maps human demonstration motion bounds into target robot reachable workspace."""

    def __init__(
        self,
        scale_factors: List[float],
        robot_base_offset: List[float],
        bounds: Optional[Dict[str, List[float]]] = None,
        align_orientation: bool = True,
    ):
        """Initializes the workspace mapping engine.

        Args:
            scale_factors: 3-element scaling [sx, sy, sz] for human wrist motion.
            robot_base_offset: 3-element offset [ox, oy, oz] in robot base frame (meters).
            bounds: Optional dictionary with 'x', 'y', 'z' [min, max] limits in robot space.
            align_orientation: If True, reorients camera frame convention to robot base convention.
        """
        self.scale = np.array(scale_factors, dtype=np.float64)
        self.offset = np.array(robot_base_offset, dtype=np.float64)
        self.align_orientation = align_orientation

        # Default reachable workspace bounds for Franka Panda / UR5
        self.bounds = bounds or {
            "x": [0.15, 0.85],
            "y": [-0.60, 0.60],
            "z": [0.02, 0.80],
        }

        # Coordinate transformation from camera optical frame (X right, Y down, Z forward)
        # to standard robot base frame (X forward, Y left, Z up)
        self.R_cam_to_robot = np.array(
            [
                [0.0, 0.0, 1.0],   # robot X = camera Z
                [-1.0, 0.0, 0.0],  # robot Y = -camera X
                [0.0, -1.0, 0.0],  # robot Z = -camera Y
            ],
            dtype=np.float64,
        )

    def map_pose(self, T_human_ee: np.ndarray) -> np.ndarray:
        """Scales human wrist translation and applies positional offset for target robot base frame.

        Args:
            T_human_ee: 4x4 transformation matrix of human hand in camera frame.

        Returns:
            T_robot_target: 4x4 target transformation matrix for robot end-effector.
        """
        T_robot_target = np.copy(T_human_ee)

        # Scale human translation and apply positional offset
        t_human = T_human_ee[:3, 3]
        t_scaled = (t_human * self.scale) + self.offset

        # Enforce spatial workspace boundary clipping
        t_clipped = np.copy(t_scaled)
        if "x" in self.bounds:
            t_clipped[0] = np.clip(t_clipped[0], self.bounds["x"][0], self.bounds["x"][1])
        if "y" in self.bounds:
            t_clipped[1] = np.clip(t_clipped[1], self.bounds["y"][0], self.bounds["y"][1])
        if "z" in self.bounds:
            t_clipped[2] = np.clip(t_clipped[2], self.bounds["z"][0], self.bounds["z"][1])

        T_robot_target[:3, 3] = t_clipped

        # Reorient end-effector rotation if requested
        if self.align_orientation:
            R_human = T_human_ee[:3, :3]
            # Standard downward-grasping base orientation with pitch/yaw delta from human
            R_down = np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=np.float64,
            )
            # Combine downward grasp frame with relative wrist motion
            T_robot_target[:3, :3] = R_down @ (self.R_cam_to_robot @ R_human @ self.R_cam_to_robot.T)

        return T_robot_target
