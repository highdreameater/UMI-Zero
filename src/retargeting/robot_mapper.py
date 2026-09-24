"""Robot kinematic mapper translating human actions to robot actuation commands."""

from __future__ import annotations

import numpy as np


class RobotMapper:
    """Maps human grasp state to robot-specific end-effector commands (e.g., 2-finger parallel gripper aperture)."""

    def __init__(self, max_aperture: float = 0.08):
        """Initializes the robot mapper.

        Args:
            max_aperture: Maximum gripper aperture in meters (e.g., 0.08m for Panda).
        """
        self.max_aperture = float(max_aperture)

    def map_gripper_action(self, pinch_distance: float, human_max_pinch: float = 0.10) -> float:
        """Translates human finger separation to target gripper finger joint positions.

        Args:
            pinch_distance: Human thumb-to-index distance in meters.
            human_max_pinch: Normalization factor for maximum human finger spread (meters).

        Returns:
            target_gripper_pos: Command value in meters [0.0, max_aperture].
        """
        normalized = np.clip(pinch_distance / human_max_pinch, 0.0, 1.0)
        return float(normalized * self.max_aperture)

    def map_binary_action(self, is_grasped: bool) -> float:
        """Maps binary grasp intent to gripper aperture (0.0 for closed, max_aperture for open).

        Args:
            is_grasped: Boolean grasp state.

        Returns:
            Target aperture in meters.
        """
        return 0.0 if is_grasped else self.max_aperture

    def get_finger_positions(self, aperture: float) -> tuple[float, float]:
        """Splits total gripper aperture across two symmetric parallel fingers.

        Args:
            aperture: Total distance between gripper fingers in meters.

        Returns:
            Tuple of (left_finger_pos, right_finger_pos).
        """
        half_aperture = float(aperture / 2.0)
        return half_aperture, half_aperture
