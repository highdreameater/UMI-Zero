"""Grasp detection module calculating 3D finger separation and grasp intent."""

from __future__ import annotations

import numpy as np


class GraspDetector:
    """Detects grasp intent from 3D hand keypoints."""

    def __init__(self, pinch_threshold: float = 0.035, max_pinch_distance: float = 0.10):
        """Initializes grasp detector.

        Args:
            pinch_threshold: Pinch distance in meters below which grasp is triggered (e.g. 0.035m).
            max_pinch_distance: Maximum expected human pinch separation for normalization (meters).
        """
        self.pinch_threshold = float(pinch_threshold)
        self.max_pinch_distance = float(max_pinch_distance)

    def evaluate(self, thumb_tip_3d: np.ndarray, index_tip_3d: np.ndarray) -> tuple[bool, float]:
        """Calculates Euclidean pinch distance and determines binary/continuous grasp activation state.

        Args:
            thumb_tip_3d: 3D coordinates of thumb tip [x, y, z] in meters.
            index_tip_3d: 3D coordinates of index fingertip [x, y, z] in meters.

        Returns:
            Tuple of:
                is_grasped: bool - True if pinch distance < pinch_threshold
                pinch_distance: float - Metric Euclidean distance between fingertips (meters)
        """
        p_thumb = np.asarray(thumb_tip_3d, dtype=np.float64)
        p_index = np.asarray(index_tip_3d, dtype=np.float64)

        dist = float(np.linalg.norm(p_thumb - p_index))
        is_grasped = bool(dist < self.pinch_threshold)

        return is_grasped, dist

    def get_normalized_aperture(self, pinch_distance: float) -> float:
        """Maps Euclidean pinch distance to a normalized aperture scalar [0.0, 1.0].

        Args:
            pinch_distance: Metric pinch separation in meters.

        Returns:
            Normalized aperture scalar where 0.0 is closed pinch and 1.0 is wide open.
        """
        return float(np.clip(pinch_distance / self.max_pinch_distance, 0.0, 1.0))
