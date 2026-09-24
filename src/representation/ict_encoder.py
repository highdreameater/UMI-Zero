"""Interaction-Centric Token (ICT) encoder module."""

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from src.representation.grasp_detector import GraspDetector
from src.representation.spatial_math import compute_relative_pose, construct_se3_matrix


class ICTEncoder:
    """Encodes per-frame vision tracking results into embodiment-agnostic Interaction Tokens."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes ICT encoder with representation parameters.

        Args:
            config: Configuration dictionary (can be full pipeline config or representation sub-dict).
        """
        rep_cfg = config.get("representation", config)
        pinch_thresh = float(rep_cfg.get("pinch_threshold_meters", 0.035))
        self.smooth_window_size = int(rep_cfg.get("smooth_window_size", 5))

        self.grasp_detector = GraspDetector(pinch_threshold=pinch_thresh)

    def encode_frame(
        self,
        hand_data: Dict[str, Any],
        object_data: Dict[str, Any],
        timestamp: float,
    ) -> Dict[str, Any]:
        """Transforms raw tracking data into an Interaction-Centric Token (ICT).

        Args:
            hand_data: Hand perception dictionary with wrist_3d, tips, and rotation.
            object_data: Object perception dictionary with object_pose (T_cam_obj).
            timestamp: Frame timestamp in seconds.

        Returns:
            Dictionary (ICT Structure):
                'timestamp': float
                'T_relative': np.ndarray (4, 4) - SE(3) pose of hand relative to object
                'T_object_world': np.ndarray (4, 4) - SE(3) pose of object in camera frame
                'is_grasped': bool - Grasp activation trigger
                'pinch_distance': float - Metric separation between fingertips
        """
        # 1. Extract 4x4 hand wrist pose in camera frame
        wrist_pos = hand_data.get("wrist_3d", np.array([0.0, 0.0, 0.5], dtype=np.float64))
        hand_rot = hand_data.get("hand_rotation", np.eye(3, dtype=np.float64))
        T_camera_wrist = construct_se3_matrix(hand_rot, wrist_pos)

        # 2. Extract 4x4 object pose in camera frame
        T_camera_object = object_data.get("object_pose", np.eye(4, dtype=np.float64))

        # 3. Compute embodiment-agnostic relative SE(3) transformation: T_rel = T_obj^-1 * T_wrist
        T_relative = compute_relative_pose(T_camera_object, T_camera_wrist)

        # 4. Evaluate grasp intent
        thumb_tip = hand_data.get("thumb_tip_3d", wrist_pos)
        index_tip = hand_data.get("index_tip_3d", wrist_pos)
        is_grasped, pinch_distance = self.grasp_detector.evaluate(thumb_tip, index_tip)

        return {
            "timestamp": float(timestamp),
            "T_relative": T_relative,
            "T_object_world": T_camera_object,
            "is_grasped": is_grasped,
            "pinch_distance": pinch_distance,
            "T_wrist_world": T_camera_wrist,
        }
