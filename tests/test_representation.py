"""Unit tests for the Interaction Abstraction Engine (spatial_math, grasp_detector, ict_encoder)."""

from __future__ import annotations

import unittest
import numpy as np

from src.representation.spatial_math import (
    construct_se3_matrix,
    invert_se3,
    compute_relative_pose,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    euler_to_rotation_matrix,
)
from src.representation.grasp_detector import GraspDetector
from src.representation.ict_encoder import ICTEncoder


class TestRepresentation(unittest.TestCase):
    """Test suite for representation and spatial math modules."""

    def test_construct_and_invert_se3(self):
        """Verifies SE(3) matrix construction and exact inverse properties."""
        R = euler_to_rotation_matrix(0.2, -0.4, 0.6)
        t = np.array([0.15, -0.25, 0.85])

        T = construct_se3_matrix(R, t)
        self.assertEqual(T.shape, (4, 4))
        np.testing.assert_allclose(T[:3, :3], R)
        np.testing.assert_allclose(T[:3, 3], t)
        np.testing.assert_allclose(T[3, :], [0.0, 0.0, 0.0, 1.0])

        # Verify T * T^-1 = I
        T_inv = invert_se3(T)
        np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-7)
        np.testing.assert_allclose(T_inv @ T, np.eye(4), atol=1e-7)

    def test_compute_relative_pose(self):
        """Verifies T_rel = (T_cam_obj)^-1 * T_cam_wrist and frame composition."""
        R_obj = euler_to_rotation_matrix(0.1, 0.0, 0.0)
        t_obj = np.array([0.0, 0.0, 0.5])
        T_cam_obj = construct_se3_matrix(R_obj, t_obj)

        R_wrist = euler_to_rotation_matrix(0.1, 0.2, 0.0)
        t_wrist = np.array([0.02, 0.03, 0.52])
        T_cam_wrist = construct_se3_matrix(R_wrist, t_wrist)

        T_rel = compute_relative_pose(T_cam_obj, T_cam_wrist)

        # Verification: T_cam_obj * T_rel must equal T_cam_wrist
        composed = T_cam_obj @ T_rel
        np.testing.assert_allclose(composed, T_cam_wrist, atol=1e-7)

    def test_quaternion_conversions(self):
        """Verifies bidirectional rotation matrix <-> quaternion roundtrip."""
        R_original = euler_to_rotation_matrix(0.5, -0.3, 0.8)
        q = rotation_matrix_to_quaternion(R_original)

        self.assertEqual(len(q), 4)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=6)

        R_reconstructed = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R_reconstructed, R_original, atol=1e-6)

    def test_grasp_detector(self):
        """Verifies binary and continuous grasp evaluation from 3D finger positions."""
        detector = GraspDetector(pinch_threshold=0.035, max_pinch_distance=0.10)

        # Case 1: Closed pinch (< 0.035m) -> Grasped
        thumb_close = np.array([0.0, 0.0, 0.5])
        index_close = np.array([0.02, 0.0, 0.5])  # dist = 0.02m
        is_grasped, dist = detector.evaluate(thumb_close, index_close)
        self.assertTrue(is_grasped)
        self.assertAlmostEqual(dist, 0.02, places=4)

        # Case 2: Open hand (> 0.035m) -> Open
        index_open = np.array([0.06, 0.0, 0.5])  # dist = 0.06m
        is_grasped, dist = detector.evaluate(thumb_close, index_open)
        self.assertFalse(is_grasped)
        self.assertAlmostEqual(dist, 0.06, places=4)

    def test_ict_encoder(self):
        """Verifies complete ICT Token generation and field specifications."""
        config = {
            "representation": {
                "pinch_threshold_meters": 0.035,
                "smooth_window_size": 5,
            }
        }
        encoder = ICTEncoder(config)

        hand_data = {
            "wrist_3d": np.array([0.1, -0.05, 0.55]),
            "thumb_tip_3d": np.array([0.11, -0.05, 0.55]),
            "index_tip_3d": np.array([0.12, -0.05, 0.55]),
            "hand_rotation": np.eye(3),
        }
        object_data = {
            "object_pose": construct_se3_matrix(np.eye(3), np.array([0.0, 0.0, 0.5])),
            "detected": True,
        }

        token = encoder.encode_frame(hand_data, object_data, timestamp=1.25)

        self.assertIn("timestamp", token)
        self.assertIn("T_relative", token)
        self.assertIn("T_object_world", token)
        self.assertIn("is_grasped", token)
        self.assertIn("pinch_distance", token)

        self.assertEqual(token["timestamp"], 1.25)
        self.assertEqual(token["T_relative"].shape, (4, 4))
        self.assertEqual(token["T_object_world"].shape, (4, 4))
        self.assertIsInstance(token["is_grasped"], bool)
        self.assertIsInstance(token["pinch_distance"], float)


if __name__ == "__main__":
    unittest.main()
