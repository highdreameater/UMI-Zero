"""Unit tests for the Perception Engine (CameraStream, HandTracker, ObjectTracker, OcclusionHandler)."""

from __future__ import annotations

import unittest
import numpy as np

from src.perception.camera_stream import CameraStream
from src.perception.hand_tracker import HandTracker
from src.perception.object_tracker import ObjectTracker
from src.perception.occlusion_handler import OcclusionHandler, KalmanFilter3D


class TestPerception(unittest.TestCase):
    """Test suite for perception components."""

    def setUp(self):
        self.camera_cfg = {
            "camera": {
                "source": "synthetic_test",
                "resolution": [640, 480],
                "fps": 30,
                "intrinsics": {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0},
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        }
        self.camera_stream = CameraStream(self.camera_cfg)
        self.K = self.camera_stream.get_intrinsics()

    def test_camera_stream_intrinsics(self):
        """Verifies camera intrinsic matrix shape and properties."""
        self.assertEqual(self.K.shape, (3, 3))
        self.assertAlmostEqual(self.K[0, 0], 615.0)
        self.assertAlmostEqual(self.K[1, 1], 615.0)
        self.assertAlmostEqual(self.K[0, 2], 320.0)
        self.assertAlmostEqual(self.K[1, 2], 240.0)
        self.assertAlmostEqual(self.K[2, 2], 1.0)

    def test_camera_stream_get_frame(self):
        """Verifies camera frame acquisition."""
        ret, frame, depth = self.camera_stream.get_frame()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_hand_tracker(self):
        """Verifies 3D hand tracking and rotation matrix orthonormality."""
        tracker = HandTracker(confidence=0.5, max_hands=1)
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process(test_img, self.K)

        self.assertIsNotNone(result)
        self.assertIn("wrist_3d", result)
        self.assertIn("thumb_tip_3d", result)
        self.assertIn("index_tip_3d", result)
        self.assertIn("hand_rotation", result)
        self.assertIn("landmarks_2d", result)

        self.assertEqual(result["wrist_3d"].shape, (3,))
        self.assertEqual(result["thumb_tip_3d"].shape, (3,))
        self.assertEqual(result["index_tip_3d"].shape, (3,))
        self.assertEqual(result["landmarks_2d"].shape, (21, 2))

        R = result["hand_rotation"]
        self.assertEqual(R.shape, (3, 3))
        # Check orthogonality R^T * R = I
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)
        # Check determinant is +1 (proper rotation, no reflection)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=4)

    def test_object_tracker(self):
        """Verifies 6-DoF object pose estimation structure."""
        tracker = ObjectTracker(method="aruco_fallback", marker_size=0.05, camera_k=self.K)
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process(test_img)

        self.assertIsNotNone(result)
        self.assertIn("object_pose", result)
        self.assertIn("detected", result)
        self.assertIn("bbox", result)

        pose = result["object_pose"]
        self.assertEqual(pose.shape, (4, 4))
        # Bottom row of SE(3) homogeneous matrix must be [0, 0, 0, 1]
        np.testing.assert_allclose(pose[3, :], [0.0, 0.0, 0.0, 1.0])

    def test_kalman_filter_3d(self):
        """Verifies 3D Kalman Filter prediction and update mechanics."""
        kf = KalmanFilter3D(dt=0.1)
        init_pos = np.array([1.0, 2.0, 3.0])
        kf.reset(init_pos)

        # Measurement at same position
        updated = kf.update(init_pos)
        np.testing.assert_allclose(updated, init_pos, atol=1e-3)

        # Constant velocity prediction without measurement
        pred = kf.predict()
        self.assertEqual(pred.shape, (3,))

    def test_occlusion_handler(self):
        """Verifies missing data imputation and occlusion flagging during dropouts."""
        handler = OcclusionHandler(state_dim=6, window_size=5)

        # Step 1: Initial valid detection
        raw_hand = {
            "wrist_3d": np.array([0.0, 0.1, 0.5]),
            "thumb_tip_3d": np.array([-0.02, 0.15, 0.5]),
            "index_tip_3d": np.array([0.02, 0.15, 0.5]),
            "hand_rotation": np.eye(3),
            "landmarks_2d": np.zeros((21, 2)),
        }
        res1 = handler.update_hand(raw_hand)
        self.assertFalse(res1["occluded"])
        np.testing.assert_allclose(res1["wrist_3d"], raw_hand["wrist_3d"], atol=0.05)

        # Step 2: Feed moving trajectory to establish non-zero velocity
        for step in range(1, 6):
            t_hand = {
                "wrist_3d": np.array([0.0, 0.1 + step * 0.01, 0.5]),
                "thumb_tip_3d": np.array([-0.02, 0.15 + step * 0.01, 0.5]),
                "index_tip_3d": np.array([0.02, 0.15 + step * 0.01, 0.5]),
                "hand_rotation": np.eye(3),
                "landmarks_2d": np.zeros((21, 2)),
            }
            handler.update_hand(t_hand)

        # Step 3: Occlusion occurs (hand_data = None)
        res_occluded = handler.update_hand(None)
        self.assertTrue(res_occluded["occluded"])
        self.assertIsNotNone(res_occluded["wrist_3d"])
        self.assertIsNotNone(res_occluded["thumb_tip_3d"])
        self.assertIsNotNone(res_occluded["index_tip_3d"])
        # Should continue along forward trajectory (+y direction)
        self.assertGreater(res_occluded["wrist_3d"][1], 0.1)


if __name__ == "__main__":
    unittest.main()
