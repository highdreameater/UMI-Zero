"""Occlusion handler using 3D linear Kalman filtering and trajectory imputation."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional
import numpy as np


class KalmanFilter3D:
    """3D Constant-Velocity Linear Kalman Filter."""

    def __init__(self, dt: float = 1.0 / 30.0, q_var: float = 0.01, r_var: float = 0.001):
        """Initializes a 6-DoF state [x, y, z, vx, vy, vz]^T Kalman filter."""
        self.dt = dt
        # State: [x, y, z, vx, vy, vz]^T
        self.x = np.zeros(6, dtype=np.float64)
        # Covariance
        self.P = np.eye(6, dtype=np.float64) * 0.1

        # State transition matrix F
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix H
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise covariance Q
        self.Q = np.eye(6, dtype=np.float64) * q_var
        # Measurement noise covariance R
        self.R = np.eye(3, dtype=np.float64) * r_var

        self.initialized = False

    def reset(self, initial_pos: np.ndarray) -> None:
        """Resets filter state to given 3D position."""
        self.x = np.zeros(6, dtype=np.float64)
        self.x[:3] = initial_pos.flatten()
        self.P = np.eye(6, dtype=np.float64) * 0.01
        self.initialized = True

    def predict(self) -> np.ndarray:
        """Advances state using constant-velocity model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Corrects state using observed measurement."""
        if not self.initialized:
            self.reset(measurement)
            return self.x[:3].copy()

        # Innovation (residual)
        y = measurement.flatten() - (self.H @ self.x)
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I_KH = np.eye(6, dtype=np.float64) - (K @ self.H)
        self.P = I_KH @ self.P
        return self.x[:3].copy()


class OcclusionHandler:
    """Handles occlusions and noisy trajectories for 3D hand joints and object 6-DoF pose."""

    def __init__(self, state_dim: int = 6, window_size: int = 5, dt: float = 1.0 / 30.0):
        """Initializes 3D linear Kalman Filters and trajectory imputation buffers.

        Args:
            state_dim: Dimension of Kalman filter state (position + velocity = 6).
            window_size: Window length for smoothing trajectory.
            dt: Discrete time step between frames.
        """
        self.state_dim = state_dim
        self.window_size = max(3, window_size)
        self.dt = dt

        # Kalman filters for hand joints
        self.kf_wrist = KalmanFilter3D(dt=dt)
        self.kf_thumb = KalmanFilter3D(dt=dt)
        self.kf_index = KalmanFilter3D(dt=dt)

        # Kalman filter for object 3D position
        self.kf_object = KalmanFilter3D(dt=dt, q_var=0.005, r_var=0.002)

        # Buffers for Savitzky-Golay / moving window smoothing
        self.wrist_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.object_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

        # Memory for last valid states
        self.last_hand_rot = np.eye(3, dtype=np.float64)
        self.last_hand_landmarks = np.zeros((21, 2), dtype=np.float64)
        self.last_obj_rot = np.eye(3, dtype=np.float64)
        self.last_obj_bbox = [0, 0, 0, 0]

        self.hand_occlusion_count = 0
        self.object_occlusion_count = 0

    def update_hand(self, hand_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Imputes missing 3D hand joint coordinates using motion continuity during occlusions.

        Args:
            hand_data: Hand tracker result or None if occluded/lost.

        Returns:
            Dictionary with filtered/imputed 3D keypoints and occlusion metadata.
        """
        if hand_data is not None and "wrist_3d" in hand_data:
            self.hand_occlusion_count = 0

            # Predict and update with measurement
            self.kf_wrist.predict()
            filtered_wrist = self.kf_wrist.update(hand_data["wrist_3d"])

            self.kf_thumb.predict()
            filtered_thumb = self.kf_thumb.update(hand_data["thumb_tip_3d"])

            self.kf_index.predict()
            filtered_index = self.kf_index.update(hand_data["index_tip_3d"])

            self.last_hand_rot = hand_data.get("hand_rotation", self.last_hand_rot).copy()
            if "landmarks_2d" in hand_data:
                self.last_hand_landmarks = hand_data["landmarks_2d"].copy()

            self.wrist_buffer.append(filtered_wrist)
            smoothed_wrist = self._smooth_buffer(self.wrist_buffer)

            return {
                "wrist_3d": smoothed_wrist,
                "thumb_tip_3d": filtered_thumb,
                "index_tip_3d": filtered_index,
                "hand_rotation": self.last_hand_rot.copy(),
                "landmarks_2d": self.last_hand_landmarks.copy(),
                "occluded": False,
            }
        else:
            # Occlusion occurred: impute state via constant velocity prediction
            self.hand_occlusion_count += 1

            imputed_wrist = self.kf_wrist.predict()
            imputed_thumb = self.kf_thumb.predict()
            imputed_index = self.kf_index.predict()

            self.wrist_buffer.append(imputed_wrist)
            smoothed_wrist = self._smooth_buffer(self.wrist_buffer)

            return {
                "wrist_3d": smoothed_wrist,
                "thumb_tip_3d": imputed_thumb,
                "index_tip_3d": imputed_index,
                "hand_rotation": self.last_hand_rot.copy(),
                "landmarks_2d": self.last_hand_landmarks.copy(),
                "occluded": True,
            }

    def update_object(self, object_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Imputes missing object pose coordinates using constant velocity assumption during occlusions.

        Args:
            object_data: Object tracker output or None.

        Returns:
            Dictionary with filtered/imputed 6-DoF pose and detection flag.
        """
        detected = (
            object_data is not None
            and object_data.get("detected", False)
            and "object_pose" in object_data
        )

        if detected and object_data is not None:
            self.object_occlusion_count = 0
            raw_pose = object_data["object_pose"]
            raw_t = raw_pose[:3, 3]
            self.last_obj_rot = raw_pose[:3, :3].copy()
            self.last_obj_bbox = object_data.get("bbox", self.last_obj_bbox)

            self.kf_object.predict()
            filtered_t = self.kf_object.update(raw_t)

            self.object_buffer.append(filtered_t)
            smoothed_t = self._smooth_buffer(self.object_buffer)

            T_out = np.eye(4, dtype=np.float64)
            T_out[:3, :3] = self.last_obj_rot
            T_out[:3, 3] = smoothed_t

            return {
                "object_pose": T_out,
                "detected": True,
                "bbox": self.last_obj_bbox,
                "occluded": False,
            }
        else:
            # Occlusion occurred (e.g. hand grasps and blocks the marker)
            self.object_occlusion_count += 1
            imputed_t = self.kf_object.predict()

            self.object_buffer.append(imputed_t)
            smoothed_t = self._smooth_buffer(self.object_buffer)

            T_out = np.eye(4, dtype=np.float64)
            T_out[:3, :3] = self.last_obj_rot
            T_out[:3, 3] = smoothed_t

            return {
                "object_pose": T_out,
                "detected": False,
                "bbox": self.last_obj_bbox,
                "occluded": True,
            }

    def _smooth_buffer(self, buffer: deque[np.ndarray]) -> np.ndarray:
        """Applies a Savitzky-Golay / polynomial weighted smoothing filter over buffer."""
        if len(buffer) < 3:
            return buffer[-1].copy()

        arr = np.array(buffer)  # shape (N, 3)
        n = len(buffer)

        # Precomputed Savitzky-Golay / parabolic filter weights for smoothing the endpoint
        if n == 3:
            weights = np.array([0.2, 0.3, 0.5])
        elif n == 4:
            weights = np.array([0.1, 0.2, 0.3, 0.4])
        else:
            # Triangular recency weighting
            weights = np.linspace(0.1, 1.0, n)
            weights = weights / weights.sum()

        smoothed = np.sum(arr * weights[:, None], axis=0)
        return smoothed
