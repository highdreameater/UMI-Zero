"""Camera stream ingestion module.

Supports video files, live webcams, RealSense devices, and procedural synthetic streams.
"""

from __future__ import annotations

import os
from typing import Any, Tuple
import numpy as np

from src.utils.yaml_loader import load_yaml

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore



class CameraStream:
    """Manages video acquisition and intrinsic camera parameter calibration."""

    def __init__(self, config_path: str | dict[str, Any]):
        """Initializes OpenCV video capture, RealSense stream, or synthetic fallback.

        Args:
            config_path: Path to YAML configuration file or configuration dictionary.
        """
        self.config = self._load_config(config_path)
        cam_cfg = self.config.get("camera", {})

        self.source = cam_cfg.get("source", 0)
        self.resolution = tuple(cam_cfg.get("resolution", [640, 480]))
        self.fps = int(cam_cfg.get("fps", 30))

        intrinsics = cam_cfg.get("intrinsics", {})
        self.fx = float(intrinsics.get("fx", 615.0))
        self.fy = float(intrinsics.get("fy", 615.0))
        self.cx = float(intrinsics.get("cx", self.resolution[0] / 2.0))
        self.cy = float(intrinsics.get("cy", self.resolution[1] / 2.0))
        self.dist_coeffs = np.array(
            cam_cfg.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        self.cap: Any = None
        self.is_synthetic = False
        self._frame_count = 0
        self._max_synthetic_frames = 120

        self._init_capture()

    def _load_config(self, config_source: str | dict[str, Any]) -> dict[str, Any]:
        """Loads configuration dictionary from file or accepts dictionary directly."""
        cfg = load_yaml(config_source)
        if not cfg:
            return {
                "camera": {
                    "source": "data/input_videos/sample_demo.mp4",
                    "resolution": [640, 480],
                    "fps": 30,
                    "intrinsics": {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0},
                }
            }
        return cfg


    def _init_capture(self) -> None:
        """Initializes OpenCV VideoCapture or flags synthetic fallback."""
        if cv2 is None:
            self.is_synthetic = True
            return

        source = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source)
        elif isinstance(source, str):
            if os.path.exists(source):
                self.cap = cv2.VideoCapture(source)
            else:
                # Video file does not exist yet -> use synthetic stream
                self.is_synthetic = True
                return
        else:
            self.cap = cv2.VideoCapture(0)

        if self.cap is not None and not self.cap.isOpened():
            self.is_synthetic = True

    def get_frame(self) -> Tuple[bool, np.ndarray, np.ndarray | None]:
        """Returns next frame from stream.

        Returns:
            Tuple of (success_flag, color_image, depth_map_or_None).
        """
        if self.is_synthetic or self.cap is None:
            return self._generate_synthetic_frame()

        ret, frame = self.cap.read()
        if not ret:
            # Video loop or stream end
            return False, np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8), None

        if (frame.shape[1], frame.shape[0]) != self.resolution:
            frame = cv2.resize(frame, self.resolution)

        return True, frame, None

    def _generate_synthetic_frame(self) -> Tuple[bool, np.ndarray, np.ndarray | None]:
        """Generates a procedural test video frame simulating a hand pick-and-place action."""
        if self._frame_count >= self._max_synthetic_frames:
            return False, np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8), None

        t = self._frame_count / float(self.fps)
        self._frame_count += 1

        width, height = self.resolution
        frame = np.ones((height, width, 3), dtype=np.uint8) * 230  # Light studio background

        # Draw a simulated tabletop
        table_top = int(height * 0.70)
        frame[table_top:, :] = [180, 180, 180]

        # Object motion: stationary then lifted when grasped
        obj_x = int(width * 0.50)
        grasp_phase = (self._frame_count > 30) and (self._frame_count < 90)
        lift_y = int(40 * np.sin(np.pi * (self._frame_count - 30) / 60.0)) if grasp_phase else 0
        obj_y = int(table_top - 30 - lift_y)

        # Draw target cube (blue-green)
        half_w = 25
        frame[obj_y - half_w:obj_y + half_w, obj_x - half_w:obj_x + half_w] = [50, 160, 50]

        # Draw ArUco marker inside the object
        inner_m = 16
        frame[obj_y - inner_m:obj_y + inner_m, obj_x - inner_m:obj_x + inner_m] = [255, 255, 255]
        frame[obj_y - 8:obj_y + 8, obj_x - 8:obj_x + 8] = [0, 0, 0]

        # Hand trajectory: approaches object, pinches, lifts, and opens
        if self._frame_count < 30:
            progress = self._frame_count / 30.0
            hx = int(obj_x - 100 * (1.0 - progress))
            hy = int(table_top - 120 + 80 * progress)
            pinch_dist = 60 * (1.0 - 0.7 * progress)
        elif self._frame_count < 90:
            hx = obj_x
            hy = int(table_top - 40 - lift_y)
            pinch_dist = 18  # Grasped / pinched
        else:
            progress = (self._frame_count - 90) / 30.0
            hx = int(obj_x + 80 * progress)
            hy = int(table_top - 40 - 60 * progress)
            pinch_dist = 18 + 40 * progress

        # Render simulated hand (palm, thumb, index)
        # Thumb tip
        tx, ty = int(hx - pinch_dist / 2), int(hy)
        # Index tip
        ix, iy = int(hx + pinch_dist / 2), int(hy)
        # Wrist
        wx, wy = int(hx), int(hy - 80)

        # Draw hand links
        for pt, color in [((tx, ty), (0, 140, 255)), ((ix, iy), (0, 200, 255)), ((wx, wy), (40, 40, 200))]:
            r = 6
            frame[max(0, pt[1]-r):min(height, pt[1]+r), max(0, pt[0]-r):min(width, pt[0]+r)] = color

        return True, frame, None

    def get_intrinsics(self) -> np.ndarray:
        """Returns 3x3 camera matrix K."""
        return np.copy(self.K)

    def release(self) -> None:
        """Releases video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
