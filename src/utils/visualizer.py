"""Visualization dashboard for bare-hand perception overlays and simulation digital twin."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


# MediaPipe hand skeleton connectivity edges
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


class Visualizer:
    """Renders 2D/3D tracking annotations, pinch vectors, and side-by-side digital twin dashboard."""

    def __init__(self, window_name: str = "Bare-Hand Teleop Pipeline"):
        """Initializes OpenCV visual rendering panel.

        Args:
            window_name: Display window title.
        """
        self.window_name = window_name
        self.window_created = False

    def render(
        self,
        color_frame: np.ndarray,
        hand_data: Optional[Dict[str, Any]],
        object_data: Optional[Dict[str, Any]],
        grasp_state: bool,
        sim_frame: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Draws 2D/3D skeletal joints, 3D object bounding boxes, and real-time system text overlays.

        Args:
            color_frame: Raw camera frame (H, W, 3).
            hand_data: Hand tracker results with landmarks_2d, tips, and occlusion state.
            object_data: Object pose estimator results with bbox and detected state.
            grasp_state: Boolean grasp intent flag.
            sim_frame: Optional rendered frame from PyBullet simulation twin.
            info: Optional metadata (fps, frame_idx, pinch_distance, etc.).

        Returns:
            rendered_frame: Visual image ready for display or video recording.
        """
        if color_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Work on a copy of perception frame
        view = color_frame.copy()
        h, w = view.shape[:2]

        if cv2 is not None:
            # 1. Draw Object Detection / Bounding Box
            if object_data is not None:
                bbox = object_data.get("bbox")
                detected = object_data.get("detected", False)
                occluded = object_data.get("occluded", False)

                if bbox and len(bbox) == 4 and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    box_color = (0, 220, 0) if detected else (0, 165, 255)  # Green or Orange
                    cv2.rectangle(view, (x1, y1), (x2, y2), box_color, 2)

                    label = "Target Object [Tracked]" if detected else "Object [Occlusion Imputed]"
                    cv2.putText(
                        view, label, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA
                    )

            # 2. Draw 2D Hand Skeletal Joints & Connections
            if hand_data is not None:
                landmarks = hand_data.get("landmarks_2d")
                occluded_hand = hand_data.get("occluded", False)
                joint_color = (0, 140, 255) if not occluded_hand else (200, 100, 255)

                if landmarks is not None and len(landmarks) >= 21:
                    pixel_pts = []
                    for lm in landmarks:
                        px = int(np.clip(lm[0] * w, 0, w - 1))
                        py = int(np.clip(lm[1] * h, 0, h - 1))
                        pixel_pts.append((px, py))

                    # Draw bones
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        if start_idx < len(pixel_pts) and end_idx < len(pixel_pts):
                            cv2.line(view, pixel_pts[start_idx], pixel_pts[end_idx], (220, 220, 220), 2, cv2.LINE_AA)

                    # Draw joints
                    for px, py in pixel_pts:
                        cv2.circle(view, (px, py), 4, joint_color, -1, cv2.LINE_AA)

                    # Draw pinch vector between Thumb (4) and Index (8)
                    thumb_pt = pixel_pts[4]
                    index_pt = pixel_pts[8]
                    pinch_color = (0, 0, 255) if grasp_state else (0, 255, 255)  # Red (Grasped) vs Yellow (Open)
                    cv2.line(view, thumb_pt, index_pt, pinch_color, 3, cv2.LINE_AA)
                    cv2.circle(view, thumb_pt, 6, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.circle(view, index_pt, 6, (0, 255, 0), -1, cv2.LINE_AA)

            # 3. Draw HUD and Teleoperation Status Overlay
            self._draw_hud(view, grasp_state, hand_data, object_data, info)

        # 4. Composite side-by-side with PyBullet Simulation Twin if available
        if sim_frame is not None:
            if sim_frame.shape[:2] != (h, w):
                sim_resized = cv2.resize(sim_frame, (w, h)) if cv2 is not None else sim_frame
            else:
                sim_resized = sim_frame

            # Annotate panels with headers
            if cv2 is not None:
                cv2.putText(view, "PERCEPTION: CAMERA STREAM", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(sim_resized, "DIGITAL TWIN: PYBULLET SIMULATION", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 100), 2, cv2.LINE_AA)

            # Horizontal concatenation
            composite = np.hstack([view, sim_resized])
            return composite

        return view

    def _draw_hud(
        self,
        view: np.ndarray,
        grasp_state: bool,
        hand_data: Optional[Dict[str, Any]],
        object_data: Optional[Dict[str, Any]],
        info: Optional[Dict[str, Any]],
    ) -> None:
        """Renders heads-up display overlay on perception frame."""
        h, w = view.shape[:2]

        # Darkened banner at bottom
        banner_h = 75
        cv2.rectangle(view, (0, h - banner_h), (w, h), (20, 20, 20), -1)
        cv2.line(view, (0, h - banner_h), (w, h - banner_h), (80, 80, 80), 1)

        # Grasp intent badge
        badge_text = "GRASP [ACTIVE]" if grasp_state else "GRASP [OPEN]"
        badge_color = (0, 200, 50) if grasp_state else (50, 150, 250)
        cv2.putText(view, badge_text, (15, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_color, 2, cv2.LINE_AA)

        # Metric pinch distance readout
        pinch_dist = info.get("pinch_distance", 0.0) if info else 0.0
        cv2.putText(
            view, f"Pinch Separation: {pinch_dist * 100.0:4.1f} cm", (15, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
        )

        # Teleop tracking stats
        frame_idx = info.get("frame_idx", 0) if info else 0
        fps = info.get("fps", 30.0) if info else 30.0
        hand_status = "Tracking OK" if (hand_data and not hand_data.get("occluded")) else "Imputing (Occluded)"
        obj_status = "Tracking OK" if (object_data and not object_data.get("occluded")) else "Imputing (Occluded)"

        cv2.putText(
            view, f"Frame: {frame_idx:04d} | FPS: {fps:4.1f}", (w - 240, h - 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
        )
        cv2.putText(
            view, f"Hand: {hand_status}", (w - 240, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA
        )
        cv2.putText(
            view, f"Object: {obj_status}", (w - 240, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA
        )

    def show(self, frame: np.ndarray, wait_ms: int = 1) -> int:
        """Displays frame in OpenCV window.

        Args:
            frame: Composite image array.
            wait_ms: Milliseconds to wait for keypress.

        Returns:
            Key code integer pressed by user, or -1.
        """
        if cv2 is None:
            return -1

        try:
            cv2.imshow(self.window_name, frame)
            self.window_created = True
            key = cv2.waitKey(wait_ms) & 0xFF
            return key
        except Exception:
            # Headless environment
            return -1

    def close(self) -> None:
        """Closes OpenCV display windows."""
        if cv2 is not None and self.window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            self.window_created = False
