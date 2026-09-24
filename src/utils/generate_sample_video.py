"""Procedural sample demonstration video generator.

Creates sample_demo.mp4 simulating a human hand pick-and-place manipulation with ArUco marker.
"""

from __future__ import annotations

import os
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def generate_sample_video(
    output_path: str = "data/input_videos/sample_demo.mp4",
    num_frames: int = 150,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> bool:
    """Generates synthetic pick-and-place video for demonstration and testing.

    Args:
        output_path: Output .mp4 video path.
        num_frames: Total number of frames to render.
        fps: Video framerate.
        width: Frame width.
        height: Frame height.

    Returns:
        True if video was generated successfully.
    """
    if cv2 is None:
        return False

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        return False

    # Marker dictionary
    aruco_dict = None
    if hasattr(cv2, "aruco"):
        dict_attr = getattr(cv2.aruco, "DICT_4X4_50", None)
        if dict_attr is not None:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_attr)

    table_top = int(height * 0.70)
    obj_init_x = int(width * 0.50)
    obj_init_y = int(table_top - 30)

    for i in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 235

        # Table surface
        cv2.rectangle(frame, (0, table_top), (width, height), (190, 190, 190), -1)
        cv2.line(frame, (0, table_top), (width, table_top), (140, 140, 140), 2)

        # Hand trajectory timeline
        # Phase 0 (0-35): Hand descends towards object
        # Phase 1 (35-45): Grasp closes
        # Phase 2 (45-105): Lift and transport
        # Phase 3 (105-120): Grasp opens
        # Phase 4 (120-150): Hand retracts
        if i < 35:
            p = i / 35.0
            hx = int(obj_init_x - 120 * (1.0 - p))
            hy = int(table_top - 140 + 100 * p)
            pinch_px = int(60 * (1.0 - 0.7 * p))
            grasped = False
            lift_y = 0
            obj_x = obj_init_x
        elif i < 45:
            p = (i - 35) / 10.0
            hx = obj_init_x
            hy = int(table_top - 40)
            pinch_px = int(18 * (1.0 - p) + 12 * p)
            grasped = True
            lift_y = 0
            obj_x = obj_init_x
        elif i < 105:
            p = (i - 45) / 60.0
            lift_y = int(60 * np.sin(np.pi * p))
            obj_x = int(obj_init_x + 50 * np.sin(2 * np.pi * p))
            hx = obj_x
            hy = int(table_top - 40 - lift_y)
            pinch_px = 12
            grasped = True
        elif i < 120:
            p = (i - 105) / 15.0
            lift_y = 0
            obj_x = obj_init_x
            hx = obj_x
            hy = int(table_top - 40)
            pinch_px = int(12 + 48 * p)
            grasped = False
        else:
            p = (i - 120) / 30.0
            lift_y = 0
            obj_x = obj_init_x
            hx = int(obj_init_x + 100 * p)
            hy = int(table_top - 40 - 100 * p)
            pinch_px = 60
            grasped = False

        obj_y = int(table_top - 30 - lift_y)

        # 1. Draw Target Cube (Green)
        half_w = 25
        cv2.rectangle(
            frame,
            (obj_x - half_w, obj_y - half_w),
            (obj_x + half_w, obj_y + half_w),
            (50, 160, 50),
            -1,
        )

        # Draw ArUco marker inside cube
        if aruco_dict is not None and hasattr(cv2.aruco, "generateImageMarker"):
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 32)
            marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            frame[obj_y - 16:obj_y + 16, obj_x - 16:obj_x + 16] = marker_bgr
        else:
            # Fallback marker pattern
            cv2.rectangle(frame, (obj_x - 16, obj_y - 16), (obj_x + 16, obj_y + 16), (255, 255, 255), -1)
            cv2.rectangle(frame, (obj_x - 8, obj_y - 8), (obj_x + 8, obj_y + 8), (0, 0, 0), -1)

        # 2. Draw Hand
        # Wrist
        wx, wy = int(hx), int(hy - 80)
        # Thumb tip
        tx, ty = int(hx - pinch_px / 2), int(hy)
        # Index tip
        ix, iy = int(hx + pinch_px / 2), int(hy)
        # Palm center
        px, py = int(hx), int(hy - 35)

        # Draw hand bones
        cv2.line(frame, (wx, wy), (px, py), (210, 180, 140), 10)
        cv2.line(frame, (px, py), (tx, ty), (210, 180, 140), 6)
        cv2.line(frame, (px, py), (ix, iy), (210, 180, 140), 6)

        # Distinct colored keypoint markers for tracking
        cv2.circle(frame, (wx, wy), 7, (40, 40, 200), -1)     # Wrist (Red/Blue)
        cv2.circle(frame, (tx, ty), 6, (0, 140, 255), -1)    # Thumb
        cv2.circle(frame, (ix, iy), 6, (0, 200, 255), -1)    # Index

        writer.write(frame)

    writer.release()
    return True


if __name__ == "__main__":
    generate_sample_video()
