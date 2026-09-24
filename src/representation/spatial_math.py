"""Spatial math utilities for SE(3) Lie group operations and 3D transformations."""

from __future__ import annotations

import numpy as np


def construct_se3_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Constructs a 4x4 homogeneous transformation matrix from a 3x3 rotation and 3x1 translation.

    Args:
        rotation: 3x3 rotation matrix (SO(3)).
        translation: 3-element translation vector (meters).

    Returns:
        4x4 homogeneous transformation matrix in SE(3).
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=np.float64).flatten()
    return T


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Computes the exact analytical inverse of an SE(3) homogeneous transformation matrix.

    Given T = [R, t; 0, 1], T^-1 = [R^T, -R^T * t; 0, 1].

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        4x4 inverted transformation matrix.
    """
    T_inv = np.eye(4, dtype=np.float64)
    R_transpose = T[:3, :3].T
    T_inv[:3, :3] = R_transpose
    T_inv[:3, 3] = -R_transpose @ T[:3, 3]
    return T_inv


def compute_relative_pose(T_camera_object: np.ndarray, T_camera_wrist: np.ndarray) -> np.ndarray:
    """Computes relative transformation from object frame to wrist frame.

    T_rel = (T_cam_obj)^-1 * T_cam_wrist

    Args:
        T_camera_object: 4x4 pose of target object in camera frame.
        T_camera_wrist: 4x4 pose of human wrist in camera frame.

    Returns:
        4x4 relative transformation matrix T_object_wrist.
    """
    return invert_se3(T_camera_object) @ T_camera_wrist


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix to a unit quaternion [qx, qy, qz, qw].

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Quaternion array [qx, qy, qz, qw] (PyBullet convention: x, y, z, w).
    """
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Converts a unit quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix.

    Args:
        q: Quaternion [qx, qy, qz, qw].

    Returns:
        3x3 rotation matrix.
    """
    qx, qy, qz, qw = q
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Computes rotation matrix from roll, pitch, yaw angles (radians, XYZ order)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx
