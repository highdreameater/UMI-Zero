"""Trajectory logging module for recording frame-by-frame ICT tokens and robot actions."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder converting NumPy types to Python primitives."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


class PipelineLogger:
    """Records frame-by-frame pipeline states into structured JSON trajectories."""

    def __init__(self, log_filepath: str = "data/output/trajectory_log.json"):
        """Initializes the trajectory logger.

        Args:
            log_filepath: Path where output JSON file will be written.
        """
        self.log_filepath = log_filepath
        self.logs: List[Dict[str, Any]] = []

        # Ensure parent directory exists
        out_dir = os.path.dirname(self.log_filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def log_frame(
        self,
        frame_idx: int,
        ict_token: Dict[str, Any],
        robot_joints: List[float],
        extra_data: Dict[str, Any] | None = None,
    ) -> None:
        """Appends temporal frame tracking data to log structure.

        Args:
            frame_idx: Sequential frame index.
            ict_token: Interaction-Centric Token (SE(3) transforms, grasp state, pinch).
            robot_joints: Commanded manipulator joint angles (radians).
            extra_data: Optional dictionary with additional telemetry.
        """
        # Convert NumPy arrays to lists for serialization
        token_serializable: Dict[str, Any] = {}
        for k, v in ict_token.items():
            if isinstance(v, np.ndarray):
                token_serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, float)):
                token_serializable[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                token_serializable[k] = int(v)
            elif isinstance(v, (np.bool_, bool)):
                token_serializable[k] = bool(v)
            else:
                token_serializable[k] = v

        entry = {
            "frame_idx": int(frame_idx),
            "timestamp": token_serializable.get("timestamp", 0.0),
            "ict_token": token_serializable,
            "robot_joints": [float(q) for q in robot_joints],
        }

        if extra_data:
            for ek, ev in extra_data.items():
                if isinstance(ev, np.ndarray):
                    entry[ek] = ev.tolist()
                else:
                    entry[ek] = ev

        self.logs.append(entry)

    def save(self) -> None:
        """Flushes memory buffers and writes log file to disk."""
        out_dir = os.path.dirname(self.log_filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        payload = {
            "total_frames": len(self.logs),
            "format": "Interaction-Centric-Tokens-v1.0",
            "trajectories": self.logs,
        }

        with open(self.log_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, cls=NumpyEncoder, indent=2)

    def get_trajectories(self) -> List[Dict[str, Any]]:
        """Returns the in-memory recorded trajectory list."""
        return list(self.logs)
