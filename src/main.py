"""Core application entry point orchestrating the Bare-Hand 3D Robot Demonstration & Learning System."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict
import numpy as np

from src.utils.yaml_loader import load_yaml

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

from src.perception.camera_stream import CameraStream
from src.perception.hand_tracker import HandTracker
from src.perception.object_tracker import ObjectTracker
from src.perception.occlusion_handler import OcclusionHandler
from src.representation.ict_encoder import ICTEncoder
from src.retargeting.ik_solver import IKSolver
from src.retargeting.robot_mapper import RobotMapper
from src.retargeting.workspace_mapper import WorkspaceMapper
from src.simulation.pybullet_sim import PyBulletSim
from src.utils.logger import PipelineLogger
from src.utils.visualizer import Visualizer


def main() -> None:
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(
        description="Bare-Hand 3D Robot Demonstration & Learning Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline configuration YAML",
    )
    parser.add_argument(
        "--camera_config",
        type=str,
        default="config/camera_config.yaml",
        help="Path to camera configuration YAML",
    )
    parser.add_argument(
        "--robot_config",
        type=str,
        default="config/robot_config.yaml",
        help="Path to robot configuration YAML",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV GUI windows",
    )
    parser.add_argument(
        "--save_video",
        type=str,
        default=None,
        help="Optional path to record output visualization video",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum frames to process before terminating",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(" BARE-HAND 3D ROBOT DEMONSTRATION & LEARNING SYSTEM")
    print("=" * 80)

    # 1. Load Configurations
    pipeline_cfg = load_yaml(args.config)
    camera_cfg = load_yaml(args.camera_config)
    robot_cfg = load_yaml(args.robot_config)

    p_sec = pipeline_cfg.get("perception", {})
    r_sec = pipeline_cfg.get("retargeting", {})
    pipe_sec = pipeline_cfg.get("pipeline", {})

    output_dir = pipe_sec.get("output_dir", "data/output")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "trajectory_log.json")

    # 2. Instantiate Pipeline Modules
    print("[1/5] Ingesting camera stream configuration...")
    camera_stream = CameraStream(args.camera_config)
    camera_k = camera_stream.get_intrinsics()

    print("[2/5] Initializing perception modules & occlusion handlers...")
    hand_tracker = HandTracker(
        confidence=float(p_sec.get("hand_confidence", 0.7)),
        max_hands=int(p_sec.get("max_num_hands", 1)),
    )
    object_tracker = ObjectTracker(
        method=p_sec.get("object_tracking_method", "aruco_fallback"),
        marker_size=float(p_sec.get("marker_size_meters", 0.05)),
        camera_k=camera_k,
        aruco_dict_name=p_sec.get("aruco_dict", "DICT_4X4_50"),
        marker_id=int(p_sec.get("aruco_marker_id", 0)),
    )
    occlusion_handler = OcclusionHandler(
        state_dim=6,
        window_size=int(pipeline_cfg.get("representation", {}).get("smooth_window_size", 5)),
        dt=1.0 / float(camera_stream.fps),
    )

    print("[3/5] Instantiating Interaction Abstraction Engine (ICT)...")
    ict_encoder = ICTEncoder(pipeline_cfg)

    print("[4/5] Initializing Retargeting Engine & Kinematic Solvers...")
    workspace_scale = r_sec.get("workspace_scale", [1.2, 1.2, 1.2])
    workspace_offset = r_sec.get("workspace_offset", [0.4, 0.0, 0.1])
    workspace_mapper = WorkspaceMapper(workspace_scale, workspace_offset)

    target_robot_name = r_sec.get("target_robot", "franka_panda")
    specific_robot_cfg = robot_cfg.get(target_robot_name, robot_cfg)
    max_aperture = float(specific_robot_cfg.get("max_gripper_aperture", 0.08))
    robot_mapper = RobotMapper(max_aperture=max_aperture)

    print("[5/5] Connecting Simulation Digital Twin & Visualizer...")
    sim_gui = not args.headless
    pybullet_sim = PyBulletSim(robot_cfg, gui=sim_gui)
    ik_solver = IKSolver(
        urdf_path=specific_robot_cfg.get("urdf_path", "franka_panda/panda.urdf"),
        ee_link_name=specific_robot_cfg.get("ee_link_name", "panda_hand"),
        pybullet_client=pybullet_sim.client_id,
        robot_id=pybullet_sim.robot_id,
        joint_names=specific_robot_cfg.get("joint_names"),
        home_configuration=specific_robot_cfg.get("home_configuration"),
    )

    visualizer = Visualizer(window_name="Bare-Hand 3D Robot Teleop Pipeline")
    logger = PipelineLogger(log_filepath=log_path)

    # Optional video writer
    video_writer = None
    if args.save_video and cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Dashboard is 2x width (perception + simulation twin)
        dash_w = camera_stream.resolution[0] * 2
        dash_h = camera_stream.resolution[1]
        video_writer = cv2.VideoWriter(args.save_video, fourcc, camera_stream.fps, (dash_w, dash_h))

    print("\n✓ Pipeline initialized successfully. Starting real-time processing loop...")
    print("Press 'q' or ESC in visualizer window to terminate early.\n")

    frame_idx = 0
    start_time = time.time()
    last_fps_time = time.time()
    fps = float(camera_stream.fps)

    try:
        while True:
            t0 = time.time()
            if args.max_frames is not None and frame_idx >= args.max_frames:
                print(f"Reached max requested frames ({args.max_frames}).")
                break

            # 1. Grab Frame
            success, color_frame, _ = camera_stream.get_frame()
            if not success or color_frame is None:
                print("End of camera stream / video source reached.")
                break

            timestamp = frame_idx / float(camera_stream.fps)

            # 2. Perception: 3D Hand Tracking & Object 6-DoF Pose
            raw_hand_data = hand_tracker.process(color_frame, camera_k)
            raw_obj_data = object_tracker.process(color_frame)

            # 3. Handle Keypoint & Object Occlusions
            hand_data = occlusion_handler.update_hand(raw_hand_data)
            object_data = occlusion_handler.update_object(raw_obj_data)

            # 4. Interaction Abstraction: Generate ICT Token
            ict_token = ict_encoder.encode_frame(hand_data, object_data, timestamp)
            is_grasped = ict_token["is_grasped"]
            pinch_dist = ict_token["pinch_distance"]

            # 5. Motion Retargeting: Scale to Robot Workspace & Aperture
            T_human_wrist = ict_token["T_wrist_world"]
            T_robot_target = workspace_mapper.map_pose(T_human_wrist)
            target_gripper_aperture = robot_mapper.map_gripper_action(pinch_dist)

            # 6. Inverse Kinematics Solving
            target_arm_joints = ik_solver.solve(
                current_joint_states=pybullet_sim.current_arm_joints,
                target_ee_pose=T_robot_target,
            )

            # 7. Step Simulation Digital Twin
            target_obj_pose = object_data["object_pose"] if object_data["detected"] else None
            pybullet_sim.step(
                target_arm_joints=target_arm_joints,
                target_gripper_pos=target_gripper_aperture,
                target_object_pose=target_obj_pose,
            )

            # 8. Render Visual Twin Image
            sim_frame = pybullet_sim.render_camera(
                width=camera_stream.resolution[0],
                height=camera_stream.resolution[1],
            )

            # 9. Trajectory Logging
            logger.log_frame(
                frame_idx=frame_idx,
                ict_token=ict_token,
                robot_joints=target_arm_joints,
                extra_data={
                    "gripper_aperture": target_gripper_aperture,
                    "T_robot_target": T_robot_target,
                },
            )

            # Calculate FPS
            frame_idx += 1
            if frame_idx % 10 == 0:
                elapsed = time.time() - last_fps_time
                fps = 10.0 / max(elapsed, 1e-4)
                last_fps_time = time.time()

            # 10. Visualization Dashboard
            dashboard_frame = visualizer.render(
                color_frame=color_frame,
                hand_data=hand_data,
                object_data=object_data,
                grasp_state=is_grasped,
                sim_frame=sim_frame,
                info={
                    "frame_idx": frame_idx,
                    "fps": fps,
                    "pinch_distance": pinch_dist,
                },
            )

            if video_writer is not None:
                video_writer.write(dashboard_frame)

            if not args.headless:
                key = visualizer.show(dashboard_frame, wait_ms=1)
                if key in (27, ord("q"), ord("Q")):
                    print("\nUser requested termination (ESC/'q').")
                    break

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"Pipeline executed {frame_idx} frames in {total_time:.2f}s ({frame_idx / max(total_time, 1e-4):.1f} FPS avg)")

        # Save trajectory log
        print(f"Saving trajectory logs to: {log_path}")
        logger.save()

        # Release resources
        camera_stream.release()
        visualizer.close()
        pybullet_sim.close()
        if video_writer is not None:
            video_writer.release()

        print("✓ Shutdown completed successfully.")
        print("=" * 80)


if __name__ == "__main__":
    main()
