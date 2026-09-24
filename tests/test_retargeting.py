"""Unit tests for the Retargeting Engine (WorkspaceMapper, RobotMapper, IKSolver)."""

from __future__ import annotations

import unittest
import numpy as np

from src.retargeting.workspace_mapper import WorkspaceMapper
from src.retargeting.robot_mapper import RobotMapper
from src.retargeting.ik_solver import IKSolver
from src.representation.spatial_math import construct_se3_matrix


class TestRetargeting(unittest.TestCase):
    """Test suite for kinematics mapping, workspace scaling, and IK solving."""

    def test_workspace_mapper_scaling_and_offset(self):
        """Verifies workspace scaling, offset application, and boundary clipping."""
        scale = [1.2, 1.2, 1.2]
        offset = [0.4, 0.0, 0.1]
        bounds = {"x": [0.1, 1.0], "y": [-0.5, 0.5], "z": [0.0, 0.8]}

        mapper = WorkspaceMapper(scale_factors=scale, robot_base_offset=offset, bounds=bounds, align_orientation=False)

        # Human wrist pose
        t_human = np.array([0.1, 0.05, 0.2])
        T_human = construct_se3_matrix(np.eye(3), t_human)

        T_robot = mapper.map_pose(T_human)
        self.assertEqual(T_robot.shape, (4, 4))

        expected_t = (t_human * np.array(scale)) + np.array(offset)
        np.testing.assert_allclose(T_robot[:3, 3], expected_t, atol=1e-5)

    def test_workspace_mapper_bounds_clipping(self):
        """Verifies that out-of-bounds positions are safely clipped."""
        scale = [1.0, 1.0, 1.0]
        offset = [0.0, 0.0, 0.0]
        bounds = {"x": [0.2, 0.8], "y": [-0.3, 0.3], "z": [0.1, 0.7]}

        mapper = WorkspaceMapper(scale, offset, bounds=bounds, align_orientation=False)

        # Extreme position far outside workspace
        t_extreme = np.array([2.5, -1.8, 1.9])
        T_extreme = construct_se3_matrix(np.eye(3), t_extreme)

        T_clipped = mapper.map_pose(T_extreme)
        pos = T_clipped[:3, 3]

        self.assertAlmostEqual(pos[0], 0.8)
        self.assertAlmostEqual(pos[1], -0.3)
        self.assertAlmostEqual(pos[2], 0.7)

    def test_robot_mapper(self):
        """Verifies grasp translation into gripper finger positions."""
        mapper = RobotMapper(max_aperture=0.08)

        # Fully closed pinch (0.0m) -> 0.0m aperture
        cmd_closed = mapper.map_gripper_action(pinch_distance=0.0, human_max_pinch=0.10)
        self.assertAlmostEqual(cmd_closed, 0.0)

        # Fully open pinch (0.10m) -> 0.08m aperture
        cmd_open = mapper.map_gripper_action(pinch_distance=0.10, human_max_pinch=0.10)
        self.assertAlmostEqual(cmd_open, 0.08)

        # Halfway pinch (0.05m) -> 0.04m aperture
        cmd_half = mapper.map_gripper_action(pinch_distance=0.05, human_max_pinch=0.10)
        self.assertAlmostEqual(cmd_half, 0.04)

        # Finger split
        f1, f2 = mapper.get_finger_positions(0.08)
        self.assertAlmostEqual(f1, 0.04)
        self.assertAlmostEqual(f2, 0.04)

    def test_ik_solver_joint_limits(self):
        """Verifies IK solver respects joint limits and outputs valid configurations."""
        solver = IKSolver(urdf_path="franka_panda/panda.urdf", ee_link_name="panda_hand")

        target_pos = np.array([0.45, 0.1, 0.3])
        T_target = construct_se3_matrix(np.eye(3), target_pos)

        current_joints = solver.home_configuration
        solution = solver.solve(current_joint_states=current_joints, target_ee_pose=T_target)

        self.assertEqual(len(solution), 7)

        # Verify all joint values are strictly within physical limits
        for i, angle in enumerate(solution):
            self.assertGreaterEqual(angle, solver.lower_limits[i] - 1e-4)
            self.assertLessEqual(angle, solver.upper_limits[i] + 1e-4)


if __name__ == "__main__":
    unittest.main()
