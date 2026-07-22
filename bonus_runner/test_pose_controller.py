from __future__ import annotations

"""Unit tests for synthetic jump, crouch, and lateral gesture transitions."""

import unittest

import numpy as np

from pose_controller import GestureController


def neutral_pose():
    points = np.zeros((17, 2), dtype=np.float32)
    points[5], points[6] = (0.45, 0.30), (0.55, 0.30)
    points[7], points[8] = (0.43, 0.41), (0.57, 0.41)
    points[9], points[10] = (0.43, 0.51), (0.57, 0.51)
    points[11], points[12] = (0.47, 0.50), (0.53, 0.50)
    points[13], points[14] = (0.47, 0.70), (0.53, 0.70)
    points[15], points[16] = (0.47, 0.91), (0.53, 0.91)
    confidence = np.ones(17, dtype=np.float32)
    return points, confidence


def calibrated_controller():
    controller = GestureController(calibration_frames=4)
    points, confidence = neutral_pose()
    for index in range(4):
        controller.update(points, confidence, now=float(index))
    return controller


class GestureControllerTest(unittest.TestCase):
    def test_left_arm_extension_is_edge_triggered(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9] = (0.34, 0.42)
        self.assertEqual(controller.update(points, confidence, now=10.0).action, "LEFT")
        self.assertIsNone(controller.update(points, confidence, now=11.0).action)

    def test_jump_uses_relative_hip_height(self):
        controller = calibrated_controller()
        neutral, confidence = neutral_pose()
        self.assertIsNone(controller.update(neutral, confidence, now=10.0).action)
        points = neutral.copy()
        points[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 1] -= 0.08
        actions = [controller.update(points, confidence, now=10.1 + index * 0.1).action for index in range(5)]
        self.assertEqual(actions.count("JUMP"), 1)

    def test_arm_wave_does_not_trigger_jump(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9] = (0.25, 0.35)
        self.assertEqual(controller.update(points, confidence, now=10.0).action, "LEFT")
        points[9] = (0.18, 0.32)
        self.assertIsNone(controller.update(points, confidence, now=10.1).action)

    def test_crouch_does_not_generate_a_game_action(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[[5, 6], 1] = 0.42
        points[[11, 12], 1] = 0.56
        self.assertIsNone(controller.update(points, confidence, now=10.0).action)


if __name__ == "__main__":
    unittest.main()
