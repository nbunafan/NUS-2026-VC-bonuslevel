from __future__ import annotations

"""Unit tests for two-hand jump and lateral gesture transitions."""

import unittest

import numpy as np

from pose_controller import GestureController


def neutral_pose():
    points = np.zeros((17, 2), dtype=np.float32)
    points[0] = (0.50, 0.16)
    points[1], points[2] = (0.48, 0.14), (0.52, 0.14)
    points[3], points[4] = (0.45, 0.16), (0.55, 0.16)
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

    def test_both_hands_above_shoulders_trigger_on_first_frame(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9], points[10] = (0.45, 0.24), (0.55, 0.24)
        self.assertEqual(controller.update(points, confidence, now=10.0).action, "JUMP")

    def test_held_raised_pose_repeats_without_lowering_arms(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9], points[10] = (0.45, 0.24), (0.55, 0.24)
        actions = [controller.update(points, confidence, now=10.0 + index * 0.1).action for index in range(20)]
        self.assertGreaterEqual(actions.count("JUMP"), 3)

    def test_one_raised_hand_never_triggers_jump(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9] = (0.45, 0.24)
        actions = [controller.update(points, confidence, now=10.0 + index * 0.1).action for index in range(5)]
        self.assertNotIn("JUMP", actions)

    def test_wrists_at_shoulder_height_are_not_a_jump(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9], points[10] = points[5], points[6]
        actions = [controller.update(points, confidence, now=10.0 + index * 0.1).action for index in range(4)]
        self.assertNotIn("JUMP", actions)

    def test_face_landmarks_are_not_required_for_fast_jump(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9], points[10] = (0.45, 0.24), (0.55, 0.24)
        confidence[:5] = 0.0
        self.assertEqual(controller.update(points, confidence, now=10.0).action, "JUMP")

    def test_low_confidence_wrist_cannot_trigger_jump(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9], points[10] = (0.45, 0.24), (0.55, 0.24)
        confidence[10] = 0.15
        actions = [controller.update(points, confidence, now=10.0 + index * 0.1).action for index in range(4)]
        self.assertNotIn("JUMP", actions)

    def test_moderate_confidence_wrists_trigger_without_extra_wait(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9], points[10] = (0.45, 0.24), (0.55, 0.24)
        confidence[9] = confidence[10] = 0.26
        self.assertEqual(controller.update(points, confidence, now=10.0).action, "JUMP")

    def test_jump_still_works_when_hips_leave_frame_after_calibration(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9], points[10] = (0.45, 0.24), (0.55, 0.24)
        confidence[11] = confidence[12] = 0.0
        self.assertEqual(controller.update(points, confidence, now=10.0).action, "JUMP")

    def test_two_hand_jump_has_priority_over_wide_arm_lane_gestures(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9], points[10] = (0.24, 0.24), (0.76, 0.24)
        actions = [controller.update(points, confidence, now=10.0 + index * 0.1).action for index in range(3)]
        self.assertEqual(actions, ["JUMP", None, None])

    def test_normal_left_and_right_waves_never_trigger_jump(self):
        controller = calibrated_controller()
        points, confidence = neutral_pose()
        points[9] = (0.25, 0.35)
        self.assertEqual(controller.update(points, confidence, now=10.0).action, "LEFT")
        self.assertIsNone(controller.update(points, confidence, now=10.1).action)

        neutral, _ = neutral_pose()
        controller.update(neutral, confidence, now=10.7)
        points = neutral.copy()
        points[10] = (0.75, 0.35)
        self.assertEqual(controller.update(points, confidence, now=11.0).action, "RIGHT")
        self.assertIsNone(controller.update(points, confidence, now=11.1).action)


if __name__ == "__main__":
    unittest.main()
