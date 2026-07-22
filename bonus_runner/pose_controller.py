from __future__ import annotations

"""Convert normalized MediaPipe body landmarks into debounced game commands."""

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16


@dataclass(frozen=True)
class GestureResult:
    action: str | None
    state: str
    calibration_progress: float
    confidence: float


def _visible(points: np.ndarray, confidence: np.ndarray, indices, threshold=0.30) -> bool:
    indices = np.asarray(indices, dtype=np.int64)
    return bool(
        np.all(confidence[indices] >= threshold)
        and np.all(points[indices, 0] > 0)
        and np.all(points[indices, 1] > 0)
    )


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = a - b
    second = c - b
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-6)
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class GestureController:
    """Convert normalized COCO body keypoints into discrete game actions.

    The detector first learns the player's neutral hip height, torso length and shoulder
    width. All thresholds are relative to that body scale, which is more robust than using
    fixed pixels when the phone distance or camera resolution changes. Edge-triggering and
    per-action cooldowns prevent one held pose from generating commands every frame.
    """

    def __init__(self, calibration_frames: int = 36):
        self.calibration_frames = calibration_frames
        self.calibration_samples: list[tuple[float, float, float]] = []
        self.baseline: tuple[float, float, float] | None = None
        self.previous_left_out = False
        self.previous_right_out = False
        self.previous_jump = False
        self.previous_smoothed_hip: float | None = None
        self.jump_candidate_frames = 0
        self.last_action_time = {"LEFT": 0.0, "RIGHT": 0.0, "JUMP": 0.0}
        # Jumping is edge-triggered rather than time-limited, so a player can jump again as
        # soon as the detector sees a new rise after the previous jump has ended.
        self.cooldown = {"LEFT": 0.55, "RIGHT": 0.55, "JUMP": 0.0}
        self.hip_history = deque(maxlen=5)

    def reset_calibration(self) -> None:
        self.calibration_samples.clear()
        self.baseline = None
        self.hip_history.clear()
        self.previous_smoothed_hip = None
        self.jump_candidate_frames = 0

    @property
    def calibrated(self) -> bool:
        return self.baseline is not None

    def update(
        self,
        points: np.ndarray | None,
        confidence: np.ndarray | None,
        now: float | None = None,
    ) -> GestureResult:
        now = time.monotonic() if now is None else now
        if points is None or confidence is None:
            return GestureResult(None, "NO PERSON", self._progress(), 0.0)

        points = np.asarray(points, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        core = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        if not _visible(points, confidence, core):
            return GestureResult(None, "SHOW FULL BODY", self._progress(), 0.0)

        shoulder_mid = points[[LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=0)
        hip_mid = points[[LEFT_HIP, RIGHT_HIP]].mean(axis=0)
        shoulder_width = float(np.linalg.norm(points[LEFT_SHOULDER] - points[RIGHT_SHOULDER]))
        torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))
        if shoulder_width < 0.025 or torso_length < 0.025:
            return GestureResult(None, "MOVE CLOSER", self._progress(), 0.0)

        if not self.calibrated:
            self.calibration_samples.append((float(hip_mid[1]), torso_length, shoulder_width))
            if len(self.calibration_samples) >= self.calibration_frames:
                samples = np.asarray(self.calibration_samples, dtype=np.float32)
                self.baseline = tuple(np.median(samples, axis=0).tolist())
            return GestureResult(None, "STAND STILL", self._progress(), float(np.mean(confidence[core])))

        baseline_hip, baseline_torso, baseline_shoulder = self.baseline
        self.hip_history.append(float(hip_mid[1]))
        smoothed_hip = float(np.mean(self.hip_history))

        # Require both a substantial hip rise and upward velocity. A hand wave may perturb the
        # detector box by a few pixels, but it cannot satisfy this two-part body-motion test.
        hip_rise = baseline_hip - smoothed_hip
        hip_velocity = 0.0 if self.previous_smoothed_hip is None else self.previous_smoothed_hip - smoothed_hip
        jump_started = (
            hip_rise > 0.16 * baseline_torso
            and hip_velocity > 0.0015
            and torso_length > 0.75 * baseline_torso
        )
        jump_still_elevated = hip_rise > 0.16 * baseline_torso and torso_length > 0.75 * baseline_torso
        jump_candidate = jump_started or (self.jump_candidate_frames > 0 and jump_still_elevated)
        self.jump_candidate_frames = self.jump_candidate_frames + 1 if jump_candidate else max(0, self.jump_candidate_frames - 1)
        # One confirmed rising sample is enough for responsive game control; the velocity
        # requirement above still rejects most arm-only detector jitter.
        jump_pose = self.jump_candidate_frames >= 1
        self.previous_smoothed_hip = smoothed_hip

        left_out = False
        if _visible(points, confidence, [LEFT_WRIST, LEFT_SHOULDER, LEFT_HIP]):
            left_out = (
                points[LEFT_WRIST, 0] < points[LEFT_SHOULDER, 0] - 0.38 * baseline_shoulder
                and points[LEFT_WRIST, 1] < points[LEFT_HIP, 1]
            )
        right_out = False
        if _visible(points, confidence, [RIGHT_WRIST, RIGHT_SHOULDER, RIGHT_HIP]):
            right_out = (
                points[RIGHT_WRIST, 0] > points[RIGHT_SHOULDER, 0] + 0.38 * baseline_shoulder
                and points[RIGHT_WRIST, 1] < points[RIGHT_HIP, 1]
            )

        candidates = []
        if jump_pose and not self.previous_jump:
            candidates.append("JUMP")
        elif left_out and not self.previous_left_out:
            candidates.append("LEFT")
        elif right_out and not self.previous_right_out:
            candidates.append("RIGHT")

        self.previous_jump = jump_pose
        self.previous_left_out = left_out
        self.previous_right_out = right_out

        action = None
        for candidate in candidates:
            if now - self.last_action_time[candidate] >= self.cooldown[candidate]:
                self.last_action_time[candidate] = now
                action = candidate
                break

        state = action or ("JUMPING" if jump_pose else "READY")
        return GestureResult(action, state, 1.0, float(np.mean(confidence[core])))

    def _progress(self) -> float:
        if self.calibrated:
            return 1.0
        return min(1.0, len(self.calibration_samples) / max(1, self.calibration_frames))
