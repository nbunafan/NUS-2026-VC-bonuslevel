from __future__ import annotations

"""Convert normalized MediaPipe body landmarks into debounced game commands."""

import time
from dataclasses import dataclass

import numpy as np


LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12


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


class GestureController:
    """Convert normalized COCO body keypoints into discrete game actions.

    The detector first learns the player's torso length and shoulder width. All thresholds
    are relative to that body scale, which is more robust than fixed pixels when the phone
    distance or camera resolution changes. Jump fires on the first reliable frame where
    both wrists are above their corresponding shoulders. While that pose is held, periodic
    requests allow another jump after landing without lowering the arms first.
    """

    def __init__(self, calibration_frames: int = 18):
        self.calibration_frames = calibration_frames
        self.calibration_samples: list[tuple[float, float]] = []
        self.baseline: tuple[float, float] | None = None
        self.previous_left_out = False
        self.previous_right_out = False
        self.last_action_time = {"LEFT": 0.0, "RIGHT": 0.0, "JUMP": 0.0}
        # Match repeat requests to the 0.86-second game jump. This prevents airborne resets
        # while allowing a held pose to start the next jump as soon as the runner lands.
        self.cooldown = {"LEFT": 0.55, "RIGHT": 0.55, "JUMP": 0.86}

    def reset_calibration(self) -> None:
        self.calibration_samples.clear()
        self.baseline = None
        self.previous_left_out = False
        self.previous_right_out = False

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
        shoulders = [LEFT_SHOULDER, RIGHT_SHOULDER]
        core = shoulders + [LEFT_HIP, RIGHT_HIP]
        shoulders_visible = _visible(points, confidence, shoulders)
        left_hip_visible = _visible(points, confidence, [LEFT_HIP])
        right_hip_visible = _visible(points, confidence, [RIGHT_HIP])
        # The pelvis is needed only while learning body scale. After calibration, a two-hand
        # jump depends on shoulders and wrists, so a hip briefly leaving a laptop-camera frame
        # must not block an otherwise clear command. Lateral gestures still validate their
        # corresponding hip separately below.
        if not shoulders_visible:
            return GestureResult(None, "SHOW FULL BODY", self._progress(), 0.0)

        shoulder_mid = points[[LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=0)
        shoulder_width = float(np.linalg.norm(points[LEFT_SHOULDER] - points[RIGHT_SHOULDER]))
        if shoulder_width < 0.025:
            return GestureResult(None, "MOVE CLOSER", self._progress(), 0.0)

        if not self.calibrated:
            if not left_hip_visible or not right_hip_visible:
                return GestureResult(None, "SHOW FULL BODY", self._progress(), 0.0)
            hip_mid = points[[LEFT_HIP, RIGHT_HIP]].mean(axis=0)
            torso_length = float(np.linalg.norm(shoulder_mid - hip_mid))
            hip_width = float(np.linalg.norm(points[LEFT_HIP] - points[RIGHT_HIP]))
            if torso_length < 0.025 or hip_width < 0.015:
                return GestureResult(None, "MOVE CLOSER", self._progress(), 0.0)
            self.calibration_samples.append((torso_length, shoulder_width))
            if len(self.calibration_samples) >= self.calibration_frames:
                samples = np.asarray(self.calibration_samples, dtype=np.float32)
                self.baseline = tuple(np.median(samples, axis=0).tolist())
            return GestureResult(None, "STAND STILL", self._progress(), float(np.mean(confidence[core])))

        baseline_torso, baseline_shoulder = self.baseline

        wrists_visible = _visible(
            points,
            confidence,
            [LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER],
            threshold=0.25,
        )
        # Normalized image y grows downward. Direct shoulder comparison avoids head-point
        # estimation and a second confirmation frame, removing the two main latency sources.
        both_hands_raised = wrists_visible and (
            points[LEFT_WRIST, 1] < points[LEFT_SHOULDER, 1]
            and points[RIGHT_WRIST, 1] < points[RIGHT_SHOULDER, 1]
        )

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
        if both_hands_raised:
            candidates.append("JUMP")
        # During a two-hand raise, horizontal displacement must not leak into a lane change.
        elif left_out and not self.previous_left_out:
            candidates.append("LEFT")
        elif right_out and not self.previous_right_out:
            candidates.append("RIGHT")

        self.previous_left_out = left_out
        self.previous_right_out = right_out

        action = None
        for candidate in candidates:
            if now - self.last_action_time[candidate] >= self.cooldown[candidate]:
                self.last_action_time[candidate] = now
                action = candidate
                break

        state = action or ("HANDS UP" if both_hands_raised else "READY")
        visible_core = shoulders + ([LEFT_HIP] if left_hip_visible else []) + ([RIGHT_HIP] if right_hip_visible else [])
        return GestureResult(action, state, 1.0, float(np.mean(confidence[visible_core])))

    def _progress(self) -> float:
        if self.calibrated:
            return 1.0
        return min(1.0, len(self.calibration_samples) / max(1, self.calibration_frames))
