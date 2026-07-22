"""Pose detection, dancer tracking, alignment, and Just Dance scoring tools."""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


BONUS_DIR = Path(__file__).resolve().parent
MODEL_DIR = BONUS_DIR / "models"
OUTPUT_DIR = BONUS_DIR / "outputs"
DEFAULT_POSE_MODEL = MODEL_DIR / "yolov8n-pose.pt"

KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

LEFT_RIGHT_PAIRS = (
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16),
)

BODY_INDICES = np.arange(5, 17, dtype=np.int64)
JOINT_WEIGHTS = np.asarray(
    [1.0, 1.0, 1.15, 1.15, 1.35, 1.35, 1.0, 1.0, 1.2, 1.2, 1.35, 1.35],
    dtype=np.float32,
)

ANGLE_TRIPLETS = (
    (5, 7, 9),    # left elbow
    (6, 8, 10),   # right elbow
    (11, 13, 15), # left knee
    (12, 14, 16), # right knee
    (7, 5, 11),   # left shoulder
    (8, 6, 12),   # right shoulder
    (5, 11, 13),  # left hip
    (6, 12, 14),  # right hip
)

BODY_AREAS = {
    "left arm": (5, 7, 9),
    "right arm": (6, 8, 10),
    "left leg": (11, 13, 15),
    "right leg": (12, 14, 16),
    "torso": (5, 6, 11, 12),
}


@dataclass(frozen=True)
class Pose:
    """One detected COCO pose in image-normalized coordinates."""

    points: np.ndarray
    confidence: np.ndarray
    box: tuple[float, float, float, float]
    detection_confidence: float = 1.0
    selection_score: float = 0.0
    aspect_ratio: float = 1.0

    def visible_mask(self, threshold: float = 0.25) -> np.ndarray:
        """Return keypoints that are confident, finite, and inside a broad image area."""
        finite = np.isfinite(self.points).all(axis=1)
        plausible = (self.points > -0.25).all(axis=1) & (self.points < 1.25).all(axis=1)
        return finite & plausible & (self.confidence >= threshold)

    @property
    def center(self) -> np.ndarray:
        """Return the normalized center of the person bounding box."""
        x1, y1, x2, y2 = self.box
        return np.asarray(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dtype=np.float32)

    @property
    def area(self) -> float:
        """Return normalized bounding-box area."""
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class NormalizedPose:
    """A pose centered and scaled by torso geometry."""

    points: np.ndarray
    confidence: np.ndarray
    scale: float


@dataclass(frozen=True)
class PoseComparison:
    """Detailed similarity between two spatially normalized poses."""

    score: float
    position_score: float
    angle_score: float
    direction_score: float
    coverage: float
    matched_joints: int
    mirrored: bool
    feedback: str
    weakest_area: str


@dataclass(frozen=True)
class DanceMatch:
    """A pose match augmented with temporal lag and smoothed game score."""

    comparison: PoseComparison
    raw_score: float
    adjusted_score: float
    score: float
    lag_seconds: float


@dataclass
class _CandidateTrack:
    """Short pose history used to measure motion before locking a dancer."""

    track_id: int
    pose: Pose
    normalized: NormalizedPose | None
    center: np.ndarray
    velocity: np.ndarray
    activity: float
    active_seconds: float
    moving_limbs: int
    last_seen: float


class PoseEstimator:
    """Small Ultralytics adapter that keeps the rest of the project testable."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_POSE_MODEL,
        confidence: float = 0.25,
        image_size: int = 512,
        device: str | None = None,
    ) -> None:
        settings_dir = OUTPUT_DIR / ".ultralytics"
        settings_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
        matplotlib_dir = OUTPUT_DIR / ".matplotlib"
        matplotlib_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run 'python -m pip install "
                "torch torchvision' and then 'python -m pip install --no-deps "
                "ultralytics'."
            ) from exc

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Pose model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def infer(self, frame: np.ndarray) -> tuple[list[Pose], float]:
        """Return every detected pose and inference latency in milliseconds."""
        start = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not results:
            return [], elapsed_ms

        result = results[0]
        if result.keypoints is None or result.keypoints.xyn is None:
            return [], elapsed_ms
        points = result.keypoints.xyn.detach().cpu().numpy()
        if len(points) == 0:
            return [], elapsed_ms

        if result.keypoints.conf is None:
            keypoint_confidence = np.ones(points.shape[:2], dtype=np.float32)
        else:
            keypoint_confidence = result.keypoints.conf.detach().cpu().numpy()

        boxes = None
        box_confidence = None
        if result.boxes is not None and result.boxes.xyxyn is not None:
            boxes = result.boxes.xyxyn.detach().cpu().numpy()
            box_confidence = result.boxes.conf.detach().cpu().numpy()

        poses: list[Pose] = []
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / max(frame_height, 1)
        for index, person_points in enumerate(points):
            if person_points.shape != (17, 2):
                continue
            confidence = np.asarray(keypoint_confidence[index], dtype=np.float32)
            if boxes is not None and index < len(boxes):
                box_values = np.clip(boxes[index], 0.0, 1.0)
                box = tuple(float(value) for value in box_values)
                detection_confidence = float(box_confidence[index])
            else:
                box = _box_from_points(person_points, confidence)
                detection_confidence = float(confidence.mean())
            poses.append(
                Pose(
                    points=np.asarray(person_points, dtype=np.float32),
                    confidence=confidence,
                    box=box,
                    detection_confidence=detection_confidence,
                    aspect_ratio=aspect_ratio,
                )
            )
        return poses, elapsed_ms


class MainDancerTracker:
    """Automatically lock a sustained dancer and survive short occlusions."""

    ACTIVITY_THRESHOLD = 0.9
    LOCK_AFTER_SECONDS = 0.6
    OCCLUSION_GRACE_SECONDS = 1.0
    LIMB_SPEED_THRESHOLD = 0.75

    def __init__(self, visibility_threshold: float = 0.25) -> None:
        self.visibility_threshold = visibility_threshold
        self.tracks: dict[int, _CandidateTrack] = {}
        self.next_track_id = 0
        self.locked_track_id: int | None = None
        self.state = "SEARCHING"
        self.last_timestamp: float | None = None

    @property
    def is_locked(self) -> bool:
        return self.locked_track_id is not None and self.state == "LOCKED"

    def reset(self) -> None:
        """Forget candidate motion, the locked identity, and occlusion state."""
        self.tracks.clear()
        self.next_track_id = 0
        self.locked_track_id = None
        self.state = "SEARCHING"
        self.last_timestamp = None

    def select(self, poses: Sequence[Pose], timestamp: float | None = None) -> Pose | None:
        """Update candidate motion and return the locked or best temporary pose."""
        timestamp = self._timestamp(timestamp)
        visible_tracks = self._update_tracks(poses, timestamp)

        if self.locked_track_id is not None:
            locked = self.tracks.get(self.locked_track_id)
            if locked is not None and locked.track_id in visible_tracks:
                self.state = "LOCKED"
                return replace(locked.pose, selection_score=self._tie_break_score(locked.pose))
            if locked is not None and timestamp - locked.last_seen <= self.OCCLUSION_GRACE_SECONDS:
                self.state = "OCCLUDED"
                return None
            self.locked_track_id = None

        eligible = [
            self.tracks[track_id]
            for track_id in visible_tracks
            if self._is_dancing(self.tracks[track_id])
        ]
        if eligible:
            # Activity determines who is dancing. If several people dance,
            # centrality and keypoint visibility decide the main dancer.
            chosen = max(eligible, key=lambda track: self._tie_break_score(track.pose))
            self.locked_track_id = chosen.track_id
            self.state = "LOCKED"
            return replace(chosen.pose, selection_score=self._tie_break_score(chosen.pose))

        self.state = "SEARCHING"
        if not visible_tracks:
            return None
        temporary = max(
            (self.tracks[track_id] for track_id in visible_tracks),
            key=lambda track: self._tie_break_score(track.pose),
        )
        return replace(temporary.pose, selection_score=self._tie_break_score(temporary.pose))

    def _timestamp(self, timestamp: float | None) -> float:
        if timestamp is None:
            timestamp = 0.0 if self.last_timestamp is None else self.last_timestamp + 1.0 / 30.0
        timestamp = float(timestamp)
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            timestamp = self.last_timestamp + 1.0 / 30.0
        self.last_timestamp = timestamp
        return timestamp

    def _update_tracks(self, poses: Sequence[Pose], timestamp: float) -> set[int]:
        candidates = [(pose, normalize_pose(pose, self.visibility_threshold)) for pose in poses]
        pair_costs: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            elapsed = float(np.clip(timestamp - track.last_seen, 1.0 / 120.0, 0.5))
            predicted_center = track.center + track.velocity * elapsed
            for candidate_index, (pose, normalized) in enumerate(candidates):
                center_distance = float(np.linalg.norm(pose.center - predicted_center))
                area_ratio = max(pose.area, 1e-5) / max(track.pose.area, 1e-5)
                pose_distance = _normalized_pose_distance(track.normalized, normalized)
                if not (0.30 <= area_ratio <= 3.0):
                    continue
                if center_distance > 0.32 or pose_distance > 1.4:
                    continue
                cost = (
                    center_distance / 0.32
                    + 0.25 * abs(math.log(area_ratio))
                    + 0.35 * pose_distance
                )
                pair_costs.append((cost, track_id, candidate_index))

        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        for _cost, track_id, candidate_index in sorted(pair_costs):
            if track_id in used_tracks or candidate_index in used_candidates:
                continue
            track = self.tracks[track_id]
            pose, normalized = candidates[candidate_index]
            elapsed = float(np.clip(timestamp - track.last_seen, 1.0 / 120.0, 0.5))
            motion, moving_limbs = _pose_motion(
                track.normalized,
                normalized,
                elapsed,
                self.LIMB_SPEED_THRESHOLD,
            )
            blend = 1.0 - math.exp(-elapsed / 0.35)
            activity = (1.0 - blend) * track.activity + blend * motion
            is_active = activity >= self.ACTIVITY_THRESHOLD and (
                moving_limbs >= 2 or motion >= 2.0
            )
            active_seconds = (
                track.active_seconds + elapsed
                if is_active
                else max(0.0, track.active_seconds - 1.5 * elapsed)
            )
            measured_velocity = (pose.center - track.center) / elapsed
            track.pose = pose
            track.normalized = normalized
            track.velocity = 0.65 * track.velocity + 0.35 * measured_velocity
            track.center = pose.center
            track.activity = activity
            track.active_seconds = active_seconds
            track.moving_limbs = moving_limbs
            track.last_seen = timestamp
            used_tracks.add(track_id)
            used_candidates.add(candidate_index)

        for candidate_index, (pose, normalized) in enumerate(candidates):
            if candidate_index in used_candidates:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = _CandidateTrack(
                track_id=track_id,
                pose=pose,
                normalized=normalized,
                center=pose.center,
                velocity=np.zeros(2, dtype=np.float32),
                activity=0.0,
                active_seconds=0.0,
                moving_limbs=0,
                last_seen=timestamp,
            )
            used_tracks.add(track_id)

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if timestamp - track.last_seen > self.OCCLUSION_GRACE_SECONDS
        ]
        for track_id in expired:
            del self.tracks[track_id]
        return used_tracks

    def _is_dancing(self, track: _CandidateTrack) -> bool:
        return (
            track.active_seconds >= self.LOCK_AFTER_SECONDS
            and track.activity >= self.ACTIVITY_THRESHOLD
            and (track.moving_limbs >= 2 or track.activity >= 2.0)
        )

    def _tie_break_score(self, pose: Pose) -> float:
        visibility = float(
            pose.visible_mask(self.visibility_threshold)[BODY_INDICES].mean()
        )
        centrality = max(
            0.0,
            1.0
            - float(np.linalg.norm(pose.center - np.asarray((0.5, 0.5))))
            / math.sqrt(0.5),
        )
        return 0.55 * centrality + 0.45 * visibility


class TemporalPoseMatcher:
    """Match a user's current pose against recent reference poses."""

    def __init__(
        self,
        max_lag_seconds: float = 1.5,
        timing_penalty_per_second: float = 4.0,
        smoothing: float = 0.25,
        visibility_threshold: float = 0.25,
        allow_mirror: bool = True,
    ) -> None:
        self.max_lag_seconds = max_lag_seconds
        self.timing_penalty_per_second = timing_penalty_per_second
        self.smoothing = smoothing
        self.visibility_threshold = visibility_threshold
        self.allow_mirror = allow_mirror
        self.reference_history: deque[tuple[float, NormalizedPose]] = deque()
        self.smoothed_score: float | None = None
        self.mirror_locked: bool | None = None
        self._direct_calibration: deque[float] = deque(maxlen=24)
        self._mirror_calibration: deque[float] = deque(maxlen=24)

    def reset(self) -> None:
        """Clear pose history and score smoothing."""
        self.reference_history.clear()
        self.smoothed_score = None
        self.mirror_locked = None
        self._direct_calibration.clear()
        self._mirror_calibration.clear()

    def add_reference(self, pose: Pose | None, timestamp: float) -> None:
        """Add one valid reference pose to the temporal search window."""
        self._prune(timestamp)
        if pose is None:
            return
        normalized = normalize_pose(pose, self.visibility_threshold)
        if normalized is not None:
            self.reference_history.append((timestamp, normalized))

    def match(self, user_pose: Pose | None, timestamp: float) -> DanceMatch | None:
        """Find the best recent reference pose and apply a small lag penalty."""
        self._prune(timestamp)
        if user_pose is None or not self.reference_history:
            return None
        normalized_user = normalize_pose(user_pose, self.visibility_threshold)
        if normalized_user is None:
            return None

        best_direct: tuple[float, float, PoseComparison] | None = None
        best_mirror: tuple[float, float, PoseComparison] | None = None
        mirrored_user = mirror_pose(normalized_user) if self.allow_mirror else None
        for reference_time, reference_pose in self.reference_history:
            direct = compare_poses(
                reference_pose,
                normalized_user,
                visibility_threshold=self.visibility_threshold,
                allow_mirror=False,
            )
            lag = max(0.0, timestamp - reference_time)
            direct_adjusted = direct.score - self.timing_penalty_per_second * lag
            if best_direct is None or direct_adjusted > best_direct[0]:
                best_direct = (direct_adjusted, lag, direct)
            if mirrored_user is not None:
                mirrored = _compare_once(
                    reference_pose,
                    mirrored_user,
                    self.visibility_threshold,
                    mirrored=True,
                )
                mirror_adjusted = mirrored.score - self.timing_penalty_per_second * lag
                if best_mirror is None or mirror_adjusted > best_mirror[0]:
                    best_mirror = (mirror_adjusted, lag, mirrored)

        if best_direct is None:
            return None
        if best_mirror is not None and self.mirror_locked is None:
            self._direct_calibration.append(best_direct[2].score)
            self._mirror_calibration.append(best_mirror[2].score)
            if len(self._direct_calibration) == self._direct_calibration.maxlen:
                direct_mean = float(np.mean(self._direct_calibration))
                mirror_mean = float(np.mean(self._mirror_calibration))
                self.mirror_locked = mirror_mean > direct_mean + 1.0

        if self.mirror_locked is True and best_mirror is not None:
            best = best_mirror
        elif self.mirror_locked is False or best_mirror is None:
            best = best_direct
        else:
            # During the short auto-calibration window, show the better mode.
            best = best_mirror if best_mirror[0] > best_direct[0] else best_direct
        adjusted, lag, comparison = best
        adjusted = float(np.clip(adjusted, 0.0, 100.0))
        if self.smoothed_score is None:
            self.smoothed_score = adjusted
        else:
            self.smoothed_score = (
                (1.0 - self.smoothing) * self.smoothed_score
                + self.smoothing * adjusted
            )
        return DanceMatch(
            comparison=comparison,
            raw_score=comparison.score,
            adjusted_score=adjusted,
            score=float(self.smoothed_score),
            lag_seconds=lag,
        )

    def _prune(self, timestamp: float) -> None:
        while self.reference_history and timestamp - self.reference_history[0][0] > self.max_lag_seconds:
            self.reference_history.popleft()


def normalize_pose(
    pose: Pose,
    visibility_threshold: float = 0.25,
) -> NormalizedPose | None:
    """Remove image translation and scale while preserving body lean and motion."""
    points = np.asarray(pose.points, dtype=np.float32).copy()
    # xyn uses separate image-width and image-height units. Convert x to
    # height-normalized units before computing Euclidean geometry so 16:9 and
    # 4:3 sources remain comparable without horizontal distortion.
    points[:, 0] *= float(pose.aspect_ratio)
    confidence = np.asarray(pose.confidence, dtype=np.float32).copy()
    visible = pose.visible_mask(visibility_threshold)

    center = _pair_midpoint(points, visible, 11, 12)
    if center is None:
        torso_indices = np.asarray([5, 6, 11, 12])
        available = torso_indices[visible[torso_indices]]
        if len(available) < 2:
            return None
        center = points[available].mean(axis=0)

    shoulder_midpoint = _pair_midpoint(points, visible, 5, 6)
    hip_midpoint = _pair_midpoint(points, visible, 11, 12)
    torso_length = (
        float(np.linalg.norm(shoulder_midpoint - hip_midpoint))
        if shoulder_midpoint is not None and hip_midpoint is not None
        else 0.0
    )
    shoulder_width = _pair_distance(points, visible, 5, 6)
    hip_width = _pair_distance(points, visible, 11, 12)
    scales = [value for value in (torso_length, shoulder_width, hip_width) if value > 1e-4]
    if not scales:
        body_points = points[BODY_INDICES[visible[BODY_INDICES]]]
        if len(body_points) < 4:
            return None
        span = np.ptp(body_points, axis=0)
        scales = [float(max(span)) * 0.35]
    scale = float(np.mean(scales))
    if not np.isfinite(scale) or scale < 1e-5:
        return None

    normalized = (points - center) / scale
    normalized[~visible] = np.nan
    return NormalizedPose(points=normalized, confidence=confidence, scale=scale)


def _normalized_pose_distance(
    previous: NormalizedPose | None,
    current: NormalizedPose | None,
) -> float:
    """Return a small association cost between two normalized poses."""
    if previous is None or current is None:
        return 0.8
    common = (
        np.isfinite(previous.points).all(axis=1)
        & np.isfinite(current.points).all(axis=1)
    )
    indices = BODY_INDICES[common[BODY_INDICES]]
    if len(indices) < 4:
        return 0.8
    distance = float(
        np.mean(np.linalg.norm(previous.points[indices] - current.points[indices], axis=1))
    )
    return float(np.clip(distance, 0.0, 2.0))


def _pose_motion(
    previous: NormalizedPose | None,
    current: NormalizedPose | None,
    elapsed: float,
    limb_speed_threshold: float,
) -> tuple[float, int]:
    """Measure relative limb motion, excluding whole-body image translation."""
    if previous is None or current is None:
        return 0.0, 0
    common = (
        np.isfinite(previous.points).all(axis=1)
        & np.isfinite(current.points).all(axis=1)
    )
    movement_indices = np.asarray((7, 8, 9, 10, 13, 14, 15, 16), dtype=np.int64)
    indices = movement_indices[common[movement_indices]]
    if len(indices) < 2:
        return 0.0, 0
    speeds = np.linalg.norm(
        current.points[indices] - previous.points[indices],
        axis=1,
    ) / max(elapsed, 1e-6)
    motion = float(np.clip(np.mean(speeds), 0.0, 6.0))

    moving_limbs = 0
    for limb in ((5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16)):
        limb_indices = np.asarray(limb, dtype=np.int64)
        limb_indices = limb_indices[common[limb_indices]]
        if len(limb_indices) < 2:
            continue
        limb_speed = float(
            np.mean(
                np.linalg.norm(
                    current.points[limb_indices] - previous.points[limb_indices],
                    axis=1,
                )
            )
            / max(elapsed, 1e-6)
        )
        if limb_speed >= limb_speed_threshold:
            moving_limbs += 1
    return motion, moving_limbs


def mirror_pose(pose: NormalizedPose) -> NormalizedPose:
    """Mirror a normalized pose and swap anatomical left/right keypoints."""
    points = pose.points.copy()
    confidence = pose.confidence.copy()
    points[:, 0] *= -1.0
    for left, right in LEFT_RIGHT_PAIRS:
        points[[left, right]] = points[[right, left]]
        confidence[[left, right]] = confidence[[right, left]]
    return NormalizedPose(points=points, confidence=confidence, scale=pose.scale)


def flip_pose_for_display(pose: Pose | None) -> Pose | None:
    """Map raw-camera pose coordinates onto a horizontally flipped display frame."""
    if pose is None:
        return None
    points = pose.points.copy()
    points[:, 0] = 1.0 - points[:, 0]
    x1, y1, x2, y2 = pose.box
    return replace(
        pose,
        points=points,
        box=(1.0 - x2, y1, 1.0 - x1, y2),
    )


def compare_poses(
    reference: NormalizedPose,
    user: NormalizedPose,
    visibility_threshold: float = 0.25,
    allow_mirror: bool = True,
) -> PoseComparison:
    """Compare positions, limb directions, and joint angles on a 0-100 scale."""
    direct = _compare_once(reference, user, visibility_threshold, mirrored=False)
    if not allow_mirror:
        return direct
    mirrored = _compare_once(
        reference,
        mirror_pose(user),
        visibility_threshold,
        mirrored=True,
    )
    return mirrored if mirrored.score > direct.score else direct


def feedback_for_score(score: float) -> str:
    """Map the numeric score to presentation-friendly feedback."""
    if score >= 88.0:
        return "PERFECT"
    if score >= 75.0:
        return "SUPER"
    if score >= 60.0:
        return "GOOD"
    if score >= 40.0:
        return "KEEP MOVING"
    return "TRY AGAIN"


def draw_pose(
    frame: np.ndarray,
    pose: Pose | None,
    color: tuple[int, int, int] = (60, 240, 100),
    visibility_threshold: float = 0.25,
    draw_box: bool = True,
    label: str | None = None,
) -> None:
    """Draw a selected dancer's skeleton and optional selection box in place."""
    if pose is None:
        return
    height, width = frame.shape[:2]
    visible = pose.visible_mask(visibility_threshold)
    pixels = np.column_stack((pose.points[:, 0] * width, pose.points[:, 1] * height))
    pixels = np.rint(pixels).astype(np.int32)

    for first, second in SKELETON:
        if visible[first] and visible[second]:
            cv2.line(
                frame,
                tuple(pixels[first]),
                tuple(pixels[second]),
                color,
                3,
                cv2.LINE_AA,
            )
    for index in BODY_INDICES:
        if visible[index]:
            cv2.circle(frame, tuple(pixels[index]), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pixels[index]), 5, color, 2, cv2.LINE_AA)

    if draw_box:
        x1, y1, x2, y2 = pose.box
        top_left = (int(x1 * width), int(y1 * height))
        bottom_right = (int(x2 * width), int(y2 * height))
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        if label:
            cv2.putText(
                frame,
                label,
                (top_left[0], max(24, top_left[1] - 8)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )


def pose_visibility(pose: Pose, threshold: float = 0.25) -> np.ndarray:
    """Return a float visibility vector convenient for analysis aggregation."""
    return pose.visible_mask(threshold).astype(np.float32)


def _compare_once(
    reference: NormalizedPose,
    user: NormalizedPose,
    visibility_threshold: float,
    mirrored: bool,
) -> PoseComparison:
    reference_visible = np.isfinite(reference.points).all(axis=1) & (
        reference.confidence >= visibility_threshold
    )
    user_visible = np.isfinite(user.points).all(axis=1) & (
        user.confidence >= visibility_threshold
    )
    common = reference_visible & user_visible
    body_common = common[BODY_INDICES]
    matched = int(body_common.sum())
    coverage = float(matched / len(BODY_INDICES))

    if matched < 4:
        return PoseComparison(
            score=0.0,
            position_score=0.0,
            angle_score=0.0,
            direction_score=0.0,
            coverage=coverage,
            matched_joints=matched,
            mirrored=mirrored,
            feedback="MOVE INTO VIEW",
            weakest_area="insufficient visible joints",
        )

    indices = BODY_INDICES[body_common]
    weights = JOINT_WEIGHTS[body_common]
    distances = np.linalg.norm(reference.points[indices] - user.points[indices], axis=1)
    mean_squared = float(np.average(np.square(distances), weights=weights))
    position_score = 100.0 * math.exp(-mean_squared / (0.72 ** 2))

    angle_scores: list[float] = []
    for first, center, last in ANGLE_TRIPLETS:
        if common[first] and common[center] and common[last]:
            reference_angle = _joint_angle(reference.points[first], reference.points[center], reference.points[last])
            user_angle = _joint_angle(user.points[first], user.points[center], user.points[last])
            difference = abs(reference_angle - user_angle)
            difference = min(difference, 2 * math.pi - difference)
            angle_scores.append(max(0.0, 1.0 - difference / math.pi))
    angle_score = 100.0 * float(np.mean(angle_scores)) if angle_scores else position_score

    direction_scores: list[float] = []
    for first, second in SKELETON:
        if first < 5 or second < 5 or not (common[first] and common[second]):
            continue
        reference_vector = reference.points[second] - reference.points[first]
        user_vector = user.points[second] - user.points[first]
        denominator = float(np.linalg.norm(reference_vector) * np.linalg.norm(user_vector))
        if denominator > 1e-7:
            cosine = float(np.clip(np.dot(reference_vector, user_vector) / denominator, -1.0, 1.0))
            direction_scores.append((cosine + 1.0) * 0.5)
    direction_score = (
        100.0 * float(np.mean(direction_scores)) if direction_scores else position_score
    )

    coverage_factor = 0.65 + 0.35 * min(1.0, coverage / 0.75)
    score = coverage_factor * (
        0.50 * position_score + 0.35 * angle_score + 0.15 * direction_score
    )
    score = float(np.clip(score, 0.0, 100.0))
    weakest_area = _weakest_body_area(reference, user, common)
    return PoseComparison(
        score=score,
        position_score=float(position_score),
        angle_score=float(angle_score),
        direction_score=float(direction_score),
        coverage=coverage,
        matched_joints=matched,
        mirrored=mirrored,
        feedback=feedback_for_score(score),
        weakest_area=weakest_area,
    )


def _weakest_body_area(
    reference: NormalizedPose,
    user: NormalizedPose,
    common: np.ndarray,
) -> str:
    errors: list[tuple[float, str]] = []
    for name, raw_indices in BODY_AREAS.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        indices = indices[common[indices]]
        if len(indices) >= 2:
            error = float(
                np.mean(np.linalg.norm(reference.points[indices] - user.points[indices], axis=1))
            )
            errors.append((error, name))
    return max(errors)[1] if errors else "whole body"


def _joint_angle(first: np.ndarray, center: np.ndarray, last: np.ndarray) -> float:
    first_vector = first - center
    last_vector = last - center
    first_angle = math.atan2(float(first_vector[1]), float(first_vector[0]))
    last_angle = math.atan2(float(last_vector[1]), float(last_vector[0]))
    return (last_angle - first_angle) % (2 * math.pi)


def _pair_midpoint(
    points: np.ndarray,
    visible: np.ndarray,
    first: int,
    second: int,
) -> np.ndarray | None:
    if visible[first] and visible[second]:
        return (points[first] + points[second]) * 0.5
    if visible[first]:
        return points[first]
    if visible[second]:
        return points[second]
    return None


def _pair_distance(
    points: np.ndarray,
    visible: np.ndarray,
    first: int,
    second: int,
) -> float:
    if visible[first] and visible[second]:
        return float(np.linalg.norm(points[first] - points[second]))
    return 0.0


def _box_from_points(
    points: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.15,
) -> tuple[float, float, float, float]:
    available = points[confidence >= threshold]
    if len(available) == 0:
        return 0.0, 0.0, 1.0, 1.0
    minimum = np.clip(available.min(axis=0) - 0.04, 0.0, 1.0)
    maximum = np.clip(available.max(axis=0) + 0.04, 0.0, 1.0)
    return float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])


def mean_or_zero(values: Iterable[float]) -> float:
    """Return a safe arithmetic mean for report generation."""
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def video_cache_key(video_path: Path | str) -> str:
    """Return a stable filename key without collisions for generic YouTube.mp4 names."""
    path = Path(video_path)
    stem = path.stem
    if stem.lower() in {"youtube", "video", "download"}:
        stem = f"{path.parent.name}_{stem}"
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
