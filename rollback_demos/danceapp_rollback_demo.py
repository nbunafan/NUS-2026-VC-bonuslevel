from __future__ import annotations

"""Pose-normalized dance comparison application with buffered reference scoring."""

import os
import random
import hashlib
import argparse
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

# Ultralytics reads its configuration during import, so this must be set first. Keeping its
# settings inside the project avoids permission errors in the user's roaming profile.
DEMO_MODE = "all"
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent if APP_DIR.name == "rollback_demos" else APP_DIR
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from pose_utils import MainDancerTracker, Pose


model = YOLO(str(BASE_DIR / "yolov8n-pose.pt"))
MODEL_LOCK = threading.Lock()
PANEL_SIZE = (600, 380)
BG, SURFACE, SURFACE_2 = "#101419", "#171d24", "#202832"
TEXT, MUTED, ACCENT, DANGER = "#f4f7fa", "#96a3af", "#36c5d8", "#ff6b5f"

skeleton = [
    (0, 5), (0, 6),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

COMPARE_POINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def select_main_person(keypoints_xyn: np.ndarray, confidences: np.ndarray | None):
    if keypoints_xyn is None or len(keypoints_xyn) == 0:
        return None, None
    best_index = 0
    best_score = -1.0
    for index, pts in enumerate(keypoints_xyn):
        valid = (pts[:, 0] > 0) & (pts[:, 1] > 0)
        if confidences is not None:
            valid &= confidences[index] > 0.25
        if valid.sum() < 5:
            continue
        xs = pts[valid, 0]
        ys = pts[valid, 1]
        area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
        conf = float(confidences[index][valid].mean()) if confidences is not None else 1.0
        score = area * conf
        if score > best_score:
            best_score = score
            best_index = index
    if best_score < 0:
        return None, None
    conf = confidences[best_index] if confidences is not None else np.ones(17, dtype=np.float32)
    return keypoints_xyn[best_index], conf


def normalize_pose(points: np.ndarray, conf: np.ndarray | None):
    pts = np.asarray(points, dtype=np.float32)
    selected = pts[COMPARE_POINTS]
    valid = (selected[:, 0] > 0) & (selected[:, 1] > 0)
    if conf is not None:
        valid &= np.asarray(conf, dtype=np.float32)[COMPARE_POINTS] > 0.25
    if valid.sum() < 4:
        return None, None
    center = selected[valid].mean(axis=0)
    shifted = selected - center
    shoulder = np.linalg.norm(pts[5] - pts[6]) if np.all(pts[[5, 6]] > 0) else 0.0
    hip = np.linalg.norm(pts[11] - pts[12]) if np.all(pts[[11, 12]] > 0) else 0.0
    scale = max(float(shoulder), float(hip), 1e-3)
    return shifted / scale, valid


def pose_similarity(ref_pose, ref_conf, cam_pose, cam_conf, tolerance=1.0):
    if DEMO_MODE in ("coordinate", "all"):
        # Deliberate rollback applies only to scoring. Motion/keyframe extraction still uses
        # stable normalized geometry, otherwise the demo accidentally removes keyframes too.
        ref = np.asarray(ref_pose, dtype=np.float32)[COMPARE_POINTS]
        cam = np.asarray(cam_pose, dtype=np.float32)[COMPARE_POINTS]
        valid = (ref > 0).all(axis=1) & (cam > 0).all(axis=1)
        if ref_conf is not None:
            valid &= np.asarray(ref_conf)[COMPARE_POINTS] > 0.25
        if cam_conf is not None:
            valid &= np.asarray(cam_conf)[COMPARE_POINTS] > 0.25
        if valid.sum() < 4:
            return None
        ref_valid, cam_valid = ref[valid], cam[valid]
        joint_distance = float(np.linalg.norm(ref_valid - cam_valid, axis=1).mean())
        center_distance = float(np.linalg.norm(ref_valid.mean(axis=0) - cam_valid.mean(axis=0)))
        ref_scale = float(np.linalg.norm(np.ptp(ref_valid, axis=0)))
        cam_scale = float(np.linalg.norm(np.ptp(cam_valid, axis=0)))
        scale_difference = abs(ref_scale - cam_scale)
        # Ignore the UI tolerance control in this legacy mode. Raw coordinate, center, and
        # apparent body-size errors are intentionally punished to expose the old limitation.
        raw_error = joint_distance + 0.75 * center_distance + 0.50 * scale_difference
        return float(np.clip(100.0 * np.exp(-3.0 * raw_error), 0.0, 100.0))
    ref_norm, ref_valid = normalize_pose(ref_pose, ref_conf)
    cam_norm, cam_valid = normalize_pose(cam_pose, cam_conf)
    if ref_norm is None or cam_norm is None:
        return None
    valid = ref_valid & cam_valid
    if valid.sum() < 4:
        return None
    distances = np.linalg.norm(ref_norm[valid] - cam_norm[valid], axis=1)
    mean_distance = float(np.mean(distances))
    # Higher difficulty multipliers make the distance penalty gentler for beginners. This
    # preserves the direction of the score instead of simply multiplying points artificially.
    tolerance = max(float(tolerance), 1.0)
    score = 100.0 * np.exp(-1.35 * mean_distance / tolerance)
    return float(np.clip(score, 0.0, 100.0))


def _angle_difference(first: float, second: float):
    """Return the smallest absolute difference between two angles in radians."""
    return abs((first - second + np.pi) % (2.0 * np.pi) - np.pi)


def pose_motion_metrics(previous, current, anchor=None):
    """Measure both immediate and accumulated movement in normalized body space.

    A simple mean over all joints hides a small hand gesture because the stationary hips and
    legs dilute it. This metric therefore gives extra weight to wrists/elbows, uses the four
    most active joints, and also compares against the pose at the start of the motion. Slow
    movements can consequently accumulate into a key action even when no individual frame
    contains a large displacement.
    """
    empty = {"instant": 0.0, "top": 0.0, "cumulative": 0.0, "extremity": 0.0, "torso": 0.0, "activity": 0.0}
    if previous is None or current is None:
        return empty
    previous_norm, previous_valid = normalize_pose(previous, None)
    current_norm, current_valid = normalize_pose(current, None)
    if previous_norm is None or current_norm is None:
        return empty
    valid = previous_valid & current_valid
    if valid.sum() < 4:
        return empty

    displacement = np.linalg.norm(current_norm - previous_norm, axis=1)
    valid_displacement = displacement[valid]
    # COMPARE_POINTS order: shoulders, elbows, wrists, hips, knees, ankles.
    joint_weights = np.array([1.0, 1.0, 1.35, 1.35, 1.8, 1.8, 0.85, 0.85, 1.0, 1.0, 1.25, 1.25], dtype=np.float32)
    instant = float(np.average(valid_displacement, weights=joint_weights[valid]))
    top = float(np.sort(valid_displacement)[-min(4, valid_displacement.size):].mean())

    cumulative = 0.0
    if anchor is not None:
        anchor_norm, anchor_valid = normalize_pose(anchor, None)
        if anchor_norm is not None:
            accumulated_valid = valid & anchor_valid
            if accumulated_valid.sum() >= 4:
                accumulated = np.linalg.norm(current_norm - anchor_norm, axis=1)[accumulated_valid]
                cumulative = float(np.sort(accumulated)[-min(4, accumulated.size):].mean())

    extremity_indices = np.array([4, 5, 10, 11])  # wrists and ankles in normalized selection
    extremity_valid = valid[extremity_indices]
    extremity = float(displacement[extremity_indices][extremity_valid].mean()) if extremity_valid.any() else instant

    # Shoulder-line and hip-line rotation detects leaning/turning even when limb travel is low.
    torso_changes = []
    for left, right in ((0, 1), (6, 7)):
        if valid[left] and valid[right]:
            old_vector = previous_norm[right] - previous_norm[left]
            new_vector = current_norm[right] - current_norm[left]
            old_angle = float(np.arctan2(old_vector[1], old_vector[0]))
            new_angle = float(np.arctan2(new_vector[1], new_vector[0]))
            torso_changes.append(_angle_difference(old_angle, new_angle) / np.pi)
    torso = float(np.mean(torso_changes)) if torso_changes else 0.0
    activity = 0.45 * top + 0.30 * cumulative + 0.15 * extremity + 0.10 * torso
    return {"instant": instant, "top": top, "cumulative": cumulative, "extremity": extremity, "torso": torso, "activity": float(activity)}


def pose_activity(previous, current):
    """Backward-compatible activity value used by any external project experiments."""
    return pose_motion_metrics(previous, current, previous)["activity"]


def pose_side(points):
    """Return the side of the strongest wrist extension for Fruit Ninja placement."""
    if points is None:
        return None
    shoulder_center = float((points[5, 0] + points[6, 0]) * 0.5)
    left_extension = shoulder_center - float(points[9, 0])
    right_extension = float(points[10, 0]) - shoulder_center
    if max(left_extension, right_extension) < 0.12:
        return None
    return "left" if left_extension > right_extension else "right"


def feedback_from_score(score: float | None):
    if score is None:
        return "FIND POSE", (0, 0, 255)
    if score >= 85:
        return "PERFECT", (0, 255, 255)
    if score >= 70:
        return "SUPER", (0, 220, 0)
    if score >= 50:
        return "GOOD", (255, 180, 0)
    return "KEEP GOING", (0, 120, 255)


def discrete_score(score: float | None, timing_error: float, tolerance: float = 1.0):
    """Convert continuous similarity into one stable, fixed-value scoring event."""
    if score is None:
        return "KEEP GOING", 0, 0
    if score >= 88:
        label, points = "PERFECT", 100
    elif score >= 76:
        label, points = "AMAZING", 80
    elif score >= 62:
        label, points = "NICE", 60
    elif score >= 48:
        label, points = "GOOD", 40
    else:
        label, points = "KEEP GOING", 10
    timing_bonus = 10 if timing_error <= 0.15 else 5 if timing_error <= 0.35 else 0
    return label, points, timing_bonus


class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Just Dance Pose Scorer")
        self.root.geometry("1280x760")
        self.root.minsize(1080, 680)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.audio_path = None
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True

        # Each reference pose is scheduled for scoring after a short preview lead. This makes
        # the video visible first and gives the player a concrete 2.5-second preparation window.
        self.preview_seconds = 2.5
        self.reference_buffer = deque(maxlen=90)
        # Keyframes must survive independently of the high-frequency continuity frames.
        # Otherwise a 2.5-second preview can expire from the 90-frame mixed buffer before the
        # webcam reaches its scoring window, silently preventing encouragement effects.
        self.keyframe_buffer = deque(maxlen=40)
        self.next_reference_pose = None
        self.next_reference_keyframe = False
        self.previous_reference_pose = None
        self.last_keyframe_id = 0
        self.last_scored_keyframe = -1
        self.last_keyframe_time = 0.0
        # Motion-segment state. A keyframe is emitted at the representative pose near the end
        # of a movement, instead of immediately when one pair of frames crosses a threshold.
        self.motion_active = False
        self.motion_anchor_pose = None
        self.motion_candidate_pose = None
        self.motion_candidate_conf = None
        self.motion_peak_activity = 0.0
        self.motion_quiet_frames = 0
        self.motion_frames = 0
        self.motion_stable_frames = 0
        self.motion_last_cumulative = 0.0
        self.recent_motion = deque(maxlen=24)
        self.latest_cam_pose = None
        self.latest_cam_conf = None
        self.main_dancer_tracker = MainDancerTracker()
        self.latest_score = None
        self.total_score = 0.0
        self.score_frames = 0
        self.combo = 0
        self.dance_points = 0.0
        self.fruit_combo = 0
        self.fruit_points = 0.0
        self.bonus_points = 0
        self.keyframe_hits = 0
        self.fruits = []
        self.last_fruit_spawn = time.monotonic()
        self.last_fruit_hit = 0.0
        self.slash_trail = deque(maxlen=10)
        self.effect_text = ""
        self.effect_color = (255, 255, 255)
        self.effect_started = 0.0
        self.effect_particles = []
        self.lock = threading.Lock()
        self.latest_ref_frame = None
        self.latest_cam_frame = None
        self.ref_frame_version = self.cam_frame_version = 0
        self.rendered_ref_version = self.rendered_cam_version = -1
        self.ref_photo = self.cam_photo = None
        self.stop_event = threading.Event()

        self.feedback_var = tk.StringVar(value="READY")
        self.difficulty_var = tk.StringVar(value="2x")
        self.difficulty_multiplier = 2.0
        self.metrics_var = tk.StringVar(value="Dance 0    Fruit 0    Combo x0")
        self.prediction_var = tk.StringVar(value="NEXT MOVE  •  2.5s PREVIEW")
        self._configure_style()
        self._build_ui()
        # Tk widgets are refreshed only on the main thread. Worker-thread UI calls were the
        # primary source of flicker and occasional Tcl/Tk instability.
        self.root.after(33, self.render_latest_frames)

    def _configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=SURFACE)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 22))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 12))
        style.configure("Accent.TButton", background=ACCENT, foreground="#081014", padding=(14, 8), borderwidth=0)
        style.configure("Secondary.TButton", background=SURFACE_2, foreground=TEXT, padding=(12, 8), borderwidth=0)
        style.configure("Danger.TButton", background="#512a2c", foreground="#ffd6d2", padding=(12, 8), borderwidth=0)

    def _build_ui(self):
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18))
        outer.pack(fill=tk.BOTH, expand=True)
        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(header, text="Dance Motion Lab", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="YOLOv8 pose matching with reaction-delay compensation", style="Sub.TLabel").pack(side=tk.LEFT, padx=16, pady=(11, 0))
        difficulty = tk.Frame(header, bg=BG)
        difficulty.pack(side=tk.RIGHT, pady=(3, 0))
        tk.Label(difficulty, text="POSE TOLERANCE", bg=BG, fg=MUTED, font=("Segoe UI Semibold", 9)).pack(side=tk.LEFT, padx=(0, 7))
        for label, value in (("1x", 1.0), ("1.5x", 1.5), ("2x", 2.0), ("3x", 3.0)):
            tk.Radiobutton(
                difficulty, text=label, value=label, variable=self.difficulty_var,
                command=lambda selected=value: self.set_difficulty(selected),
                indicatoron=False, width=4, padx=3, pady=5,
                bg=SURFACE_2, fg=TEXT, selectcolor="#287c88",
                activebackground=ACCENT, activeforeground="#081014",
                relief=tk.FLAT, bd=0, font=("Segoe UI Semibold", 9),
            ).pack(side=tk.LEFT, padx=2)
        scorebar = tk.Frame(outer, bg=SURFACE_2, padx=14, pady=10)
        scorebar.pack(fill=tk.X, pady=(0, 14))
        self.feedback_label = tk.Label(scorebar, textvariable=self.feedback_var, bg=SURFACE_2, fg=ACCENT, font=("Segoe UI Semibold", 16), width=18, anchor="w")
        self.feedback_label.pack(side=tk.LEFT)
        tk.Label(scorebar, textvariable=self.metrics_var, bg=SURFACE_2, fg=TEXT, font=("Segoe UI", 12)).pack(side=tk.RIGHT)
        content = ttk.Frame(outer, style="App.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure((0, 1), weight=1, uniform="panels")
        content.rowconfigure(0, weight=1)
        self.left_frame, self.label_file, self.controls_file = self._panel(content, 0, "REFERENCE VIDEO")
        self.right_frame, self.label_cam, self.controls_cam = self._panel(content, 1, "WEBCAM PLAYER")
        ttk.Button(self.controls_file, text="Open", command=self.load_video, style="Secondary.TButton").pack(side=tk.LEFT)
        ttk.Button(self.controls_file, text="Play", command=self.start_video, style="Accent.TButton").pack(side=tk.LEFT, padx=7)
        ttk.Button(self.controls_file, text="Stop", command=self.stop_video, style="Danger.TButton").pack(side=tk.LEFT)
        ttk.Button(self.controls_cam, text="Start camera", command=self.start_cam, style="Accent.TButton").pack(side=tk.LEFT)
        ttk.Button(self.controls_cam, text="Stop", command=self.stop_cam, style="Danger.TButton").pack(side=tk.LEFT, padx=7)
        ttk.Button(self.controls_cam, text="Reset score", command=self.reset_score, style="Secondary.TButton").pack(side=tk.RIGHT)

        # Just Dance-style pictograms travel from right to left. They live outside the video
        # panels so the reference remains the original, unobstructed source video.
        timeline = tk.Frame(outer, bg=SURFACE_2, padx=12, pady=8)
        timeline.pack(fill=tk.X, pady=(12, 0))
        tk.Label(timeline, text="UPCOMING KEY MOVES", bg=SURFACE_2, fg="#ffd166", font=("Segoe UI Semibold", 10)).pack(side=tk.LEFT)
        tk.Label(timeline, textvariable=self.prediction_var, bg=SURFACE_2, fg=MUTED, font=("Segoe UI", 9), padx=12).pack(side=tk.LEFT)
        self.preview_canvas = tk.Canvas(timeline, width=560, height=82, bg="#080b0e", bd=0, highlightthickness=0)
        self.preview_canvas.pack(side=tk.RIGHT)

    @staticmethod
    def _panel(parent, column, title):
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        panel.grid(row=0, column=column, sticky="nsew", padx=(0, 7) if column == 0 else (7, 0))
        ttk.Label(panel, text=title, style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))
        label = tk.Label(panel, bg="#080b0e", bd=0, highlightthickness=0)
        label.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(panel, style="Panel.TFrame")
        controls.pack(fill=tk.X, pady=(10, 0))
        return panel, label, controls

    def load_video(self):
        path = filedialog.askopenfilename(
            initialdir=str(BASE_DIR),
            filetypes=[("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*")],
        )
        if path:
            self.stop_video_audio()
            self.video_path = path
            self.audio_path = None
            messagebox.showinfo("Video Selected", os.path.basename(path))

    def start_video(self):
        if not self.video_path:
            default_video = BASE_DIR / "dance_example_1.mp4"
            if default_video.exists():
                self.video_path = str(default_video)
            else:
                messagebox.showwarning("No Video", "Please select a video first.")
                return
        if not self.running_file:
            self.prepare_video_audio()
            self.running_file = True
            self.start_video_audio()
            threading.Thread(target=self.process_video_file, daemon=True).start()

    def stop_video(self):
        self.running_file = False
        self.stop_video_audio()
        if self.cap_file:
            self.cap_file.release()

    def prepare_video_audio(self):
        """Extract the selected video's original soundtrack once into a local WAV cache."""
        if not self.video_path or self.audio_path is not None:
            return
        source = Path(self.video_path)
        cache_dir = BASE_DIR / ".audio_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
        target = cache_dir / f"{digest}.wav"
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            self.audio_path = str(target)
            return
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [ffmpeg, "-y", "-i", str(source), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(target)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            self.audio_path = str(target)
        except Exception as error:
            # Video and pose scoring remain usable when the optional audio extraction fails.
            self.audio_path = None
            print(f"Audio extraction skipped: {error}")

    def start_video_audio(self):
        """Play the cached soundtrack asynchronously and loop it with the reference video."""
        if not self.audio_path:
            return
        try:
            import winsound
            winsound.PlaySound(self.audio_path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        except (ImportError, RuntimeError):
            pass

    @staticmethod
    def stop_video_audio():
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except (ImportError, RuntimeError):
            pass

    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame

    def start_cam(self):
        if not self.running_cam:
            self.main_dancer_tracker.reset()
            self.running_cam = True
            threading.Thread(target=self.process_webcam, daemon=True).start()

    def stop_cam(self):
        self.running_cam = False
        if self.cap_cam:
            self.cap_cam.release()

    def reset_score(self):
        with self.lock:
            self.latest_score = None
            self.total_score = 0.0
            self.score_frames = 0
            self.combo = 0
            self.dance_points = 0.0
            self.fruit_combo = 0
            self.fruit_points = 0.0
            self.bonus_points = 0
            self.keyframe_hits = 0
            self.last_scored_keyframe = -1
            self.fruits.clear()

    def set_difficulty(self, multiplier):
        """Apply a beginner-friendly pose tolerance immediately and thread-safely."""
        with self.lock:
            self.difficulty_multiplier = float(multiplier)

    def reset_motion_detector(self):
        """Reset reference extraction so replaying a video starts with a clean motion state."""
        self.previous_reference_pose = None
        self.motion_active = False
        self.motion_anchor_pose = None
        self.motion_candidate_pose = None
        self.motion_candidate_conf = None
        self.motion_peak_activity = 0.0
        self.motion_quiet_frames = 0
        self.motion_frames = 0
        self.motion_stable_frames = 0
        self.motion_last_cumulative = 0.0
        self.recent_motion.clear()

    def update_motion_detector(self, pose, conf):
        """Return a representative key pose when a detected movement settles.

        The start threshold adapts to recent camera/model noise but retains a low ceiling so
        deliberate small gestures remain detectable. Hysteresis uses a lower stop threshold;
        two quieter inference frames mark the end of an action and prevent rapid flickering.
        """
        if self.previous_reference_pose is None:
            self.previous_reference_pose = pose.copy()
            self.motion_anchor_pose = pose.copy()
            return None

        metrics = pose_motion_metrics(self.previous_reference_pose, pose, self.motion_anchor_pose)
        self.recent_motion.append(metrics["instant"])
        noise_floor = float(np.median(self.recent_motion)) if self.recent_motion else 0.0
        start_threshold = float(np.clip(0.055 + 1.8 * noise_floor, 0.075, 0.14))
        stop_threshold = max(0.028, start_threshold * 0.42)

        if not self.motion_active and metrics["activity"] >= start_threshold:
            self.motion_active = True
            self.motion_candidate_pose = pose.copy()
            self.motion_candidate_conf = None if conf is None else conf.copy()
            self.motion_peak_activity = metrics["activity"]
            self.motion_quiet_frames = 0
            self.motion_frames = 0
            self.motion_stable_frames = 0
            self.motion_last_cumulative = metrics["cumulative"]

        if self.motion_active:
            self.motion_frames += 1
            # Prefer a pose with strong accumulated change. Continuing to update during the
            # movement usually selects its visually meaningful end rather than a transition.
            if metrics["activity"] >= self.motion_peak_activity * 0.82 or metrics["cumulative"] >= self.motion_peak_activity:
                self.motion_candidate_pose = pose.copy()
                self.motion_candidate_conf = None if conf is None else conf.copy()
                self.motion_peak_activity = max(self.motion_peak_activity, metrics["activity"], metrics["cumulative"])
            self.motion_quiet_frames = self.motion_quiet_frames + 1 if metrics["instant"] < stop_threshold else 0
            cumulative_growth = metrics["cumulative"] - self.motion_last_cumulative
            self.motion_stable_frames = self.motion_stable_frames + 1 if cumulative_growth < 0.012 else 0
            self.motion_last_cumulative = max(self.motion_last_cumulative, metrics["cumulative"])

        emitted = None
        # YOLO keypoints contain a little jitter, so some videos never produce two perfectly
        # quiet samples. A plateau or a one-second upper bound also closes the movement and
        # guarantees that useful preview cards continue to be generated.
        motion_finished = self.motion_quiet_frames >= 2 or self.motion_stable_frames >= 3 or self.motion_frames >= 8
        if self.motion_active and motion_finished:
            # Reject tiny noise-only segments, but allow accumulated slow gestures that became
            # sufficiently different from their starting pose.
            final_metrics = pose_motion_metrics(self.motion_anchor_pose, self.motion_candidate_pose, self.motion_anchor_pose)
            if final_metrics["cumulative"] >= 0.075 or self.motion_peak_activity >= start_threshold * 1.15:
                emitted = (self.motion_candidate_pose.copy(), self.motion_candidate_conf, self.motion_peak_activity)
            self.motion_active = False
            self.motion_anchor_pose = pose.copy()
            self.motion_candidate_pose = None
            self.motion_candidate_conf = None
            self.motion_peak_activity = 0.0
            self.motion_quiet_frames = 0
            self.motion_frames = 0
            self.motion_stable_frames = 0
            self.motion_last_cumulative = 0.0

        self.previous_reference_pose = pose.copy()
        return emitted

    def process_video_file(self):
        self.cap_file = cv2.VideoCapture(self.video_path)
        fps = self.cap_file.get(cv2.CAP_PROP_FPS)
        fps = fps if 5 <= fps <= 60 else 30.0
        frame_period = 1.0 / fps
        frame_index = 0
        last_pose = last_conf = None
        with self.lock:
            self.reset_motion_detector()
            self.reference_buffer.clear()
            self.keyframe_buffer.clear()
        while self.cap_file.isOpened() and self.running_file:
            started = time.perf_counter()
            ret, frame = self.cap_file.read()
            if not ret:
                self.cap_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Restart both streams at the loop boundary to prevent long-term audio drift.
                self.stop_video_audio()
                self.start_video_audio()
                with self.lock:
                    self.reset_motion_detector()
                    self.reference_buffer.clear()
                    self.keyframe_buffer.clear()
                    self.last_keyframe_time = 0.0
                continue
            # Decode every frame for smooth playback but run heavy YOLO only about 8 times
            # per second. The last skeleton is reused between inference frames.
            infer_every = max(1, int(round(fps / 8.0)))
            inferred = frame_index % infer_every == 0
            if inferred:
                raw_frame = frame.copy()
                _annotated, last_pose, last_conf = self.process_pose(frame, is_reference=True)
                # Keep the reference video untouched. Future pose cards are rendered in the
                # separate timeline below, like a Just Dance cue lane.
                frame = raw_frame
            else:
                frame = frame.copy()
            pose, conf = last_pose, last_conf
            if pose is not None:
                with self.lock:
                    now = time.monotonic()
                    if DEMO_MODE in ("fixed_interval", "all"):
                        # Deliberate rollback: sample whatever pose happens to be visible every
                        # 2.0 seconds. It ignores motion magnitude and action completion.
                        detected = None
                        is_keyframe = inferred and now - self.last_keyframe_time >= 2.0
                    else:
                        # Improved extractor: advance only on a new inference and emit a
                        # representative pose when the detected motion segment settles.
                        detected = self.update_motion_detector(pose, conf) if inferred else None
                        is_keyframe = detected is not None and now - self.last_keyframe_time >= 0.35
                    if is_keyframe:
                        self.last_keyframe_id += 1
                        self.last_keyframe_time = now
                        if detected is None:
                            scheduled_pose, scheduled_conf, activity = pose, conf, 0.0
                        else:
                            scheduled_pose, scheduled_conf, activity = detected
                    else:
                        scheduled_pose, scheduled_conf, activity = pose, conf, 0.0
                    keyframe_id = self.last_keyframe_id if is_keyframe else -1
                    due_time = now + self.preview_seconds
                    entry = (due_time, scheduled_pose.copy(), None if scheduled_conf is None else scheduled_conf.copy(), is_keyframe, keyframe_id, activity)
                    self.reference_buffer.append(entry)
                    if is_keyframe:
                        self.keyframe_buffer.append(entry)
                    self.next_reference_pose = scheduled_pose.copy()
                    self.next_reference_keyframe = is_keyframe
            with self.lock:
                self.latest_ref_frame = frame
                self.ref_frame_version += 1
            frame_index += 1
            time.sleep(max(0.0, frame_period - (time.perf_counter() - started)))
        self.cap_file.release()

    def process_webcam(self):
        self.cap_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap_cam.isOpened():
            self.cap_cam.release()
            self.cap_cam = cv2.VideoCapture(0)
        self.cap_cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.cap_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_index = 0
        last_pose = last_conf = None
        while self.cap_cam.isOpened() and self.running_cam:
            ret, frame = self.cap_cam.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            # Infer every third camera frame and reuse the skeleton between updates. Together
            # with the reference stream this keeps total YOLO load practical on a CPU.
            inferred = frame_index % 3 == 0
            if inferred:
                frame, last_pose, last_conf = self.process_pose(frame, is_reference=False)
            else:
                frame = self.draw_pose_overlay(frame.copy(), last_pose, last_conf)
            pose, conf = last_pose, last_conf
            with self.lock:
                self.latest_cam_pose = pose
                self.latest_cam_conf = conf
                if inferred:
                    score, keyframe_score, keyframe_id, timing_error = self.scheduled_score(pose, conf)
                    self.latest_score = score
                    if score is not None:
                        self.score_frames += 1
                    if DEMO_MODE in ("continuous", "all") and score is not None:
                        # Deliberate rollback: every inferred camera frame adds points. The
                        # total therefore depends on frame rate and rises while holding a pose.
                        self.dance_points += score * 0.10
                        self.total_score = self.dance_points
                    elif keyframe_score is not None and keyframe_id != self.last_scored_keyframe:
                        self.last_scored_keyframe = keyframe_id
                        label, base_points, timing_bonus = discrete_score(keyframe_score, timing_error, self.difficulty_multiplier)
                        self.combo = self.combo + 1 if base_points >= 60 else 0
                        combo_bonus = min(max(self.combo - 1, 0), 4) * 5 if base_points >= 60 else 0
                        event_points = base_points + timing_bonus + combo_bonus
                        self.dance_points += event_points
                        self.total_score = self.dance_points
                        self.keyframe_hits += 1 if base_points >= 60 else 0
                        self.start_score_effect(keyframe_score, event_points)
            self.update_fruits(frame, pose)
            self.draw_score_effect(frame)
            with self.lock:
                self.latest_cam_frame = frame
                self.cam_frame_version += 1
            frame_index += 1
        self.cap_cam.release()

    def start_score_effect(self, score, awarded_points=0):
        """Create one rating burst showing the discrete points awarded for a keyframe."""
        if score >= 88:
            text, color = "PERFECT", (80, 255, 255)
        elif score >= 76:
            text, color = "AMAZING", (255, 120, 245)
        elif score >= 62:
            text, color = "NICE", (255, 210, 80)
        elif score >= 48:
            text, color = "GOOD", (110, 230, 120)
        else:
            text, color = "KEEP GOING", (90, 150, 255)
        text = f"{text} +{int(awarded_points)}"
        self.effect_text, self.effect_color = text, color
        self.effect_started = time.monotonic()
        self.effect_particles = [(random.uniform(-1, 1), random.uniform(-1, 1), random.randint(2, 5)) for _ in range(28)]

    def draw_score_effect(self, frame):
        """Draw outlined text, radial rays, and particles without changing the Tk layout."""
        age = time.monotonic() - self.effect_started
        if not self.effect_text or age >= 1.15:
            return
        height, width = frame.shape[:2]
        progress = age / 1.15
        pulse = 1.0 + 0.22 * np.sin(min(progress * 2.5, 1.0) * np.pi)
        scale = 1.15 * pulse
        thickness = 3
        (text_width, text_height), _ = cv2.getTextSize(self.effect_text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        origin = ((width - text_width) // 2, max(62, int(height * 0.18)))
        # Dark outline keeps the rating readable on any camera background.
        cv2.putText(frame, self.effect_text, origin, cv2.FONT_HERSHEY_DUPLEX, scale, (8, 10, 14), thickness + 6, cv2.LINE_AA)
        cv2.putText(frame, self.effect_text, origin, cv2.FONT_HERSHEY_DUPLEX, scale, self.effect_color, thickness, cv2.LINE_AA)
        center = (width // 2, origin[1] - text_height // 2)
        fade = max(0.0, 1.0 - progress)
        for dx, dy, radius in self.effect_particles:
            distance = int((45 + 150 * progress) * np.hypot(dx, dy))
            norm = max(np.hypot(dx, dy), 1e-5)
            x = int(center[0] + dx / norm * distance)
            y = int(center[1] + dy / norm * distance)
            color = tuple(int(channel * fade) for channel in self.effect_color)
            cv2.circle(frame, (x, y), max(1, int(radius * fade)), color, -1, cv2.LINE_AA)

    def scheduled_score(self, cam_pose, cam_conf):
        """Score the pose that became due after the 2.5-second preview window.

        The closest scheduled pose supplies a low-weight continuity score. A keyframe inside
        a wider timing window supplies 80% of the displayed score and becomes a major point
        event. This rewards arriving at important poses rather than matching every video frame.
        """
        if cam_pose is None:
            return None, None, -1, 9.0
        now = time.monotonic()
        tolerance = self.difficulty_multiplier
        entries = list(self.reference_buffer)
        # Easier modes forgive pose displacement and a small reaction-time error. The raw
        # score is not multiplied, so users still have to move in the correct direction.
        timing_window = 0.45 + 0.10 * (tolerance - 1.0)
        due = [entry for entry in entries if abs(entry[0] - now) <= timing_window]
        continuity = None
        if due:
            closest = min(due, key=lambda entry: abs(entry[0] - now))
            continuity = pose_similarity(closest[1], closest[2], cam_pose, cam_conf, tolerance)
        # Keyframes use their own durable queue. A slightly wider window tolerates webcam and
        # reference inference jitter while still scoring each keyframe only once by its ID.
        keyframe_window = 0.60 + 0.10 * (tolerance - 1.0)
        keyframes = [entry for entry in self.keyframe_buffer if abs(entry[0] - now) <= keyframe_window and entry[4] != self.last_scored_keyframe]
        if not keyframes:
            return continuity, None, -1, 9.0
        target = min(keyframes, key=lambda entry: abs(entry[0] - now))
        keyframe_score = pose_similarity(target[1], target[2], cam_pose, cam_conf, tolerance)
        if keyframe_score is None:
            return continuity, None, -1, 9.0
        continuity = continuity if continuity is not None else keyframe_score
        displayed = 0.80 * keyframe_score + 0.20 * continuity
        return displayed, keyframe_score, target[4], abs(target[0] - now)

    def process_pose(self, frame, is_reference: bool):
        # Ultralytics model objects are not safe or efficient when invoked concurrently. One
        # shared lock prevents the two streams from oversubscribing the CPU/GPU.
        with MODEL_LOCK:
            results = model.predict(frame, conf=0.3, imgsz=384, verbose=False)
        height, width = frame.shape[:2]
        overlay = frame.copy() if self.show_video_frame or not is_reference else np.ones_like(frame) * 255

        selected_pose = None
        selected_conf = None
        for result in results:
            if result.keypoints is None:
                continue
            keypoints_xyn = result.keypoints.xyn.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            if is_reference or DEMO_MODE in ("lock", "all"):
                # Deliberate rollback: select the largest visible person independently on every
                # frame. A passer-by can immediately replace the original player.
                selected_pose, selected_conf = select_main_person(keypoints_xyn, conf)
            else:
                # Convert every YOLO person to pose_utils.Pose, then keep the sustained dancer
                # identity through brief occlusion instead of switching to a passing person.
                boxes = result.boxes.xyxyn.cpu().numpy() if result.boxes is not None and result.boxes.xyxyn is not None else None
                box_scores = result.boxes.conf.cpu().numpy() if result.boxes is not None and result.boxes.conf is not None else None
                candidates = []
                for index, points in enumerate(keypoints_xyn):
                    scores = conf[index] if conf is not None else np.ones(17, dtype=np.float32)
                    valid = scores > .25
                    if boxes is not None and index < len(boxes):
                        box = tuple(float(value) for value in boxes[index])
                    elif valid.any():
                        low, high = points[valid].min(axis=0), points[valid].max(axis=0)
                        box = (float(low[0]), float(low[1]), float(high[0]), float(high[1]))
                    else:
                        continue
                    candidates.append(Pose(np.asarray(points, dtype=np.float32), np.asarray(scores, dtype=np.float32), box,
                                           float(box_scores[index]) if box_scores is not None and index < len(box_scores) else float(np.mean(scores)),
                                           aspect_ratio=width / max(height, 1)))
                locked = self.main_dancer_tracker.select(candidates, timestamp=time.monotonic())
                selected_pose = locked.points if locked is not None else None
                selected_conf = locked.confidence if locked is not None else None
            if selected_pose is None:
                continue
            keypoints_px = [(int(x * width), int(y * height)) for x, y in selected_pose]
            for pt1, pt2 in skeleton:
                x1, y1 = keypoints_px[pt1]
                x2, y2 = keypoints_px[pt2]
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
            for index, (x, y) in enumerate(keypoints_px):
                if x > 0 and y > 0:
                    color = (0, 0, 255) if index in [9, 10] else (255, 80, 30)
                    cv2.circle(overlay, (x, y), 5, color, -1)
            break
        return overlay, selected_pose, selected_conf

    def draw_pose_overlay(self, overlay, pose, conf):
        """Draw a previously inferred skeleton on a newer display frame."""
        if pose is None:
            return overlay
        height, width = overlay.shape[:2]
        keypoints_px = [(int(x * width), int(y * height)) for x, y in pose]
        for pt1, pt2 in skeleton:
            if conf is not None and (conf[pt1] < 0.25 or conf[pt2] < 0.25):
                continue
            x1, y1 = keypoints_px[pt1]
            x2, y2 = keypoints_px[pt2]
            if min(x1, y1, x2, y2) > 0:
                cv2.line(overlay, (x1, y1), (x2, y2), (101, 214, 142), 3, cv2.LINE_AA)
        for index, (x, y) in enumerate(keypoints_px):
            if x > 0 and y > 0 and (conf is None or conf[index] >= 0.25):
                cv2.circle(overlay, (x, y), 4, (95, 107, 255) if index in (9, 10) else (216, 197, 54), -1, cv2.LINE_AA)
        return overlay

    def update_fruits(self, frame, pose):
        now = time.monotonic()
        h, w = frame.shape[:2]
        if now - self.last_fruit_spawn > 1.6 and len(self.fruits) < 3:
            expected_side = None if DEMO_MODE in ("random_fruit", "all") else pose_side(self.next_reference_pose)
            # Most fruit supports the upcoming reference gesture. An object deliberately
            # placed on the opposite side becomes a bomb, teaching the player not to abandon
            # the dance direction merely to chase every moving target.
            fruit_side = expected_side if expected_side and random.random() < 0.75 else random.choice(("left", "right"))
            object_kind = "bomb" if expected_side is not None and fruit_side != expected_side else "fruit"
            x_range = (55, max(70, w // 2 - 35)) if fruit_side == "left" else (min(w - 70, w // 2 + 35), w - 55)
            x = random.randint(*x_range)
            y = random.randint(80, max(100, h - 100))
            colors = [(70, 95, 255), (70, 210, 125), (40, 185, 245)]
            self.fruits.append([x, y, 24, now, fruit_side, expected_side, random.choice(colors), object_kind])
            self.last_fruit_spawn = now

        wrists = []
        if pose is not None:
            for index in [9, 10]:
                x = int(pose[index, 0] * w)
                y = int(pose[index, 1] * h)
                if x > 0 and y > 0:
                    wrists.append((x, y))
                    self.slash_trail.append((x, y, now))

        # A tapered wrist trail makes the Fruit Ninja interaction legible without obscuring
        # the pose skeleton.
        trail = [(x, y, born) for x, y, born in self.slash_trail if now - born < 0.45]
        self.slash_trail = deque(trail, maxlen=10)
        for index in range(1, len(trail)):
            alpha = index / len(trail)
            cv2.line(frame, trail[index - 1][:2], trail[index][:2], (255, int(180 + 70 * alpha), 70), max(1, int(5 * alpha)), cv2.LINE_AA)

        remaining = []
        for x, y, radius, born, fruit_side, expected_side, color, object_kind in self.fruits:
            hit = any(np.hypot(wx - x, wy - y) < radius + 28 for wx, wy in wrists)
            expired = now - born > 5.0
            if hit:
                with self.lock:
                    if object_kind == "bomb":
                        # A bomb costs more when it breaks a strong combo, but the capped
                        # penalty and zero floor keep one mistake from ruining the whole run.
                        penalty = min(50, 25 + min(self.fruit_combo, 5) * 5)
                        self.fruit_points = max(0.0, self.fruit_points - penalty)
                        self.fruit_combo = 0
                        self.last_fruit_hit = 0.0
                        gained = -penalty
                    else:
                        self.fruit_combo = self.fruit_combo + 1 if now - self.last_fruit_hit <= 2.2 else 1
                        multiplier = 1.0 + min(self.fruit_combo - 1, 8) * 0.30
                        gained = int(round(20 * multiplier))
                        self.fruit_points += gained
                        self.last_fruit_hit = now
                    self.bonus_points = int(self.fruit_points)
                result_text = f"BOOM {gained}" if object_kind == "bomb" else f"+{gained}  x{self.fruit_combo}"
                result_color = (40, 60, 255) if object_kind == "bomb" else (70, 240, 255)
                cv2.putText(frame, result_text, (x - 48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, result_color, 2, cv2.LINE_AA)
            elif not expired:
                remaining.append([x, y, radius, born, fruit_side, expected_side, color, object_kind])
                if object_kind == "bomb":
                    cv2.circle(frame, (x, y), radius, (26, 29, 34), -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), radius, (110, 118, 128), 3, cv2.LINE_AA)
                    cv2.line(frame, (x - 9, y - 9), (x + 9, y + 9), (45, 70, 255), 4, cv2.LINE_AA)
                    cv2.line(frame, (x + 9, y - 9), (x - 9, y + 9), (45, 70, 255), 4, cv2.LINE_AA)
                    cv2.line(frame, (x + 10, y - radius + 4), (x + 18, y - radius - 10), (75, 120, 190), 3, cv2.LINE_AA)
                    spark = (x + 20, y - radius - 13)
                    cv2.circle(frame, spark, 5, (30, 210, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, spark, 2, (230, 250, 255), -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), radius, (245, 250, 255), 2, cv2.LINE_AA)
                    cv2.ellipse(frame, (x + 6, y - radius + 3), (8, 4), -25, 0, 360, (70, 205, 105), -1, cv2.LINE_AA)
                    cv2.line(frame, (x - radius + 5, y + radius - 5), (x + radius - 5, y - radius + 5), (255, 255, 255), 2, cv2.LINE_AA)
            else:
                with self.lock:
                    self.fruit_combo = 0
        self.fruits = remaining

    def draw_scoreboard(self, frame):
        with self.lock:
            score = self.latest_score
            avg = self.total_score / self.score_frames if self.score_frames else 0.0
            combo = self.combo
            bonus = self.bonus_points
        text, color = feedback_from_score(score)
        cv2.rectangle(frame, (12, 12), (360, 126), (15, 15, 15), -1)
        cv2.putText(frame, text, (26, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.05, color, 3, cv2.LINE_AA)
        cv2.putText(frame, f"Frame score: {0 if score is None else score:5.1f}", (26, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Average: {avg:5.1f}   Combo: {combo}   Bonus: {bonus}", (26, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    def frame_to_photo(self, frame, width, height):
        """Letterbox one BGR frame into a stable-size Tk photo."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (8, 11, 14))
        canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
        return ImageTk.PhotoImage(image=canvas)

    def _draw_pose_card(self, pose, x_center, color, footer):
        """Draw one compact pose pictogram without coupling it to scheduling semantics."""
        xs, ys = pose[:, 0], pose[:, 1]
        valid = (xs > 0) & (ys > 0)
        if valid.sum() < 5:
            return
        min_x, max_x = float(xs[valid].min()), float(xs[valid].max())
        min_y, max_y = float(ys[valid].min()), float(ys[valid].max())
        scale = min(68 / max(max_x - min_x, 1e-3), 54 / max(max_y - min_y, 1e-3))
        points = [((x - (min_x + max_x) / 2) * scale + x_center, (y - min_y) * scale + 11) for x, y in pose]
        self.preview_canvas.create_rectangle(x_center - 39, 7, x_center + 39, 72, outline=color, width=3)
        for first, second in skeleton:
            x1, y1 = points[first]
            x2, y2 = points[second]
            self.preview_canvas.create_line(x1, y1, x2, y2, fill=color, width=2)
        for point_index in COMPARE_POINTS:
            x, y = points[point_index]
            self.preview_canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#f4f7fa", outline="")
        self.preview_canvas.create_text(x_center, 79, text=footer, fill=color, font=("Segoe UI Semibold", 7))

    def draw_future_timeline(self, current_entry, future_entries, now):
        """Show the currently scored pose in orange and later preview poses in purple."""
        self.preview_canvas.delete("all")
        canvas_width = int(self.preview_canvas.winfo_width() or 560)
        self.preview_canvas.create_text(48, 4, text="CURRENT", fill="#ff9f43", anchor="sw", font=("Segoe UI Semibold", 7))
        if current_entry is not None:
            self._draw_pose_card(current_entry[1], 88, "#ff9f43", "MATCH NOW")
        else:
            self.preview_canvas.create_rectangle(49, 7, 127, 72, outline="#5b4a38", width=2)
            self.preview_canvas.create_text(88, 40, text="WAIT", fill="#8b745c", font=("Segoe UI", 7))

        self.preview_canvas.create_line(145, 7, 145, 74, fill="#39434f", width=1)
        self.preview_canvas.create_text(158, 4, text="UP NEXT", fill="#7d8cff", anchor="sw", font=("Segoe UI Semibold", 7))
        if not future_entries:
            self.preview_canvas.create_text(canvas_width // 2, 42, text="WAITING FOR REFERENCE KEYFRAMES", fill=MUTED, font=("Segoe UI", 8))
            return
        for index, entry in enumerate(future_entries[:4]):
            due, pose, _conf, _is_keyframe, _keyframe_id, _activity = entry
            x_center = 195 + index * 94
            remaining = max(0.0, due - now)
            self._draw_pose_card(pose, x_center, "#7d8cff", f"{remaining:.1f}s")

    def render_latest_frames(self):
        """Render at 30 Hz on Tk's main thread; workers only publish numpy arrays."""
        with self.lock:
            ref_frame, ref_version = self.latest_ref_frame, self.ref_frame_version
            cam_frame, cam_version = self.latest_cam_frame, self.cam_frame_version
            score = self.latest_score
            dance_points, fruit_points = self.dance_points, self.fruit_points
            dance_combo, fruit_combo = self.combo, self.fruit_combo
            now = time.monotonic()
            all_keyframes = list(self.keyframe_buffer)
            # The orange card represents the action being scored now. Purple cards retain the
            # original 2.5-second advance warning and never occupy the current-action slot.
            current_candidates = [entry for entry in all_keyframes if -0.55 <= entry[0] - now <= 0.15]
            current_entry = min(current_candidates, key=lambda entry: abs(entry[0] - now)) if current_candidates else None
            future_entries = [entry for entry in all_keyframes if entry[0] - now > 0.15]
            future_entries.sort(key=lambda entry: entry[0])
            due_times = [entry[0] for entry in future_entries]
            effect_is_active = bool(self.effect_text) and now - self.effect_started < 1.15
            effect_text, effect_color = self.effect_text, self.effect_color
        if ref_frame is not None and ref_version != self.rendered_ref_version:
            width, height = max(320, self.label_file.winfo_width()), max(220, self.label_file.winfo_height())
            self.ref_photo = self.frame_to_photo(ref_frame, width, height)
            self.label_file.configure(image=self.ref_photo)
            self.rendered_ref_version = ref_version
        if cam_frame is not None and cam_version != self.rendered_cam_version:
            width, height = max(320, self.label_cam.winfo_width()), max(220, self.label_cam.winfo_height())
            self.cam_photo = self.frame_to_photo(cam_frame, width, height)
            self.label_cam.configure(image=self.cam_photo)
            self.rendered_cam_version = cam_version
        # Keep the same encouragement word visible in the top bar for the duration of the
        # webcam burst. Between keyframes, fall back to the continuous pose feedback.
        if effect_is_active:
            feedback, color_bgr = effect_text, effect_color
        else:
            feedback, color_bgr = feedback_from_score(score)
        # Both feedback helpers return OpenCV BGR tuples; convert to a Tk hex colour.
        b, g, r = color_bgr
        self.feedback_label.configure(fg=f"#{r:02x}{g:02x}{b:02x}")
        self.feedback_var.set(feedback)
        score_text = "--" if score is None else f"{score:.1f}"
        self.metrics_var.set(
            f"Dance {dance_points:.0f}  Combo x{dance_combo}    "
            f"Fruit {fruit_points:.0f}  Fruit Combo x{fruit_combo}"
        )
        if due_times:
            lead = max(0.0, min(due_times) - time.monotonic())
            label = "KEYFRAME"
            self.prediction_var.set(f"NEXT {label}  •  IN {lead:.1f}s  •  MATCH THE HIGHLIGHTED ACTION")
        else:
            self.prediction_var.set("NEXT MOVE  •  START THE REFERENCE VIDEO TO BUILD THE 2.5s PREVIEW")
        self.draw_future_timeline(current_entry, future_entries, now)
        if not self.stop_event.is_set():
            self.root.after(33, self.render_latest_frames)

    def close(self):
        self.stop_event.set()
        self.running_file = self.running_cam = False
        self.stop_video_audio()
        if self.cap_file:
            self.cap_file.release()
        if self.cap_cam:
            self.cap_cam.release()
        self.root.after(80, self.root.destroy)


def run_demo(mode="all"):
    global DEMO_MODE
    DEMO_MODE = mode
    root = tk.Tk()
    app = PoseApp(root)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one deliberate legacy regression for recording.")
    parser.add_argument("--mode", choices=("coordinate", "lock", "continuous", "random_fruit", "fixed_interval", "all"), default="all")
    run_demo(parser.parse_args().mode)
