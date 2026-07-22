from __future__ import annotations

"""Retained experimental dance application variant; see danceapp.py for the main pipeline."""

import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".ultralytics"))

model = YOLO("yolov8n-pose.pt")

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


def pose_similarity(ref_pose, ref_conf, cam_pose, cam_conf):
    ref_norm, ref_valid = normalize_pose(ref_pose, ref_conf)
    cam_norm, cam_valid = normalize_pose(cam_pose, cam_conf)
    if ref_norm is None or cam_norm is None:
        return None
    valid = ref_valid & cam_valid
    if valid.sum() < 4:
        return None
    distances = np.linalg.norm(ref_norm[valid] - cam_norm[valid], axis=1)
    mean_distance = float(np.mean(distances))
    score = 100.0 * np.exp(-1.35 * mean_distance)
    return float(np.clip(score, 0.0, 100.0))


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


class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Just Dance Pose Scorer")
        self.root.geometry("1320x760")

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True

        self.reference_buffer = deque(maxlen=45)
        self.latest_cam_pose = None
        self.latest_cam_conf = None
        self.latest_score = None
        self.total_score = 0.0
        self.score_frames = 0
        self.combo = 0
        self.bonus_points = 0
        self.fruits = []
        self.last_fruit_spawn = time.time()
        self.lock = threading.Lock()

        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=8)
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=8)

        tk.Label(self.left_frame, text="Reference Video").pack()
        self.label_file = tk.Label(self.left_frame)
        self.label_file.pack()
        self.controls_file = tk.Frame(self.left_frame)
        self.controls_file.pack(pady=4)
        tk.Button(self.controls_file, text="Open Video", command=self.load_video).pack(side=tk.LEFT, padx=4)
        tk.Button(self.controls_file, text="Start Video", command=self.start_video).pack(side=tk.LEFT, padx=4)
        tk.Button(self.controls_file, text="Stop Video", command=self.stop_video).pack(side=tk.LEFT, padx=4)
        tk.Button(self.controls_file, text="Show/Hide Video", command=self.toggle_video_display).pack(side=tk.LEFT, padx=4)

        tk.Label(self.right_frame, text="Webcam Player").pack()
        self.label_cam = tk.Label(self.right_frame)
        self.label_cam.pack()
        self.controls_cam = tk.Frame(self.right_frame)
        self.controls_cam.pack(pady=4)
        tk.Button(self.controls_cam, text="Start Webcam", command=self.start_cam).pack(side=tk.LEFT, padx=4)
        tk.Button(self.controls_cam, text="Stop Webcam", command=self.stop_cam).pack(side=tk.LEFT, padx=4)
        tk.Button(self.controls_cam, text="Reset Score", command=self.reset_score).pack(side=tk.LEFT, padx=4)

    def load_video(self):
        path = filedialog.askopenfilename(
            initialdir=str(BASE_DIR),
            filetypes=[("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*")],
        )
        if path:
            self.video_path = path
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
            self.running_file = True
            threading.Thread(target=self.process_video_file, daemon=True).start()

    def stop_video(self):
        self.running_file = False
        if self.cap_file:
            self.cap_file.release()

    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame

    def start_cam(self):
        if not self.running_cam:
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
            self.bonus_points = 0
            self.fruits.clear()

    def process_video_file(self):
        self.cap_file = cv2.VideoCapture(self.video_path)
        while self.cap_file.isOpened() and self.running_file:
            ret, frame = self.cap_file.read()
            if not ret:
                self.cap_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame, pose, conf = self.process_pose(frame, is_reference=True)
            if pose is not None:
                with self.lock:
                    self.reference_buffer.append((time.time(), pose, conf))
            self.update_label(self.label_file, frame)
        self.cap_file.release()

    def process_webcam(self):
        self.cap_cam = cv2.VideoCapture(0)
        while self.cap_cam.isOpened() and self.running_cam:
            ret, frame = self.cap_cam.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame, pose, conf = self.process_pose(frame, is_reference=False)
            with self.lock:
                self.latest_cam_pose = pose
                self.latest_cam_conf = conf
                score = self.best_lag_score(pose, conf)
                self.latest_score = score
                if score is not None:
                    self.total_score += score
                    self.score_frames += 1
                    self.combo = self.combo + 1 if score >= 70 else 0
            self.update_fruits(frame, pose)
            self.draw_scoreboard(frame)
            self.update_label(self.label_cam, frame)
        self.cap_cam.release()

    def best_lag_score(self, cam_pose, cam_conf):
        if cam_pose is None:
            return None
        scores = []
        for _timestamp, ref_pose, ref_conf in list(self.reference_buffer):
            score = pose_similarity(ref_pose, ref_conf, cam_pose, cam_conf)
            if score is not None:
                scores.append(score)
        return max(scores) if scores else None

    def process_pose(self, frame, is_reference: bool):
        results = model(frame, conf=0.3, verbose=False)
        height, width = frame.shape[:2]
        overlay = frame.copy() if self.show_video_frame or not is_reference else np.ones_like(frame) * 255

        selected_pose = None
        selected_conf = None
        for result in results:
            if result.keypoints is None:
                continue
            keypoints_xyn = result.keypoints.xyn.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            selected_pose, selected_conf = select_main_person(keypoints_xyn, conf)
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

    def update_fruits(self, frame, pose):
        now = time.time()
        h, w = frame.shape[:2]
        if now - self.last_fruit_spawn > 1.8 and len(self.fruits) < 4:
            self.fruits.append([random.randint(60, w - 60), random.randint(80, h - 90), 24, now])
            self.last_fruit_spawn = now

        wrists = []
        if pose is not None:
            for index in [9, 10]:
                x = int(pose[index, 0] * w)
                y = int(pose[index, 1] * h)
                if x > 0 and y > 0:
                    wrists.append((x, y))

        remaining = []
        for x, y, radius, born in self.fruits:
            hit = any(np.hypot(wx - x, wy - y) < radius + 28 for wx, wy in wrists)
            expired = now - born > 5.0
            if hit:
                with self.lock:
                    self.bonus_points += 25
                cv2.putText(frame, "+25", (x - 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif not expired:
                remaining.append([x, y, radius, born])
                cv2.circle(frame, (x, y), radius, (0, 170, 255), -1)
                cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
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

    def update_label(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((640, 420))
        canvas = Image.new("RGB", (640, 420), (20, 20, 20))
        canvas.paste(img, ((640 - img.width) // 2, (420 - img.height) // 2))
        imgtk = ImageTk.PhotoImage(image=canvas)
        label.imgtk = imgtk
        label.configure(image=imgtk)


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseApp(root)
    root.mainloop()
