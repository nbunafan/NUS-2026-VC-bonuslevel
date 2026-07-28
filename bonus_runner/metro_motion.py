from __future__ import annotations

"""Split-screen pose-controlled three-lane runner for the Bonus Level demo."""

import argparse
import math
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pygame

from pose_controller import GestureController


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
ASSET_DIR = BASE_DIR / "assets"
MODEL_PATH = PROJECT_DIR / "yolov8n-pose.pt"
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".ultralytics"))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pose_utils import MainDancerTracker, Pose

SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def parse_camera_source(value: str):
    """Accept a local camera number or a phone stream URL.

    Common phone apps expose URLs such as http://PHONE_IP:8080/video. DroidCam
    can instead appear as a Windows virtual camera, in which case use 1 or 2.
    """
    return int(value) if value.strip().isdigit() else value.strip()


def select_main_person(keypoints: np.ndarray, confidence: np.ndarray):
    best = None
    best_score = -1.0
    for points, conf in zip(keypoints, confidence):
        visible = (points[:, 0] > 0) & (points[:, 1] > 0) & (conf > 0.25)
        if visible.sum() < 8:
            continue
        span = points[visible].max(axis=0) - points[visible].min(axis=0)
        score = float(span[0] * span[1] * conf[visible].mean())
        if score > best_score:
            best_score = score
            best = (points, conf)
    return best


class PoseCameraWorker:
    """Run camera capture and YOLO inference away from the 60 FPS game loop."""

    def __init__(self, source, mirror: bool, image_size: int = 320):
        self.source = source
        self.mirror = mirror
        self.image_size = image_size
        self.actions: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.latest_frame: np.ndarray | None = None
        self.latest_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.status = "STARTING CAMERA"
        self.gesture = GestureController()
        self.dancer_tracker = MainDancerTracker()
        self.running = False
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.inference_fps = 0.0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def reset_calibration(self) -> None:
        self.gesture.reset_calibration()
        self.dancer_tracker.reset()

    def snapshot(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            pose = None if self.latest_pose is None else (self.latest_pose[0].copy(), self.latest_pose[1].copy())
            return frame, self.status, self.inference_fps, self.gesture._progress(), pose

    def _open_capture(self):
        """Open a camera with bounded waits and Windows backend fallback.

        DirectShow can stop working when a virtual-camera driver is installed or restarted.
        Trying MSMF first and CAP_ANY last lets OpenCV recover without changing the game
        command. Network/phone streams use CAP_ANY directly.
        """
        if not isinstance(self.source, int):
            backends = [("STREAM", cv2.CAP_ANY)]
        elif os.name == "nt":
            backends = [
                ("DSHOW", cv2.CAP_DSHOW),
                ("MSMF", cv2.CAP_MSMF),
                ("AUTO", cv2.CAP_ANY),
            ]
        else:
            backends = [("AUTO", cv2.CAP_ANY)]

        # OpenCV's MSMF and DirectShow index backends reject timeout properties 53/54 during
        # construction. They are valid for network streams only; passing them to camera 0/1/2
        # prevents the backend from reaching the device at all.
        timeout_params = []
        if not isinstance(self.source, int):
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                timeout_params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3500])
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                timeout_params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3500])

        for backend_name, backend in backends:
            with self.lock:
                self.status = f"OPENING CAMERA {self.source} / {backend_name}"
            try:
                if timeout_params:
                    capture = cv2.VideoCapture(self.source, backend, timeout_params)
                else:
                    capture = cv2.VideoCapture(self.source, backend)
            except (cv2.error, TypeError):
                capture = cv2.VideoCapture(self.source, backend)
            if capture.isOpened():
                return capture, backend_name
            capture.release()
        return None, None

    def _run(self) -> None:
        # Import after YOLO_CONFIG_DIR is set so Ultralytics never writes to a protected
        # roaming directory on the classroom Windows setup.
        from ultralytics import YOLO

        model = YOLO(str(MODEL_PATH))
        capture, backend_name = self._open_capture()
        if capture is None:
            with self.lock:
                self.status = f"CAMERA {self.source} NOT AVAILABLE - TRY --camera 1"
            self.running = False
            return
        if isinstance(self.source, int):
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        with self.lock:
            self.status = f"CAMERA {self.source} CONNECTED / {backend_name}"

        previous = time.perf_counter()
        while self.running:
            ok, frame = capture.read()
            if not ok:
                with self.lock:
                    self.status = "CAMERA FRAME LOST"
                time.sleep(0.1)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)

            results = model.predict(frame, imgsz=self.image_size, conf=0.25, verbose=False)
            points = None
            confidence = None
            if results and results[0].keypoints is not None:
                result = results[0]
                keypoints = result.keypoints.xyn.cpu().numpy()
                confidence_tensor = result.keypoints.conf
                confidences = confidence_tensor.cpu().numpy() if confidence_tensor is not None else np.ones(keypoints.shape[:2], dtype=np.float32)
                boxes = result.boxes.xyxyn.cpu().numpy() if result.boxes is not None and result.boxes.xyxyn is not None else None
                box_scores = result.boxes.conf.cpu().numpy() if result.boxes is not None and result.boxes.conf is not None else None
                candidates = []
                for index, person_points in enumerate(keypoints):
                    visible = confidences[index] > 0.25
                    if boxes is not None and index < len(boxes):
                        box = tuple(float(value) for value in boxes[index])
                    elif visible.any():
                        low, high = person_points[visible].min(axis=0), person_points[visible].max(axis=0)
                        box = (float(low[0]), float(low[1]), float(high[0]), float(high[1]))
                    else:
                        continue
                    candidates.append(Pose(
                        points=np.asarray(person_points, dtype=np.float32),
                        confidence=np.asarray(confidences[index], dtype=np.float32),
                        box=box,
                        detection_confidence=float(box_scores[index]) if box_scores is not None and index < len(box_scores) else float(np.mean(confidences[index])),
                        aspect_ratio=frame.shape[1] / max(frame.shape[0], 1),
                    ))
                selected = self.dancer_tracker.select(candidates, timestamp=time.monotonic())
                if selected is not None:
                    points, confidence = selected.points, selected.confidence

            gesture = self.gesture.update(points, confidence)
            if gesture.action:
                self.actions.put(gesture.action)
            annotated = self._draw_pose(frame, points, confidence)
            self._draw_camera_status(annotated, gesture.state, gesture.calibration_progress)

            now = time.perf_counter()
            instantaneous = 1.0 / max(now - previous, 1e-6)
            previous = now
            with self.lock:
                self.inference_fps = 0.85 * self.inference_fps + 0.15 * instantaneous
                self.latest_frame = annotated
                self.latest_pose = None if points is None else (points.copy(), confidence.copy())
                self.status = f"{gesture.state} / {self.dancer_tracker.state}"
        capture.release()

    @staticmethod
    def _draw_pose(frame, points, confidence):
        output = frame.copy()
        if points is None or confidence is None:
            return output
        height, width = output.shape[:2]
        pixels = np.column_stack((points[:, 0] * width, points[:, 1] * height)).astype(int)
        for first, second in SKELETON:
            if confidence[first] > 0.25 and confidence[second] > 0.25:
                cv2.line(output, tuple(pixels[first]), tuple(pixels[second]), (65, 235, 170), 4, cv2.LINE_AA)
        for index, point in enumerate(pixels):
            if confidence[index] > 0.25:
                cv2.circle(output, tuple(point), 6, (40, 65, 245), -1, cv2.LINE_AA)
        return output

    @staticmethod
    def _draw_camera_status(frame, state: str, progress: float) -> None:
        cv2.rectangle(frame, (14, 14), (360, 82), (20, 23, 28), -1)
        cv2.putText(frame, state, (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        if progress < 1.0:
            cv2.rectangle(frame, (28, 60), (330, 72), (70, 76, 83), -1)
            cv2.rectangle(frame, (28, 60), (28 + int(302 * progress), 72), (48, 210, 155), -1)


@dataclass
class Obstacle:
    lane: int
    kind: str
    progress: float = 0.0
    resolved: bool = False
    checked: bool = False
    vanish_remaining: float = 0.0
    vanish_duration: float = 0.0
    spin_offset: float = field(default_factory=lambda: random.random() * math.tau)

    def resolve(self, duration: float) -> None:
        """Start a short visual confirmation instead of leaving a collected object in lane."""
        self.resolved = True
        self.vanish_duration = duration
        self.vanish_remaining = duration


@dataclass(frozen=True)
class GameEvent:
    """A renderer-agnostic gameplay event consumed by the Pygame effects layer."""

    kind: str
    lane: int = 1
    label: str = ""
    color: tuple[int, int, int] = (255, 255, 255)


class RunnerGame:
    MISSION_DEFINITIONS = (
        ("COLLECT 5 PICKUPS", "pickup", 5.0, 120),
        ("CLEAR 3 BARRIERS", "dodge", 3.0, 180),
        ("RUN 250 METRES", "distance", 250.0, 240),
    )

    def __init__(self):
        self.lane = 1
        self.target_lane = 1.0
        self.visual_lane = 1.0
        self.jump_remaining = 0.0
        self.obstacles: list[Obstacle] = []
        self.spawn_timer = 1.2
        self.score = 0.0
        self.coins = 0
        self.orange_coins_collected = 0
        self.blue_coins_collected = 0
        self.nus_milestone_unlocked = False
        self.nus_animation_duration = 2.8
        self.nus_animation_remaining = 0.0
        self.lives = 3
        self.speed = 0.26
        self.distance = 0.0
        self.run_time = 0.0
        self.combo = 0
        self.max_combo = 0
        self.combo_timer = 0.0
        self.invincible_remaining = 0.0
        self.successful_dodges = 0
        self.pickups_collected = 0
        self.speed_stage = 0
        self.mission_index = 0
        self.mission_progress = 0.0
        self.events: list[GameEvent] = []
        self.game_over = False
        self.last_action = "READY"
        self.action_flash = 0.0
        self.started = False
        self.paused = False

    def reset(self) -> None:
        self.__init__()

    def start(self) -> None:
        """Start spawning and scoring only after the player presses START RUN."""
        self.started = True
        self.paused = False
        self.last_action = "GO!"
        self.action_flash = 0.8

    def toggle_pause(self) -> None:
        if self.started and not self.game_over and self.nus_animation_remaining <= 0:
            self.paused = not self.paused

    @property
    def combo_multiplier(self) -> int:
        """Stepwise multiplier keeps the scoring easy to explain during a demo."""
        if self.combo >= 20:
            return 4
        if self.combo >= 10:
            return 3
        if self.combo >= 5:
            return 2
        return 1

    @property
    def mission(self) -> tuple[str, str, float, int]:
        return self.MISSION_DEFINITIONS[self.mission_index % len(self.MISSION_DEFINITIONS)]

    @property
    def mission_ratio(self) -> float:
        return min(1.0, self.mission_progress / max(self.mission[2], 1.0))

    @property
    def speed_label(self) -> str:
        return ("CRUISE", "FLOW", "RUSH", "HYPER")[min(self.speed_stage, 3)]

    def pop_events(self) -> list[GameEvent]:
        events, self.events = self.events, []
        return events

    def command(self, action: str) -> None:
        if self.game_over or not self.started or self.paused or self.nus_animation_remaining > 0:
            return
        accepted = False
        if action == "LEFT" and self.lane > 0:
            self.lane -= 1
            self.target_lane = float(self.lane)
            accepted = True
        elif action == "RIGHT" and self.lane < 2:
            self.lane += 1
            self.target_lane = float(self.lane)
            accepted = True
        elif action == "JUMP" and self.jump_remaining <= 0:
            self.jump_remaining = 0.86
            accepted = True
        if not accepted:
            return
        self.last_action = action
        self.action_flash = 0.40

    def jump_height_factor(self) -> float:
        """Return 0 on the ground and 1 near the apex of the current jump."""
        if self.jump_remaining <= 0:
            return 0.0
        phase = 1.0 - self.jump_remaining / 0.86
        return max(0.0, math.sin(math.pi * phase))

    def _register_success(
        self,
        label: str,
        base_points: int,
        event_kind: str,
        lane: int,
        color: tuple[int, int, int],
    ) -> None:
        """Award one discrete event, extend the combo and emit visual feedback."""
        self.combo += 1
        self.max_combo = max(self.max_combo, self.combo)
        self.combo_timer = 4.0
        awarded = base_points * self.combo_multiplier
        self.score += awarded
        self.last_action = f"{label} +{awarded}"
        self.action_flash = 0.72
        self.events.append(GameEvent(event_kind, lane, self.last_action, color))

    def _advance_mission(self, event_kind: str, amount: float = 1.0) -> None:
        label, required_kind, goal, reward = self.mission
        if required_kind != event_kind:
            return
        self.mission_progress = min(goal, self.mission_progress + amount)
        if self.mission_progress < goal:
            return
        self.score += reward
        self.events.append(GameEvent("mission", self.lane, f"MISSION +{reward}", (255, 214, 92)))
        self.last_action = f"MISSION COMPLETE +{reward}"
        self.action_flash = 1.1
        self.mission_index = (self.mission_index + 1) % len(self.MISSION_DEFINITIONS)
        self.mission_progress = 0.0

    def _collect_coin(self, obstacle: Obstacle, multiplier: int) -> None:
        self.coins += multiplier
        self.pickups_collected += 1
        if obstacle.kind == "air_coin":
            self.blue_coins_collected += 1
        else:
            self.orange_coins_collected += 1
        obstacle.checked = True
        obstacle.resolve(0.16)
        if multiplier > 1:
            self._register_success("BLUE COIN", 75, "blue_coin", obstacle.lane, (73, 196, 255))
        else:
            self._register_success("COIN", 25, "coin", obstacle.lane, (255, 178, 52))
        self._advance_mission("pickup")
        if (
            not self.nus_milestone_unlocked
            and self.orange_coins_collected >= 3
            and self.blue_coins_collected >= 3
        ):
            self.nus_milestone_unlocked = True
            self.nus_animation_remaining = self.nus_animation_duration
            self.events.append(GameEvent("nus", self.lane, "NUS UNLOCKED", (255, 255, 255)))

    def _spawn_wave(self) -> None:
        """Spawn readable patterns while always keeping at least one safe lane."""
        difficulty = min(1.0, self.distance / 900.0)
        kind = random.choices(
            ["barrier", "coin", "air_coin"],
            weights=[0.50 + 0.12 * difficulty, 0.34 - 0.06 * difficulty, 0.16],
        )[0]
        primary_lane = random.randrange(3)
        self.obstacles.append(Obstacle(primary_lane, kind))

        # Later runs sometimes show a second, complementary object. Two barriers can never
        # occupy all three lanes, so every wave remains physically solvable.
        if self.distance > 140 and random.random() < 0.18 + 0.16 * difficulty:
            other_lanes = [lane for lane in range(3) if lane != primary_lane]
            secondary_lane = random.choice(other_lanes)
            if kind == "barrier":
                secondary_kind = random.choice(("coin", "air_coin"))
            else:
                secondary_kind = random.choices(("barrier", "coin"), weights=(0.65, 0.35))[0]
            self.obstacles.append(Obstacle(secondary_lane, secondary_kind, progress=-0.04))

    def update(self, dt: float) -> None:
        if self.game_over or not self.started or self.paused:
            return
        if self.nus_animation_remaining > 0:
            # Freeze hazards and commands during the short milestone cutscene while camera
            # inference continues in its worker thread.
            self.nus_animation_remaining = max(0.0, self.nus_animation_remaining - dt)
            self.action_flash = 0.0
            return
        self.run_time += dt
        distance_delta = dt * (18.0 + (self.speed - 0.26) * 72.0)
        self.distance += distance_delta
        self.score += dt * 10.0
        self.speed = min(0.46, 0.26 + self.distance / 4200.0)
        new_stage = min(3, int(self.distance // 180.0))
        if new_stage > self.speed_stage:
            self.speed_stage = new_stage
            self.events.append(GameEvent("speed", self.lane, f"{self.speed_label} MODE", (255, 95, 104)))
            self.last_action = f"{self.speed_label} MODE"
            self.action_flash = 1.0
        self._advance_mission("distance", distance_delta)
        self.jump_remaining = max(0.0, self.jump_remaining - dt)
        self.action_flash = max(0.0, self.action_flash - dt)
        self.invincible_remaining = max(0.0, self.invincible_remaining - dt)
        if self.combo_timer > 0:
            self.combo_timer = max(0.0, self.combo_timer - dt)
            if self.combo_timer <= 0:
                self.combo = 0
        self.visual_lane += (self.target_lane - self.visual_lane) * min(1.0, dt * 12.0)

        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self._spawn_wave()
            speed_ratio = self.speed / 0.26
            self.spawn_timer = max(0.66, random.uniform(1.08, 1.62) / speed_ratio)

        remaining = []
        for obstacle in self.obstacles:
            obstacle.progress += dt * self.speed
            if obstacle.kind == "air_coin" and not obstacle.checked:
                # Keep a wider collision window than ground objects because camera gesture
                # inference arrives less frequently than the 60 FPS rendering loop.
                in_pickup_window = 0.84 <= obstacle.progress <= 1.04
                if (
                    obstacle.lane == self.lane
                    and in_pickup_window
                    and self.jump_height_factor() >= 0.45
                ):
                    self._collect_coin(obstacle, multiplier=3)
                elif obstacle.progress > 1.04:
                    obstacle.checked = True
            elif not obstacle.checked and obstacle.progress >= 0.90:
                obstacle.checked = True
                if obstacle.lane == self.lane:
                    if obstacle.kind == "coin":
                        self._collect_coin(obstacle, multiplier=1)
                    elif obstacle.kind == "barrier" and self.jump_remaining <= 0.18:
                        self._hit()
                        obstacle.resolve(0.20)
                    elif obstacle.kind == "barrier":
                        # A correctly timed two-hand jump earns a discrete dodge reward.
                        obstacle.resolve(0.20)
                        self.successful_dodges += 1
                        self._register_success(
                            "PERFECT DODGE", 45, "dodge", obstacle.lane, (85, 238, 181)
                        )
                        self._advance_mission("dodge")
            if obstacle.resolved:
                obstacle.vanish_remaining = max(0.0, obstacle.vanish_remaining - dt)
                if obstacle.vanish_remaining > 0.0:
                    remaining.append(obstacle)
            elif obstacle.progress < 1.12:
                remaining.append(obstacle)
        self.obstacles = remaining

    def _hit(self) -> None:
        if self.invincible_remaining > 0:
            return
        self.lives -= 1
        self.score = max(0.0, self.score - 40)
        self.combo = 0
        self.combo_timer = 0.0
        self.invincible_remaining = 1.15
        self.last_action = "IMPACT -40"
        self.action_flash = 0.8
        self.events.append(GameEvent("hit", self.lane, "IMPACT -40", (255, 88, 92)))
        if self.lives <= 0:
            self.game_over = True


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: tuple[int, int, int]
    radius: float
    gravity: float = 0.0

    def update(self, dt: float) -> bool:
        self.life -= dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += self.gravity * dt
        return self.life > 0


@dataclass
class FloatingLabel:
    text: str
    x: float
    y: float
    color: tuple[int, int, int]
    life: float = 0.9
    max_life: float = 0.9

    def update(self, dt: float) -> bool:
        self.life -= dt
        self.y -= 54.0 * dt
        return self.life > 0


class ProceduralSoundBank:
    """Small generated sound effects; audio failure never prevents the game starting."""

    def __init__(self):
        self.enabled = False
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.sounds = {
                "coin": self._tone((660, 880), 0.13, 0.20),
                "blue_coin": self._tone((520, 780, 1040), 0.19, 0.22),
                "dodge": self._tone((380, 620), 0.16, 0.18),
                "hit": self._tone((170, 105), 0.22, 0.24),
                "mission": self._tone((440, 660, 880), 0.32, 0.20),
                "speed": self._tone((330, 495, 740), 0.24, 0.17),
                "nus": self._tone((392, 523, 659, 784), 0.45, 0.21),
            }
            self.enabled = True
        except (pygame.error, TypeError, ValueError):
            self.enabled = False

    @staticmethod
    def _tone(frequencies: tuple[int, ...], duration: float, volume: float) -> pygame.mixer.Sound:
        sample_rate = 44100
        sample_count = max(1, int(sample_rate * duration))
        t = np.arange(sample_count, dtype=np.float32) / sample_rate
        wave = np.zeros(sample_count, dtype=np.float32)
        segment = max(1, sample_count // len(frequencies))
        for index, frequency in enumerate(frequencies):
            start = index * segment
            end = sample_count if index == len(frequencies) - 1 else (index + 1) * segment
            local_t = t[start:end] - t[start]
            wave[start:end] = np.sin(math.tau * frequency * local_t)
        envelope = np.minimum(1.0, t * 45.0) * np.exp(-5.0 * t / max(duration, 0.01))
        mono = np.asarray(wave * envelope * volume * 32767, dtype=np.int16)
        stereo = np.column_stack((mono, mono))
        return pygame.sndarray.make_sound(stereo)

    def play(self, kind: str) -> None:
        sound = self.sounds.get(kind)
        if self.enabled and sound is not None:
            sound.play()

    def toggle(self) -> None:
        if self.sounds:
            self.enabled = not self.enabled


class MetroMotionApp:
    WIDTH, HEIGHT = 1440, 810
    GAME_WIDTH = 870

    def __init__(self, camera_source, mirror: bool, image_size: int, worker=None):
        pygame.init()
        pygame.display.set_caption("Metro Motion - Pose Controlled Runner")
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 23)
        self.small = pygame.font.SysFont("segoeui", 16)
        self.micro = pygame.font.SysFont("segoeui", 13, bold=True)
        self.hud_value = pygame.font.SysFont("segoeui", 30, bold=True)
        self.combo_font = pygame.font.SysFont("segoeui", 40, bold=True)
        self.large = pygame.font.SysFont("segoeui", 50, bold=True)
        self.hero = pygame.font.SysFont("segoeui", 66, bold=True)
        self.nus_font = pygame.font.SysFont("segoeui", 150, bold=True)
        self.milestone_font = pygame.font.SysFont("segoeui", 28, bold=True)
        self.game = RunnerGame()
        self.worker = worker if worker is not None else PoseCameraWorker(camera_source, mirror, image_size)
        self.worker.start()
        self.running = True
        self.assets = self._load_assets()
        self.sounds = ProceduralSoundBank()
        self.elapsed = 0.0
        self.particles: list[Particle] = []
        self.floating_labels: list[FloatingLabel] = []
        self.flash_color = (255, 255, 255)
        self.flash_alpha = 0.0
        self.screen_shake = 0.0
        self.background_cache: tuple[tuple[int, int], pygame.Surface] | None = None
        self.start_button_rect = pygame.Rect(0, 0, 0, 0)
        self.restart_button_rect = pygame.Rect(0, 0, 0, 0)
        self.pause_button_rect = pygame.Rect(0, 0, 0, 0)
        self.resume_button_rect = pygame.Rect(0, 0, 0, 0)
        self.current_game_panel = pygame.Rect(0, 0, self.GAME_WIDTH, self.HEIGHT)

    def _load_assets(self):
        return {
            name: pygame.image.load(str(ASSET_DIR / f"{name}.png")).convert_alpha()
            for name in ("runner", "barrier", "coin")
        } | {"skyline": pygame.image.load(str(ASSET_DIR / "skyline.png")).convert()}

    def run(self) -> None:
        while self.running:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            self.elapsed += dt
            self._events()
            while not self.worker.actions.empty():
                self.game.command(self.worker.actions.get())
            self.game.update(dt)
            self._consume_game_events()
            self._update_effects(dt)
            self._draw()
            pygame.display.flip()
        self.worker.stop()
        pygame.quit()

    def _events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                size = (max(1100, event.w), max(620, event.h))
                self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                actions = {
                    pygame.K_LEFT: "LEFT", pygame.K_a: "LEFT",
                    pygame.K_RIGHT: "RIGHT", pygame.K_d: "RIGHT",
                    pygame.K_UP: "JUMP", pygame.K_w: "JUMP", pygame.K_SPACE: "JUMP",
                }
                if event.key in actions:
                    self.game.command(actions[event.key])
                elif event.key == pygame.K_c:
                    self.worker.reset_calibration()
                elif event.key == pygame.K_r:
                    self.game.reset()
                    self.particles.clear()
                    self.floating_labels.clear()
                elif event.key == pygame.K_p:
                    self.game.toggle_pause()
                elif event.key == pygame.K_m:
                    self.sounds.toggle()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.start_button_rect.collidepoint(event.pos) and not self.game.started:
                    self.game.start()
                elif self.restart_button_rect.collidepoint(event.pos) and self.game.game_over:
                    self.game.reset()
                    self.game.start()
                    self.particles.clear()
                    self.floating_labels.clear()
                elif self.pause_button_rect.collidepoint(event.pos):
                    self.game.toggle_pause()
                elif self.resume_button_rect.collidepoint(event.pos) and self.game.paused:
                    self.game.toggle_pause()

    def _draw(self) -> None:
        width, height = self.screen.get_size()
        game_width = int(width * 0.615)
        self.current_game_panel = pygame.Rect(0, 0, game_width, height)
        self._draw_game(self.current_game_panel)
        self._draw_camera(pygame.Rect(game_width, 0, width - game_width, height))

    def _consume_game_events(self) -> None:
        panel = self.current_game_panel
        horizon = panel.top + int(panel.height * 0.27)
        road_top = panel.width * 0.105
        road_bottom = panel.width * 0.54
        for event in self.game.pop_events():
            x = self._lane_x(event.lane, 0.91, panel.centerx, road_top, road_bottom)
            y = horizon + (panel.bottom - horizon) * (0.91 ** 1.55) - 60
            count = 24 if event.kind in ("mission", "nus") else 15
            for index in range(count):
                angle = math.tau * index / count + random.uniform(-0.18, 0.18)
                force = random.uniform(80.0, 225.0)
                self.particles.append(Particle(
                    x=x,
                    y=y,
                    vx=math.cos(angle) * force,
                    vy=math.sin(angle) * force - 42,
                    life=random.uniform(0.42, 0.82),
                    max_life=0.82,
                    color=event.color,
                    radius=random.uniform(2.5, 6.0),
                    gravity=180.0,
                ))
            self.floating_labels.append(FloatingLabel(event.label, x, y - 16, event.color))
            self.sounds.play(event.kind)
            if event.kind == "hit":
                self.flash_color = event.color
                self.flash_alpha = 145.0
                self.screen_shake = 0.35
            elif event.kind in ("mission", "nus"):
                self.flash_color = event.color
                self.flash_alpha = 75.0

    def _update_effects(self, dt: float) -> None:
        self.particles = [particle for particle in self.particles if particle.update(dt)]
        self.floating_labels = [label for label in self.floating_labels if label.update(dt)]
        self.flash_alpha = max(0.0, self.flash_alpha - dt * 260.0)
        self.screen_shake = max(0.0, self.screen_shake - dt)

    @staticmethod
    def _mix_color(first, second, amount: float) -> tuple[int, int, int]:
        return tuple(int(a + (b - a) * amount) for a, b in zip(first, second))

    def _build_world_background(self, size: tuple[int, int]) -> pygame.Surface:
        """Cache the expensive skyline and sky gradient until the window is resized."""
        if self.background_cache is not None and self.background_cache[0] == size:
            return self.background_cache[1]
        width, height = size
        surface = pygame.Surface(size)
        horizon = int(height * 0.28)
        sky_top = (15, 31, 55)
        sky_bottom = (232, 106, 92)
        for y in range(max(1, horizon + 1)):
            ratio = y / max(horizon, 1)
            pygame.draw.line(surface, self._mix_color(sky_top, sky_bottom, ratio), (0, y), (width, y))
        surface.fill((21, 27, 35), (0, horizon, width, height - horizon))

        # Reuse the original, deterministic city artwork only for the skyline crop. The
        # gameplay track is still rendered live so sleepers and lights respond to distance.
        skyline = self.assets["skyline"]
        crop_height = max(1, int(skyline.height * 0.58))
        city_crop = skyline.subsurface((0, 0, skyline.width, crop_height))
        city = pygame.transform.smoothscale(city_crop, (width, horizon + 18))
        city.set_alpha(182)
        surface.blit(city, (0, 0))

        warmth = pygame.Surface((width, horizon + 1), pygame.SRCALPHA)
        for y in range(max(1, horizon + 1)):
            ratio = y / max(horizon, 1)
            pygame.draw.line(warmth, (255, 104, 78, int(58 * ratio)), (0, y), (width, y))
        surface.blit(warmth, (0, 0))

        # Warm horizon bands break up the night palette and make the track silhouette crisp.
        for index, color in enumerate(((255, 180, 106), (242, 118, 92), (74, 205, 199))):
            band_y = horizon - 9 + index * 6
            pygame.draw.rect(surface, color, (0, band_y, width, 6))

        self.background_cache = (size, surface)
        return surface

    def _draw_game(self, panel: pygame.Rect) -> None:
        self.screen.blit(self._build_world_background(panel.size), panel)
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(panel)
        horizon = panel.top + int(panel.height * 0.27)
        bottom = panel.bottom
        center = panel.centerx
        road_top = panel.width * 0.105
        road_bottom = panel.width * 0.54
        self._draw_track(panel, horizon, road_top, road_bottom)

        for obstacle in sorted(self.game.obstacles, key=lambda item: item.progress):
            self._draw_obstacle(panel, obstacle, horizon, road_top, road_bottom)
        self._draw_runner(panel, road_bottom)
        self._draw_effects(panel)
        self._draw_hud(panel)
        self._draw_nus_milestone(panel)
        if self.flash_alpha > 0:
            flash = pygame.Surface(panel.size, pygame.SRCALPHA)
            flash.fill((*self.flash_color, int(self.flash_alpha)))
            self.screen.blit(flash, panel)
        self.screen.set_clip(previous_clip)

    def _draw_track(self, panel: pygame.Rect, horizon: int, road_top: float, road_bottom: float) -> None:
        """Draw a moving, three-lane rail corridor using one consistent perspective."""
        center = panel.centerx
        bottom = panel.bottom
        left_edge_top, right_edge_top = center - road_top, center + road_top
        left_edge_bottom, right_edge_bottom = center - road_bottom, center + road_bottom

        pygame.draw.polygon(
            self.screen,
            (35, 39, 45),
            [(panel.left, horizon), (left_edge_top, horizon), (left_edge_bottom, bottom), (panel.left, bottom)],
        )
        pygame.draw.polygon(
            self.screen,
            (35, 39, 45),
            [(right_edge_top, horizon), (panel.right, horizon), (panel.right, bottom), (right_edge_bottom, bottom)],
        )
        pygame.draw.polygon(
            self.screen,
            (51, 50, 52),
            [(left_edge_top, horizon), (right_edge_top, horizon), (right_edge_bottom, bottom), (left_edge_bottom, bottom)],
        )

        # Slightly different lane tones improve readability without breaking the shared track.
        for lane, color in enumerate(((56, 55, 57), (49, 49, 52), (56, 55, 57))):
            far_left = self._lane_x(lane - 0.48, 0.0, center, road_top, road_bottom)
            far_right = self._lane_x(lane + 0.48, 0.0, center, road_top, road_bottom)
            near_left = self._lane_x(lane - 0.48, 1.0, center, road_top, road_bottom)
            near_right = self._lane_x(lane + 0.48, 1.0, center, road_top, road_bottom)
            pygame.draw.polygon(
                self.screen,
                color,
                [(far_left, horizon), (far_right, horizon), (near_right, bottom), (near_left, bottom)],
            )

        travel = (self.game.distance * 0.032) % 1.0
        sleeper_progress = sorted((index / 19.0 + travel) % 1.0 for index in range(19))
        for p in sleeper_progress:
            curve = p ** 1.55
            y = horizon + (bottom - horizon) * curve
            thickness = max(1, int(2 + 10 * p))
            for lane in range(3):
                lane_center = self._lane_x(lane, p, center, road_top, road_bottom)
                half_width = (road_top + (road_bottom - road_top) * p) * 0.25
                pygame.draw.line(
                    self.screen,
                    (91, 76, 68),
                    (lane_center - half_width, y),
                    (lane_center + half_width, y),
                    thickness,
                )

        # Each lane gets a true pair of rails rather than one generic divider line.
        for lane in range(3):
            for rail_offset in (-0.17, 0.17):
                far_x = self._lane_x(lane + rail_offset, 0.0, center, road_top, road_bottom)
                near_x = self._lane_x(lane + rail_offset, 1.0, center, road_top, road_bottom)
                pygame.draw.line(self.screen, (30, 31, 34), (far_x + 2, horizon), (near_x + 5, bottom), 9)
                pygame.draw.line(self.screen, (175, 181, 184), (far_x, horizon), (near_x, bottom), 4)
                pygame.draw.line(self.screen, (234, 217, 171), (far_x - 1, horizon), (near_x - 1, bottom), 1)

        # Amber/cyan edge beacons move towards the player and communicate the speed increase.
        beacon_progress = sorted((index / 11.0 + travel * 0.72) % 1.0 for index in range(11))
        for index, p in enumerate(beacon_progress):
            curve = p ** 1.55
            y = horizon + (bottom - horizon) * curve
            half_width = road_top + (road_bottom - road_top) * p
            size = max(2, int(3 + p * 8))
            color = (55, 206, 202) if index % 2 else (255, 165, 66)
            for side in (-1, 1):
                x = center + side * half_width * 1.035
                pygame.draw.circle(self.screen, (18, 23, 29), (int(x), int(y)), size + 3)
                pygame.draw.circle(self.screen, color, (int(x), int(y)), size)

        if self.game.speed_stage >= 1 and self.game.started and not self.game.paused:
            intensity = 8 + self.game.speed_stage * 5
            for index in range(intensity):
                phase = (index * 0.173 + self.elapsed * (0.8 + self.game.speed_stage * 0.25)) % 1.0
                y = horizon + phase * (bottom - horizon)
                side = -1 if index % 2 else 1
                x = center + side * (road_top + (road_bottom - road_top) * phase) * 1.18
                length = 12 + int(55 * phase)
                pygame.draw.line(self.screen, (244, 142, 104), (x, y), (x + side * length, y + length), 2)

    def _draw_effects(self, panel: pygame.Rect) -> None:
        layer = pygame.Surface(panel.size, pygame.SRCALPHA)
        for particle in self.particles:
            ratio = max(0.0, particle.life / max(particle.max_life, 1e-6))
            radius = max(1, int(particle.radius * (0.55 + ratio)))
            color = (*particle.color, int(255 * ratio))
            pygame.draw.circle(layer, color, (int(particle.x - panel.left), int(particle.y - panel.top)), radius)
        self.screen.blit(layer, panel)

        for label in self.floating_labels:
            ratio = max(0.0, label.life / max(label.max_life, 1e-6))
            rendered = self.small.render(label.text, True, label.color)
            rendered.set_alpha(int(255 * min(1.0, ratio * 1.7)))
            self.screen.blit(rendered, rendered.get_rect(center=(int(label.x), int(label.y))))

    def _lane_x(self, lane: float, progress: float, center: float, road_top: float, road_bottom: float) -> float:
        half_width = road_top + (road_bottom - road_top) * progress
        return center + (lane - 1.0) * half_width * 0.58

    def _draw_obstacle(self, panel, obstacle, horizon, road_top, road_bottom):
        p = max(0.02, min(1.0, obstacle.progress))
        y = horizon + (panel.bottom - horizon) * (p ** 1.55)
        x = self._lane_x(obstacle.lane, p, panel.centerx, road_top, road_bottom)
        scale = 0.18 + 0.82 * p
        alpha = 255
        if obstacle.resolved and obstacle.vanish_duration > 0:
            remaining_ratio = obstacle.vanish_remaining / obstacle.vanish_duration
            # Collected coins pop slightly while cleared barriers collapse, then disappear.
            scale *= 1.0 + (0.22 if obstacle.kind in ("coin", "air_coin") else 0.10) * (1.0 - remaining_ratio)
            alpha = max(0, min(255, int(255 * remaining_ratio)))
        image = self.assets["coin" if obstacle.kind == "air_coin" else obstacle.kind]
        target_width = max(20, int(image.get_width() * scale))
        target_height = max(20, int(image.get_height() * scale))
        if obstacle.kind in ("coin", "air_coin"):
            spin = 0.24 + 0.76 * abs(math.cos(self.elapsed * 6.8 + obstacle.spin_offset))
            target_width = max(8, int(target_width * spin))
        sprite = pygame.transform.smoothscale(image, (target_width, target_height))
        if obstacle.kind in ("coin", "air_coin"):
            glow_color = (60, 172, 255) if obstacle.kind == "air_coin" else (255, 163, 48)
            glow_size = max(target_width, target_height) + max(14, int(28 * scale))
            glow = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*glow_color, 35), glow.get_rect().center, glow_size // 2)
            pygame.draw.circle(glow, (*glow_color, 75), glow.get_rect().center, max(3, glow_size // 3), 2)
            glow_y = y - (int(panel.height * (0.04 + 0.12 * p)) if obstacle.kind == "air_coin" else 0)
            self.screen.blit(glow, glow.get_rect(center=(int(x), int(glow_y - target_height * 0.45))))
            if obstacle.kind == "air_coin":
                mask = pygame.mask.from_surface(sprite)
                sprite = mask.to_surface(setcolor=(48, 152, 255, 255), unsetcolor=(0, 0, 0, 0)).convert_alpha()
                pygame.draw.ellipse(
                    sprite,
                    (198, 235, 255),
                    sprite.get_rect().inflate(-max(2, target_width // 3), -max(3, target_height // 4)),
                    max(1, target_width // 12),
                )
        if obstacle.kind == "air_coin":
            y -= int(panel.height * (0.04 + 0.12 * p))
        elif obstacle.kind == "barrier":
            shadow = pygame.Rect(0, 0, max(18, int(target_width * 0.92)), max(5, int(13 * scale)))
            shadow.midbottom = (int(x), int(y + 5))
            shadow_layer = pygame.Surface(panel.size, pygame.SRCALPHA)
            pygame.draw.ellipse(
                shadow_layer,
                (3, 5, 8, int(95 * alpha / 255)),
                shadow.move(-panel.left, -panel.top),
            )
            self.screen.blit(shadow_layer, panel)
        if alpha < 255:
            sprite = sprite.copy()
            sprite.set_alpha(alpha)
        self.screen.blit(sprite, sprite.get_rect(midbottom=(int(x), int(y))))

    def _draw_runner(self, panel, road_bottom):
        x = self._lane_x(self.game.visual_lane, 1.0, panel.centerx, panel.width * 0.105, road_bottom)
        if self.screen_shake > 0:
            x += math.sin(self.elapsed * 73.0) * 6.0 * min(1.0, self.screen_shake / 0.35)
        base_y = panel.bottom - 34
        jump_factor = self.game.jump_height_factor()
        jump_height = jump_factor * panel.height * 0.205
        bob = 0.0
        if self.game.started and not self.game.paused and jump_factor <= 0:
            bob = abs(math.sin(self.elapsed * (10.0 + self.game.speed_stage))) * 5.0

        shadow_width = int(108 * (1.0 - 0.38 * jump_factor))
        shadow = pygame.Surface((shadow_width + 20, 30), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (3, 5, 8, int(125 * (1.0 - 0.45 * jump_factor))), shadow.get_rect())
        self.screen.blit(shadow, shadow.get_rect(center=(int(x), int(base_y + 1))))

        image = self.assets["runner"]
        sprite_width = max(84, int(panel.width * 0.125))
        sprite_height = int(sprite_width * 1.38)
        stretch = 1.0 + 0.08 * math.sin(math.pi * min(1.0, jump_factor))
        sprite = pygame.transform.smoothscale(
            image,
            (max(1, int(sprite_width / stretch)), max(1, int(sprite_height * stretch))),
        )
        lane_motion = self.game.target_lane - self.game.visual_lane
        sprite = pygame.transform.rotate(sprite, max(-9.0, min(9.0, -lane_motion * 16.0)))

        runner_bottom = int(base_y - jump_height - bob)
        if jump_factor > 0.08 or self.game.speed_stage >= 2:
            trail = sprite.copy()
            trail.fill((76, 208, 202, 80), special_flags=pygame.BLEND_RGBA_MULT)
            trail.set_alpha(58)
            for offset, trail_alpha in ((18, 42), (34, 22)):
                trail.set_alpha(trail_alpha)
                self.screen.blit(trail, trail.get_rect(midbottom=(int(x), runner_bottom + offset)))

        if self.game.invincible_remaining > 0 and int(self.elapsed * 12) % 2:
            sprite.set_alpha(78)
        self.screen.blit(sprite, sprite.get_rect(midbottom=(int(x), runner_bottom)))

    def _draw_hud(self, panel):
        self.start_button_rect = pygame.Rect(0, 0, 0, 0)
        self.restart_button_rect = pygame.Rect(0, 0, 0, 0)
        self.resume_button_rect = pygame.Rect(0, 0, 0, 0)
        self.pause_button_rect = pygame.Rect(0, 0, 0, 0)

        overlay = pygame.Surface((panel.width, 108), pygame.SRCALPHA)
        overlay.fill((12, 17, 23, 232))
        pygame.draw.line(overlay, (255, 157, 65, 220), (0, 106), (panel.width // 2, 106), 2)
        pygame.draw.line(overlay, (52, 205, 201, 220), (panel.width // 2, 106), (panel.width, 106), 2)
        self.screen.blit(overlay, panel.topleft)

        brand = self.small.render("METRO // MOTION", True, (245, 248, 250))
        self.screen.blit(brand, (panel.left + 18, panel.top + 8))
        stage = self.micro.render(self.game.speed_label, True, (255, 174, 82))
        self.screen.blit(stage, stage.get_rect(midtop=(panel.centerx, panel.top + 11)))

        heart_start = panel.right - 121
        for index in range(3):
            self._draw_heart(
                (heart_start + index * 24, panel.top + 17),
                (255, 93, 103) if index < self.game.lives else (69, 74, 82),
            )
        if self.game.started and not self.game.game_over:
            self.pause_button_rect = pygame.Rect(panel.right - 38, panel.top + 5, 28, 28)
            pygame.draw.rect(self.screen, (35, 42, 50), self.pause_button_rect, border_radius=5)
            icon_color = (220, 227, 232)
            pygame.draw.rect(self.screen, icon_color, (self.pause_button_rect.left + 8, self.pause_button_rect.top + 7, 4, 14))
            pygame.draw.rect(self.screen, icon_color, (self.pause_button_rect.left + 16, self.pause_button_rect.top + 7, 4, 14))

        values = (
            ("SCORE", f"{int(self.game.score):05d}", (255, 255, 255)),
            ("DISTANCE", f"{int(self.game.distance)}m", (87, 219, 204)),
            ("PACE", f"{self.game.speed / 0.26:.1f}x", (255, 169, 74)),
            ("COIN VALUE", str(self.game.coins), (255, 202, 79)),
            ("COMBO", f"{self.game.combo} / x{self.game.combo_multiplier}", (255, 105, 112)),
        )
        metrics_left = panel.left + 18
        metric_width = max(95, (panel.width - 36) // len(values))
        for index, (label, value, color) in enumerate(values):
            x = metrics_left + index * metric_width
            if index:
                pygame.draw.line(self.screen, (55, 61, 69), (x - 9, panel.top + 45), (x - 9, panel.top + 94), 1)
            self.screen.blit(self.micro.render(label, True, (143, 153, 162)), (x, panel.top + 40))
            rendered = self.hud_value.render(value, True, color)
            max_width = metric_width - 14
            if rendered.width > max_width:
                rendered = pygame.transform.smoothscale(rendered, (max_width, rendered.height))
            self.screen.blit(rendered, (x, panel.top + 58))

        if self.game.started and not self.game.game_over:
            mission_label, _, goal, _ = self.game.mission
            mission_rect = pygame.Rect(panel.left + 18, panel.bottom - 65, panel.width - 36, 47)
            mission_layer = pygame.Surface(mission_rect.size, pygame.SRCALPHA)
            mission_layer.fill((13, 18, 24, 218))
            pygame.draw.rect(mission_layer, (81, 89, 98, 230), mission_layer.get_rect(), 1, border_radius=6)
            progress_width = int((mission_rect.width - 22) * self.game.mission_ratio)
            pygame.draw.rect(mission_layer, (45, 53, 61), (11, 31, mission_rect.width - 22, 6), border_radius=3)
            pygame.draw.rect(mission_layer, (53, 206, 197), (11, 31, progress_width, 6), border_radius=3)
            self.screen.blit(mission_layer, mission_rect)
            progress_value = f"{int(self.game.mission_progress)}/{int(goal)}"
            self.screen.blit(self.micro.render("RUN OBJECTIVE", True, (255, 169, 74)), (mission_rect.left + 11, mission_rect.top + 7))
            mission_text = self.micro.render(mission_label, True, (235, 239, 242))
            self.screen.blit(mission_text, (mission_rect.left + 126, mission_rect.top + 7))
            progress_text = self.micro.render(progress_value, True, (98, 225, 214))
            self.screen.blit(progress_text, progress_text.get_rect(topright=(mission_rect.right - 11, mission_rect.top + 7)))

        if self.game.combo >= 2 and self.game.started and not self.game.game_over:
            combo_rect = pygame.Rect(panel.left + 18, panel.top + 118, 226, 40)
            combo_layer = pygame.Surface(combo_rect.size, pygame.SRCALPHA)
            combo_layer.fill((13, 18, 24, 210))
            pygame.draw.rect(combo_layer, (255, 192, 91, 190), combo_layer.get_rect(), 1, border_radius=5)
            self.screen.blit(combo_layer, combo_rect)
            combo = self.hud_value.render(f"{self.game.combo} COMBO", True, (255, 232, 161))
            multiplier = self.small.render(f"x{self.game.combo_multiplier}", True, (255, 105, 112))
            self.screen.blit(combo, (combo_rect.left + 10, combo_rect.top + 4))
            self.screen.blit(multiplier, multiplier.get_rect(midright=(combo_rect.right - 10, combo_rect.centery)))

        danger = any(
            obstacle.kind == "barrier"
            and obstacle.lane == self.game.lane
            and not obstacle.checked
            and 0.58 <= obstacle.progress <= 0.86
            for obstacle in self.game.obstacles
        )
        if danger and self.game.jump_height_factor() < 0.20 and not self.game.game_over:
            warning_color = (255, 92, 92) if int(self.elapsed * 7) % 2 else (255, 198, 82)
            warning = self.large.render("HANDS UP", True, warning_color)
            self.screen.blit(warning, warning.get_rect(center=(panel.centerx, panel.top + 184)))
        elif self.game.action_flash > 0 and self.game.started and not self.game.game_over:
            alpha = min(255, int(255 * self.game.action_flash / 0.35))
            label = self.large.render(self.game.last_action, True, (255, 239, 172))
            max_width = panel.width - 90
            if label.width > max_width:
                label = pygame.transform.smoothscale(label, (max_width, label.height))
            label.set_alpha(alpha)
            self.screen.blit(label, label.get_rect(center=(panel.centerx, panel.top + 184)))

        if self.game.game_over:
            shade = pygame.Surface(panel.size, pygame.SRCALPHA)
            shade.fill((8, 11, 16, 218))
            self.screen.blit(shade, panel)
            eyebrow = self.small.render("FINAL RUN REPORT", True, (87, 219, 204))
            title = self.hero.render("RUN OVER", True, (255, 255, 255))
            self.screen.blit(eyebrow, eyebrow.get_rect(center=(panel.centerx, panel.centery - 145)))
            self.screen.blit(title, title.get_rect(center=(panel.centerx, panel.centery - 92)))
            summary = f"{int(self.game.score)} SCORE     {int(self.game.distance)}m     {self.game.max_combo} MAX COMBO"
            summary_text = self.small.render(summary, True, (203, 211, 217))
            self.screen.blit(summary_text, summary_text.get_rect(center=(panel.centerx, panel.centery - 28)))
            self.restart_button_rect = pygame.Rect(panel.centerx - 132, panel.centery + 22, 264, 62)
            self._draw_action_button(self.restart_button_rect, "RUN AGAIN")
            key_hint = self.micro.render("R", True, (126, 137, 147))
            self.screen.blit(key_hint, key_hint.get_rect(center=(panel.centerx, panel.centery + 108)))
        elif not self.game.started:
            shade = pygame.Surface(panel.size, pygame.SRCALPHA)
            shade.fill((8, 12, 18, 195))
            self.screen.blit(shade, panel)
            eyebrow = self.small.render("CAMERA-POWERED RUNNER", True, (83, 222, 207))
            metro = self.hero.render("METRO", True, (255, 255, 255))
            motion = self.hero.render("MOTION", True, (255, 165, 65))
            self.screen.blit(eyebrow, eyebrow.get_rect(center=(panel.centerx, panel.centery - 202)))
            self.screen.blit(metro, metro.get_rect(midright=(panel.centerx - 7, panel.centery - 142)))
            self.screen.blit(motion, motion.get_rect(midleft=(panel.centerx + 7, panel.centery - 142)))
            self._draw_gesture_hint(pygame.Rect(panel.centerx - 238, panel.centery - 74, 142, 92), "left", "LEFT")
            self._draw_gesture_hint(pygame.Rect(panel.centerx - 71, panel.centery - 74, 142, 92), "jump", "JUMP")
            self._draw_gesture_hint(pygame.Rect(panel.centerx + 96, panel.centery - 74, 142, 92), "right", "RIGHT")
            self.start_button_rect = pygame.Rect(panel.centerx - 142, panel.centery + 48, 284, 66)
            self._draw_action_button(self.start_button_rect, "START RUN")
            status = self.micro.render("CAMERA WARM-UP ACTIVE", True, (192, 201, 208))
            self.screen.blit(status, status.get_rect(center=(panel.centerx, panel.centery + 139)))
        elif self.game.paused:
            shade = pygame.Surface(panel.size, pygame.SRCALPHA)
            shade.fill((8, 11, 16, 205))
            self.screen.blit(shade, panel)
            title = self.hero.render("PAUSED", True, (255, 255, 255))
            self.screen.blit(title, title.get_rect(center=(panel.centerx, panel.centery - 62)))
            self.resume_button_rect = pygame.Rect(panel.centerx - 132, panel.centery + 10, 264, 62)
            self._draw_action_button(self.resume_button_rect, "RESUME")

    def _draw_action_button(self, rect: pygame.Rect, label: str) -> None:
        shadow = rect.move(0, 6)
        pygame.draw.rect(self.screen, (8, 18, 20), shadow, border_radius=7)
        pygame.draw.rect(self.screen, (58, 214, 196), rect, border_radius=7)
        pygame.draw.rect(self.screen, (199, 255, 244), rect, 2, border_radius=7)
        rendered = self.font.render(label, True, (8, 31, 29))
        self.screen.blit(rendered, rendered.get_rect(center=rect.center))

    def _draw_heart(self, center: tuple[int, int], color: tuple[int, int, int]) -> None:
        x, y = center
        pygame.draw.circle(self.screen, color, (x - 4, y - 2), 6)
        pygame.draw.circle(self.screen, color, (x + 4, y - 2), 6)
        pygame.draw.polygon(self.screen, color, ((x - 10, y), (x + 10, y), (x, y + 12)))

    def _draw_gesture_hint(self, rect: pygame.Rect, kind: str, label: str) -> None:
        surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        surface.fill((24, 31, 39, 225))
        pygame.draw.rect(surface, (73, 84, 94), surface.get_rect(), 1, border_radius=6)
        center_x, shoulder_y = rect.width // 2, 36
        pose_color = (85, 222, 205) if kind == "jump" else (255, 174, 74)
        pygame.draw.circle(surface, (236, 241, 244), (center_x, 19), 8, 2)
        pygame.draw.line(surface, pose_color, (center_x, 28), (center_x, 56), 4)
        pygame.draw.line(surface, pose_color, (center_x, 55), (center_x - 11, 70), 4)
        pygame.draw.line(surface, pose_color, (center_x, 55), (center_x + 11, 70), 4)
        if kind == "jump":
            pygame.draw.line(surface, pose_color, (center_x, shoulder_y), (center_x - 18, 8), 4)
            pygame.draw.line(surface, pose_color, (center_x, shoulder_y), (center_x + 18, 8), 4)
        elif kind == "left":
            pygame.draw.line(surface, pose_color, (center_x, shoulder_y), (center_x - 28, 30), 4)
            pygame.draw.line(surface, pose_color, (center_x, shoulder_y), (center_x + 12, 47), 4)
        else:
            pygame.draw.line(surface, pose_color, (center_x, shoulder_y), (center_x + 28, 30), 4)
            pygame.draw.line(surface, pose_color, (center_x, shoulder_y), (center_x - 12, 47), 4)
        text = self.micro.render(label, True, (226, 232, 236))
        surface.blit(text, text.get_rect(midbottom=(rect.width // 2, rect.height - 6)))
        self.screen.blit(surface, rect)

    def _draw_nus_milestone(self, panel: pygame.Rect) -> None:
        """Draw the one-off 3-orange + 3-blue reward without stopping camera capture."""
        if self.game.nus_animation_remaining <= 0:
            return

        elapsed = self.game.nus_animation_duration - self.game.nus_animation_remaining
        fade_in = min(1.0, elapsed / 0.25)
        fade_out = min(1.0, self.game.nus_animation_remaining / 0.35)
        opacity = int(235 * min(fade_in, fade_out))
        layer = pygame.Surface(panel.size, pygame.SRCALPHA)
        layer.fill((5, 9, 18, opacity))

        center = (panel.width // 2, panel.height // 2)
        # Rotating rays make the milestone readable as a deliberate reward cutscene.
        ray_color = (255, 255, 255, int(55 * min(fade_in, fade_out)))
        for index in range(20):
            angle = elapsed * 0.9 + index * math.tau / 20
            inner = 125
            outer = max(panel.width, panel.height) * 0.72
            start = (center[0] + math.cos(angle) * inner, center[1] + math.sin(angle) * inner)
            end = (center[0] + math.cos(angle) * outer, center[1] + math.sin(angle) * outer)
            pygame.draw.line(layer, ray_color, start, end, 5)

        letter_colors = ((245, 125, 35), (255, 255, 255), (45, 145, 255))
        letter_x = (center[0] - 150, center[0], center[0] + 150)
        for index, (letter, color, x) in enumerate(zip("NUS", letter_colors, letter_x)):
            local_time = elapsed - 0.18 * index
            if local_time < 0:
                continue
            # Each letter overshoots once, then settles at full size.
            scale = min(1.0, local_time / 0.24)
            scale = max(0.05, scale + math.sin(min(1.0, local_time / 0.42) * math.pi) * 0.18)
            glyph = self.nus_font.render(letter, True, color)
            size = (max(1, int(glyph.width * scale)), max(1, int(glyph.height * scale)))
            glyph = pygame.transform.smoothscale(glyph, size)
            layer.blit(glyph, glyph.get_rect(center=(x, center[1] - 10)))

        heading = self.milestone_font.render("MILESTONE UNLOCKED", True, (255, 230, 150))
        detail = self.font.render("3 ORANGE + 3 BLUE", True, (225, 235, 245))
        layer.blit(heading, heading.get_rect(center=(center[0], center[1] - 145)))
        layer.blit(detail, detail.get_rect(center=(center[0], center[1] + 135)))
        self.screen.blit(layer, panel.topleft)

    def _draw_camera(self, panel: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (18, 24, 29), panel)
        frame, status, fps, progress, pose = self.worker.snapshot()
        pygame.draw.rect(self.screen, (12, 16, 21), (panel.left, panel.top, panel.width, 58))
        pygame.draw.line(self.screen, (55, 205, 198), (panel.left, 57), (panel.right, 57), 2)
        self.screen.blit(self.font.render("PLAYER TRACKING", True, (247, 249, 250)), (panel.left + 18, panel.top + 15))
        live_color = (73, 218, 165) if frame is not None else (255, 157, 75)
        pygame.draw.circle(self.screen, live_color, (panel.right - 98, panel.top + 28), 5)
        live = self.micro.render(f"POSE {fps:.0f} FPS", True, (176, 187, 195))
        self.screen.blit(live, live.get_rect(midleft=(panel.right - 86, panel.top + 28)))

        camera_width = max(1, panel.width - 36)
        camera_height = min(int(camera_width * 9 / 16), int(panel.height * 0.47))
        camera_area = pygame.Rect(panel.left + 18, panel.top + 74, camera_width, max(1, camera_height))
        pygame.draw.rect(self.screen, (6, 9, 12), camera_area, border_radius=6)
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surface = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
            scale = min(camera_area.width / surface.get_width(), camera_area.height / surface.get_height())
            size = (max(1, int(surface.get_width() * scale)), max(1, int(surface.get_height() * scale)))
            surface = pygame.transform.smoothscale(surface, size)
            self.screen.blit(surface, surface.get_rect(center=camera_area.center))
        else:
            message = self.small.render("WAITING FOR CAMERA", True, (194, 202, 208))
            self.screen.blit(message, message.get_rect(center=camera_area.center))
        pygame.draw.rect(self.screen, (74, 87, 96), camera_area, 2, border_radius=6)

        state_rect = pygame.Rect(panel.left + 18, camera_area.bottom + 14, panel.width - 36, 66)
        pygame.draw.rect(self.screen, (27, 34, 41), state_rect, border_radius=6)
        pygame.draw.rect(self.screen, (61, 72, 81), state_rect, 1, border_radius=6)
        self.screen.blit(self.micro.render("ACTIVE PLAYER", True, (111, 222, 203)), (state_rect.left + 12, state_rect.top + 9))
        status_text = self._render_fitted(status, self.small, (231, 235, 238), state_rect.width - 24)
        self.screen.blit(status_text, (state_rect.left + 12, state_rect.top + 27))
        pygame.draw.rect(self.screen, (49, 58, 66), (state_rect.left + 12, state_rect.bottom - 10, state_rect.width - 24, 4), border_radius=2)
        pygame.draw.rect(
            self.screen,
            (55, 207, 197),
            (state_rect.left + 12, state_rect.bottom - 10, int((state_rect.width - 24) * progress), 4),
            border_radius=2,
        )

        self._draw_stickman_inset(panel, pose)

        control_left = panel.left + 18
        control_top = state_rect.bottom + 17
        self.screen.blit(self.micro.render("GESTURE INPUT", True, (139, 150, 159)), (control_left, control_top))
        controls = (("LEFT ARM", "LEFT"), ("HANDS UP", "JUMP"), ("RIGHT ARM", "RIGHT"))
        for index, (gesture, action) in enumerate(controls):
            row = pygame.Rect(control_left, control_top + 25 + index * 44, max(142, panel.width - 252), 35)
            pygame.draw.rect(self.screen, (29, 37, 44), row, border_radius=5)
            pygame.draw.rect(self.screen, (53, 64, 72), row, 1, border_radius=5)
            self.screen.blit(self.micro.render(gesture, True, (220, 226, 230)), (row.left + 10, row.top + 9))
            action_text = self.micro.render(action, True, (255, 177, 78) if action != "JUMP" else (79, 217, 203))
            self.screen.blit(action_text, action_text.get_rect(midright=(row.right - 9, row.centery)))

        key_y = panel.bottom - 35
        sound_state = "ON" if self.sounds.enabled else "OFF"
        keys = self.micro.render(f"C CALIBRATE   P PAUSE   M SOUND {sound_state}   ESC EXIT", True, (130, 141, 149))
        self.screen.blit(keys, (panel.left + 18, key_y))
        pygame.draw.line(self.screen, (62, 70, 77), (panel.left, panel.top), (panel.left, panel.bottom), 3)

    @staticmethod
    def _render_fitted(text: str, font: pygame.font.Font, color, max_width: int) -> pygame.Surface:
        rendered = font.render(text, True, color)
        if rendered.width <= max_width:
            return rendered
        shortened = text
        while shortened and font.size(shortened + "...")[0] > max_width:
            shortened = shortened[:-1]
        return font.render(shortened + "...", True, color)

    def _draw_stickman_inset(self, panel: pygame.Rect, pose) -> None:
        """Render the locked dancer as a background-free live stickman in the lower right."""
        inset_width = min(214, max(176, int(panel.width * 0.39)))
        inset_height = min(246, max(205, int(panel.height * 0.30)))
        rect = pygame.Rect(panel.right - inset_width - 18, panel.bottom - inset_height - 50, inset_width, inset_height)
        surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        surface.fill((12, 17, 22, 238))
        pygame.draw.rect(surface, (72, 215, 201), surface.get_rect(), 2, border_radius=6)
        pygame.draw.circle(surface, (72, 220, 170) if pose is not None else (126, 134, 142), (14, 17), 4)
        title = self.micro.render("LIVE STICKMAN", True, (137, 238, 220))
        surface.blit(title, (24, 9))
        if pose is None:
            waiting = self.micro.render("LOCKING PLAYER", True, (150, 157, 165))
            surface.blit(waiting, waiting.get_rect(center=(rect.width // 2, rect.height // 2)))
        else:
            points, confidence = pose
            visible = confidence > 0.25
            valid_points = points[visible]
            if len(valid_points) >= 4:
                low, high = valid_points.min(axis=0), valid_points.max(axis=0)
                span = np.maximum(high - low, 1e-4)
                drawing = pygame.Rect(18, 35, rect.width - 36, rect.height - 48)
                scale = min(drawing.width / span[0], drawing.height / span[1])
                center = (low + high) * 0.5
                pixels = np.column_stack((
                    (points[:, 0] - center[0]) * scale + drawing.centerx,
                    (points[:, 1] - center[1]) * scale + drawing.centery,
                )).astype(int)
                for first, second in SKELETON:
                    if visible[first] and visible[second]:
                        pygame.draw.line(surface, (49, 211, 194), pixels[first], pixels[second], 5)
                for index, point in enumerate(pixels):
                    if visible[index]:
                        pygame.draw.circle(surface, (245, 248, 250), point, 5)
                        pygame.draw.circle(surface, (49, 211, 194), point, 5, 2)
        self.screen.blit(surface, rect)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose-controlled three-lane runner Bonus Level prototype.")
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index or phone stream URL, e.g. http://192.168.1.8:8080/video",
    )
    parser.add_argument("--no-mirror", action="store_true", help="Do not horizontally mirror the camera.")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference size; use 256 for lower latency or 416 for more detail.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Pose model not found: {MODEL_PATH}")
    required_assets = [ASSET_DIR / f"{name}.png" for name in ("runner", "barrier", "coin", "skyline")]
    if not all(path.exists() for path in required_assets):
        raise FileNotFoundError("Game assets are missing. Run: python bonus_runner\\create_assets.py")
    MetroMotionApp(parse_camera_source(args.camera), not args.no_mirror, args.imgsz).run()


if __name__ == "__main__":
    main()
