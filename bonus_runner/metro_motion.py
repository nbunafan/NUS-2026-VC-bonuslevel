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
from dataclasses import dataclass
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

    def __init__(self, source, mirror: bool, image_size: int = 416):
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

    def _run(self) -> None:
        # Import after YOLO_CONFIG_DIR is set so Ultralytics never writes to a protected
        # roaming directory on the classroom Windows setup.
        from ultralytics import YOLO

        model = YOLO(str(MODEL_PATH))
        backend = cv2.CAP_DSHOW if isinstance(self.source, int) and os.name == "nt" else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.source, backend)
        if isinstance(self.source, int):
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            with self.lock:
                self.status = "CAMERA CONNECTION FAILED"
            self.running = False
            return

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

            results = model.predict(frame, imgsz=self.image_size, conf=0.30, verbose=False)
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


class RunnerGame:
    def __init__(self):
        self.lane = 1
        self.target_lane = 1.0
        self.visual_lane = 1.0
        self.jump_remaining = 0.0
        self.obstacles: list[Obstacle] = []
        self.spawn_timer = 1.2
        self.score = 0.0
        self.coins = 0
        self.lives = 3
        self.speed = 0.26
        self.game_over = False
        self.last_action = "READY"
        self.action_flash = 0.0
        self.started = False

    def reset(self) -> None:
        self.__init__()

    def start(self) -> None:
        """Start spawning and scoring only after the player presses START RUN."""
        self.started = True

    def command(self, action: str) -> None:
        if self.game_over or not self.started:
            return
        if action == "LEFT" and self.lane > 0:
            self.lane -= 1
            self.target_lane = float(self.lane)
        elif action == "RIGHT" and self.lane < 2:
            self.lane += 1
            self.target_lane = float(self.lane)
        elif action == "JUMP" and self.jump_remaining <= 0:
            self.jump_remaining = 0.86
        self.last_action = action
        self.action_flash = 0.40

    def update(self, dt: float) -> None:
        if self.game_over or not self.started:
            return
        self.score += dt * 12.0
        self.speed = min(0.43, 0.26 + self.score / 5000.0)
        self.jump_remaining = max(0.0, self.jump_remaining - dt)
        self.action_flash = max(0.0, self.action_flash - dt)
        self.visual_lane += (self.target_lane - self.visual_lane) * min(1.0, dt * 12.0)

        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            kind = random.choices(["barrier", "coin"], weights=[0.58, 0.42])[0]
            self.obstacles.append(Obstacle(random.randrange(3), kind))
            self.spawn_timer = random.uniform(1.05, 1.70) / (self.speed / 0.26)

        remaining = []
        for obstacle in self.obstacles:
            obstacle.progress += dt * self.speed
            if not obstacle.resolved and obstacle.progress >= 0.90:
                obstacle.resolved = True
                if obstacle.lane == self.lane:
                    if obstacle.kind == "coin":
                        self.coins += 1
                        self.score += 25
                    elif obstacle.kind == "barrier" and self.jump_remaining <= 0.18:
                        self._hit()
            if obstacle.progress < 1.12:
                remaining.append(obstacle)
        self.obstacles = remaining

    def _hit(self) -> None:
        self.lives -= 1
        self.score = max(0.0, self.score - 40)
        if self.lives <= 0:
            self.game_over = True


class MetroMotionApp:
    WIDTH, HEIGHT = 1440, 810
    GAME_WIDTH = 870

    def __init__(self, camera_source, mirror: bool, image_size: int):
        pygame.init()
        pygame.display.set_caption("Metro Motion - Pose Controlled Runner")
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 24)
        self.small = pygame.font.SysFont("segoeui", 17)
        self.large = pygame.font.SysFont("segoeui", 52, bold=True)
        self.game = RunnerGame()
        self.worker = PoseCameraWorker(camera_source, mirror, image_size)
        self.worker.start()
        self.running = True
        self.assets = self._load_assets()

    def _load_assets(self):
        return {
            name: pygame.image.load(str(ASSET_DIR / f"{name}.png")).convert_alpha()
            for name in ("runner", "barrier", "coin")
        } | {"skyline": pygame.image.load(str(ASSET_DIR / "skyline.png")).convert()}

    def run(self) -> None:
        while self.running:
            dt = min(self.clock.tick(60) / 1000.0, 0.05)
            self._events()
            while not self.worker.actions.empty():
                self.game.command(self.worker.actions.get())
            self.game.update(dt)
            self._draw()
            pygame.display.flip()
        self.worker.stop()
        pygame.quit()

    def _events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
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
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if getattr(self, "start_button_rect", pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                    self.game.start()

    def _draw(self) -> None:
        width, height = self.screen.get_size()
        game_width = int(width * 0.60)
        self._draw_game(pygame.Rect(0, 0, game_width, height))
        self._draw_camera(pygame.Rect(game_width, 0, width - game_width, height))

    def _draw_game(self, panel: pygame.Rect) -> None:
        skyline = pygame.transform.smoothscale(self.assets["skyline"], panel.size)
        self.screen.blit(skyline, panel)
        horizon = panel.top + int(panel.height * 0.29)
        bottom = panel.bottom
        center = panel.centerx
        road_top = panel.width * 0.13
        road_bottom = panel.width * 0.55
        pygame.draw.polygon(
            self.screen,
            (46, 51, 57),
            [(center - road_top, horizon), (center + road_top, horizon), (center + road_bottom, bottom), (center - road_bottom, bottom)],
        )
        for divider in (-0.5, 0.5):
            x_top = center + divider * road_top * 1.35
            x_bottom = center + divider * road_bottom * 1.35
            pygame.draw.line(self.screen, (235, 206, 92), (x_top, horizon), (x_bottom, bottom), 5)
        for offset in (-1.0, 0.0, 1.0):
            x_top = center + offset * road_top * 0.90
            x_bottom = center + offset * road_bottom * 0.90
            pygame.draw.line(self.screen, (144, 151, 157), (x_top, horizon), (x_bottom, bottom), 4)

        for obstacle in sorted(self.game.obstacles, key=lambda item: item.progress):
            self._draw_obstacle(panel, obstacle, horizon, road_top, road_bottom)
        self._draw_runner(panel, road_bottom)
        self._draw_hud(panel)

    def _lane_x(self, lane: float, progress: float, center: float, road_top: float, road_bottom: float) -> float:
        half_width = road_top + (road_bottom - road_top) * progress
        return center + (lane - 1.0) * half_width * 0.58

    def _draw_obstacle(self, panel, obstacle, horizon, road_top, road_bottom):
        p = max(0.02, min(1.0, obstacle.progress))
        y = horizon + (panel.bottom - horizon) * (p ** 1.55)
        x = self._lane_x(obstacle.lane, p, panel.centerx, road_top, road_bottom)
        scale = 0.18 + 0.82 * p
        image = self.assets[obstacle.kind]
        target_width = max(20, int(image.get_width() * scale))
        target_height = max(20, int(image.get_height() * scale))
        sprite = pygame.transform.smoothscale(image, (target_width, target_height))
        self.screen.blit(sprite, sprite.get_rect(midbottom=(int(x), int(y))))

    def _draw_runner(self, panel, road_bottom):
        x = self._lane_x(self.game.visual_lane, 1.0, panel.centerx, panel.width * 0.13, road_bottom)
        base_y = panel.bottom - 36
        jump_height = 0.0
        if self.game.jump_remaining > 0:
            phase = 1.0 - self.game.jump_remaining / 0.86
            jump_height = math.sin(math.pi * phase) * panel.height * 0.20
        image = self.assets["runner"]
        sprite = pygame.transform.smoothscale(image, (112, 154))
        self.screen.blit(sprite, sprite.get_rect(midbottom=(int(x), int(base_y - jump_height))))

    def _draw_hud(self, panel):
        overlay = pygame.Surface((panel.width, 92), pygame.SRCALPHA)
        overlay.fill((16, 19, 24, 220))
        self.screen.blit(overlay, panel.topleft)
        self.screen.blit(self.font.render(f"SCORE {int(self.game.score):05d}", True, (255, 255, 255)), (22, 18))
        self.screen.blit(self.font.render(f"COINS {self.game.coins}", True, (250, 203, 59)), (22, 50))
        lives = "LIVES " + "|".join("1" for _ in range(self.game.lives))
        self.screen.blit(self.font.render(lives, True, (239, 88, 82)), (panel.width - 170, 18))
        if self.game.action_flash > 0:
            label = self.large.render(self.game.last_action, True, (255, 238, 130))
            self.screen.blit(label, label.get_rect(center=(panel.centerx, 130)))
        if self.game.game_over:
            shade = pygame.Surface(panel.size, pygame.SRCALPHA)
            shade.fill((10, 12, 15, 185))
            self.screen.blit(shade, panel)
            title = self.large.render("RUN ENDED", True, (255, 255, 255))
            note = self.font.render("Press R to restart", True, (250, 203, 59))
            self.screen.blit(title, title.get_rect(center=(panel.centerx, panel.centery - 25)))
            self.screen.blit(note, note.get_rect(center=(panel.centerx, panel.centery + 35)))
        elif not self.game.started:
            shade = pygame.Surface(panel.size, pygame.SRCALPHA)
            shade.fill((8, 11, 15, 155))
            self.screen.blit(shade, panel)
            title = self.large.render("READY TO RUN?", True, (255, 255, 255))
            self.screen.blit(title, title.get_rect(center=(panel.centerx, panel.centery - 65)))
            self.start_button_rect = pygame.Rect(panel.centerx - 118, panel.centery - 5, 236, 62)
            pygame.draw.rect(self.screen, (54, 205, 166), self.start_button_rect, border_radius=10)
            pygame.draw.rect(self.screen, (190, 255, 230), self.start_button_rect, 2, border_radius=10)
            label = self.font.render("START RUN", True, (7, 25, 22))
            self.screen.blit(label, label.get_rect(center=self.start_button_rect.center))
            hint = self.small.render("Camera can calibrate before you begin", True, (210, 220, 225))
            self.screen.blit(hint, hint.get_rect(center=(panel.centerx, panel.centery + 88)))

    def _draw_camera(self, panel: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (22, 25, 30), panel)
        frame, status, fps, progress, pose = self.worker.snapshot()
        camera_area = panel.inflate(-24, -116)
        camera_area.top += 45
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surface = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
            scale = min(camera_area.width / surface.get_width(), camera_area.height / surface.get_height())
            size = (max(1, int(surface.get_width() * scale)), max(1, int(surface.get_height() * scale)))
            surface = pygame.transform.smoothscale(surface, size)
            self.screen.blit(surface, surface.get_rect(center=camera_area.center))
        else:
            message = self.font.render(status, True, (230, 230, 230))
            self.screen.blit(message, message.get_rect(center=camera_area.center))

        self._draw_stickman_inset(panel, pose)

        pygame.draw.line(self.screen, (55, 61, 68), (panel.left, 0), (panel.left, panel.bottom), 3)
        self.screen.blit(self.font.render("PLAYER CAMERA", True, (255, 255, 255)), (panel.left + 18, 12))
        footer_y = panel.bottom - 58
        self.screen.blit(self.small.render(f"{status}  |  Pose {fps:.1f} FPS", True, (190, 198, 205)), (panel.left + 18, footer_y))
        controls = "C recalibrate  |  Arrows/WASD fallback  |  ESC exit"
        self.screen.blit(self.small.render(controls, True, (138, 220, 190)), (panel.left + 18, footer_y + 26))

    def _draw_stickman_inset(self, panel: pygame.Rect, pose) -> None:
        """Render the locked dancer as a background-free live stickman in the lower right."""
        inset_width = min(210, max(150, panel.width // 3))
        inset_height = min(250, max(190, panel.height // 3))
        rect = pygame.Rect(panel.right - inset_width - 18, panel.bottom - inset_height - 76, inset_width, inset_height)
        surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        surface.fill((10, 13, 18, 220))
        pygame.draw.rect(surface, (72, 220, 170), surface.get_rect(), 2, border_radius=6)
        title = self.small.render("LIVE STICKMAN", True, (130, 245, 205))
        surface.blit(title, (12, 9))
        if pose is None:
            waiting = self.small.render("LOCKING DANCER", True, (150, 157, 165))
            surface.blit(waiting, waiting.get_rect(center=(rect.width // 2, rect.height // 2)))
        else:
            points, confidence = pose
            visible = confidence > 0.25
            valid_points = points[visible]
            if len(valid_points) >= 4:
                low, high = valid_points.min(axis=0), valid_points.max(axis=0)
                span = np.maximum(high - low, 1e-4)
                drawing = pygame.Rect(18, 38, rect.width - 36, rect.height - 54)
                scale = min(drawing.width / span[0], drawing.height / span[1])
                center = (low + high) * 0.5
                pixels = np.column_stack((
                    (points[:, 0] - center[0]) * scale + drawing.centerx,
                    (points[:, 1] - center[1]) * scale + drawing.centery,
                )).astype(int)
                for first, second in SKELETON:
                    if visible[first] and visible[second]:
                        pygame.draw.line(surface, (72, 220, 170), pixels[first], pixels[second], 4)
                for index, point in enumerate(pixels):
                    if visible[index]:
                        pygame.draw.circle(surface, (245, 248, 250), point, 4)
                        pygame.draw.circle(surface, (72, 220, 170), point, 4, 2)
        self.screen.blit(surface, rect)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pose-controlled three-lane runner Bonus Level prototype.")
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index or phone stream URL, e.g. http://192.168.1.8:8080/video",
    )
    parser.add_argument("--no-mirror", action="store_true", help="Do not horizontally mirror the camera.")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference size; lower this on a slow CPU.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Pose model not found: {MODEL_PATH}")
    required_assets = [ASSET_DIR / f"{name}.png" for name in ("runner", "barrier", "coin", "skyline")]
    if not all(path.exists() for path in required_assets):
        raise FileNotFoundError("Game assets are missing. Run: python bonus_runner\\create_assets.py")
    MetroMotionApp(parse_camera_source(args.camera), not args.no_mirror, args.imgsz).run()


if __name__ == "__main__":
    main()
