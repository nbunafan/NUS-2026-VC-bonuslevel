from __future__ import annotations

"""Render deterministic UI screenshots without opening a camera or loading YOLO."""

import argparse
import os
import queue
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import numpy as np
import pygame

from metro_motion import GameEvent, MetroMotionApp, Obstacle


class PreviewWorker:
    def __init__(self):
        self.actions: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.frame, self.pose = self._make_camera_frame()

    @staticmethod
    def _make_camera_frame():
        height, width = 720, 1280
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            ratio = y / max(1, height - 1)
            frame[y, :, :] = (
                int(35 + 40 * ratio),
                int(30 + 24 * ratio),
                int(24 + 18 * ratio),
            )
        cv2.rectangle(frame, (0, 565), (width, height), (35, 43, 47), -1)
        cv2.putText(frame, "DEMO CAMERA", (48, 660), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 135, 140), 2, cv2.LINE_AA)

        points = np.zeros((17, 2), dtype=np.float32)
        points[0] = (0.50, 0.16)
        points[5], points[6] = (0.43, 0.31), (0.57, 0.31)
        points[7], points[8] = (0.36, 0.22), (0.64, 0.22)
        points[9], points[10] = (0.31, 0.10), (0.69, 0.10)
        points[11], points[12] = (0.46, 0.52), (0.54, 0.52)
        points[13], points[14] = (0.43, 0.70), (0.57, 0.70)
        points[15], points[16] = (0.40, 0.91), (0.60, 0.91)
        confidence = np.ones(17, dtype=np.float32)

        pixels = np.column_stack((points[:, 0] * width, points[:, 1] * height)).astype(int)
        skeleton = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16))
        for first, second in skeleton:
            cv2.line(frame, tuple(pixels[first]), tuple(pixels[second]), (61, 213, 194), 7, cv2.LINE_AA)
        for index in (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
            cv2.circle(frame, tuple(pixels[index]), 9, (245, 245, 245), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pixels[index]), 9, (61, 213, 194), 3, cv2.LINE_AA)
        return frame, (points, confidence)

    def start(self):
        pass

    def stop(self):
        pass

    def reset_calibration(self):
        pass

    def snapshot(self):
        return self.frame.copy(), "HANDS UP / PLAYER LOCKED", 28.4, 1.0, self.pose


def render(output: Path, start_screen: bool, size: tuple[int, int]) -> None:
    worker = PreviewWorker()
    app = MetroMotionApp(0, True, 320, worker=worker)
    app.screen = pygame.display.set_mode(size)
    if not start_screen:
        game = app.game
        game.start()
        game.score = 2840
        game.distance = 426
        game.speed = 0.37
        game.speed_stage = 2
        game.combo = 8
        game.max_combo = 12
        game.combo_timer = 3.4
        game.action_flash = 0.0
        game.coins = 17
        game.orange_coins_collected = 4
        game.blue_coins_collected = 3
        game.mission_index = 1
        game.mission_progress = 2
        game.obstacles = [
            Obstacle(0, "coin", 0.46),
            Obstacle(1, "barrier", 0.68),
            Obstacle(2, "air_coin", 0.35),
        ]
        game.events.append(GameEvent("dodge", 1, "PERFECT DODGE +90", (85, 238, 181)))
        app.current_game_panel = pygame.Rect(0, 0, int(size[0] * 0.615), size[1])
        app._consume_game_events()
    app.elapsed = 3.7
    app._draw()
    output.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(app.screen, str(output))
    app.worker.stop()
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Metro Motion without a live camera.")
    parser.add_argument("output", nargs="?", default="metro_motion_preview.png")
    parser.add_argument("--start-screen", action="store_true")
    parser.add_argument("--size", default="1440x810", help="Preview size, for example 1280x720.")
    args = parser.parse_args()
    width_text, height_text = args.size.lower().split("x", 1)
    render(Path(args.output).resolve(), args.start_screen, (int(width_text), int(height_text)))


if __name__ == "__main__":
    main()
