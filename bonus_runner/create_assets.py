"""Generate the original Metro Motion sprite set with only pygame.

The small gameplay sprites are drawn at four times their delivery size and
then downsampled.  This keeps edges clean while preserving the dimensions
that ``metro_motion.py`` uses for perspective scaling and collision feedback.
The skyline is delivered at 1920x1080 because it is always scaled to the game
panel and benefits from retaining detail on large displays.

Run from any directory:

    python create_assets.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import pygame


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
SUPERSAMPLE = 4

DELIVERY_SIZES = {
    "runner.png": (160, 220),
    "barrier.png": (220, 150),
    "coin.png": (96, 96),
    "skyline.png": (1920, 1080),
}


Color = tuple[int, int, int] | tuple[int, int, int, int]
Point = tuple[float, float]


class ScaledCanvas:
    """Tiny vector helper whose coordinates stay in delivery pixels."""

    def __init__(self, size: tuple[int, int], scale: int = SUPERSAMPLE):
        self.size = size
        self.scale = scale
        self.surface = pygame.Surface(
            (size[0] * scale, size[1] * scale), pygame.SRCALPHA
        )

    def p(self, point: Point) -> tuple[int, int]:
        return (round(point[0] * self.scale), round(point[1] * self.scale))

    def r(self, rect: Sequence[float]) -> pygame.Rect:
        return pygame.Rect(*(round(value * self.scale) for value in rect))

    def polygon(self, color: Color, points: Iterable[Point]) -> None:
        scaled = [self.p(point) for point in points]
        pygame.draw.polygon(self.surface, color, scaled)
        pygame.draw.aalines(self.surface, color, True, scaled)

    def line(
        self,
        color: Color,
        start: Point,
        end: Point,
        width: float,
        rounded: bool = True,
    ) -> None:
        pygame.draw.line(
            self.surface,
            color,
            self.p(start),
            self.p(end),
            max(1, round(width * self.scale)),
        )
        if rounded:
            radius = max(1, round(width * self.scale / 2))
            pygame.draw.circle(self.surface, color, self.p(start), radius)
            pygame.draw.circle(self.surface, color, self.p(end), radius)

    def ellipse(
        self,
        color: Color,
        rect: Sequence[float],
        width: float = 0,
    ) -> None:
        pygame.draw.ellipse(
            self.surface,
            color,
            self.r(rect),
            max(0, round(width * self.scale)),
        )

    def rect(
        self,
        color: Color,
        rect: Sequence[float],
        width: float = 0,
        radius: float = 0,
    ) -> None:
        pygame.draw.rect(
            self.surface,
            color,
            self.r(rect),
            max(0, round(width * self.scale)),
            border_radius=max(0, round(radius * self.scale)),
        )

    def finish(self) -> pygame.Surface:
        return pygame.transform.smoothscale(self.surface, self.size)


def _jointed_limb(
    canvas: ScaledCanvas,
    points: Sequence[Point],
    fill: Color,
    width: float,
    outline: Color = (12, 21, 39, 255),
    outline_width: float = 4,
) -> None:
    """Draw a readable bent limb, including clean rounded joints."""

    for start, end in zip(points, points[1:]):
        canvas.line(outline, start, end, width + outline_width)
    for start, end in zip(points, points[1:]):
        canvas.line(fill, start, end, width)


def build_runner() -> pygame.Surface:
    """Create a bright, original city athlete in a clear running pose."""

    c = ScaledCanvas((160, 220))

    # A compact contact shadow anchors the sprite without producing a dark box.
    for index, alpha in enumerate((18, 28, 38, 50)):
        inset = index * 3
        c.ellipse((7, 13, 28, alpha), (25 + inset, 203 + inset / 3, 112 - 2 * inset, 14 - inset / 2))

    # Rear arm and leg are deliberately cooler and darker to create depth.
    _jointed_limb(c, ((56, 75), (31, 101), (20, 130)), (51, 155, 190), 13)
    c.ellipse((255, 181, 135), (13, 123, 15, 17))
    _jointed_limb(c, ((77, 125), (104, 159), (122, 199)), (31, 62, 91), 21)
    c.polygon((18, 29, 49), ((111, 194), (132, 194), (144, 205), (139, 213), (115, 211)))
    c.polygon((51, 227, 207), ((116, 196), (133, 196), (141, 204), (121, 207)))

    # Front leg has a coral knee panel and a luminous shoe for instant silhouette recognition.
    _jointed_limb(c, ((76, 127), (55, 164), (42, 202)), (39, 83, 116), 23)
    c.ellipse((247, 100, 94), (43, 150, 25, 25))
    c.polygon((14, 25, 44), ((30, 198), (48, 197), (55, 207), (48, 215), (23, 211)))
    c.polygon((255, 197, 74), ((29, 201), (49, 201), (53, 207), (29, 208)))

    # Jacket: asymmetrical panels make the character feel in motion even at rest.
    c.polygon((10, 20, 38), ((51, 59), (83, 52), (111, 68), (108, 126), (77, 143), (48, 122)))
    c.polygon((244, 82, 91), ((54, 62), (80, 56), (77, 137), (50, 119)))
    c.polygon((255, 183, 63), ((80, 56), (107, 70), (104, 121), (79, 138)))
    c.polygon((255, 229, 173), ((80, 61), (92, 67), (80, 128), (72, 130)))
    c.line((24, 38, 61), (80, 57), (78, 137), 3, rounded=False)
    c.polygon((35, 219, 206), ((53, 104), (77, 113), (77, 124), (50, 115)))
    c.polygon((17, 52, 71), ((53, 105), (63, 108), (62, 119), (51, 115)))

    # Forward arm uses the warm jacket color and a raised elbow.
    _jointed_limb(c, ((104, 74), (130, 94), (142, 69)), (245, 96, 91), 14)
    c.ellipse((255, 188, 142), (135, 59, 16, 18))
    c.line((255, 207, 159), (140, 68), (147, 62), 4)

    # Head, hair and visor remain legible after the in-game 112x154 resize.
    c.ellipse((12, 21, 39), (59, 17, 45, 48))
    c.ellipse((255, 190, 143), (63, 20, 38, 41))
    c.polygon((19, 31, 52), ((62, 34), (67, 18), (91, 15), (105, 30), (91, 27), (82, 38)))
    c.polygon((44, 226, 210), ((62, 33), (101, 28), (104, 36), (64, 41)))
    c.polygon((201, 255, 244), ((70, 33), (96, 30), (92, 36), (72, 39)))
    c.ellipse((25, 39, 57), (89, 41, 3, 3))
    c.line((157, 69, 66), (82, 53), (92, 51), 1.5, rounded=False)

    # Small reflective accents add polish without creating unreadable noise.
    c.line((232, 255, 248), (98, 79), (101, 102), 2)
    c.line((109, 246, 229), (58, 144), (45, 183), 2)
    c.ellipse((255, 245, 189), (132, 201, 8, 4))
    return c.finish()


def build_barrier() -> pygame.Surface:
    """Create a sturdy neon transit barrier with a strong readable outline."""

    c = ScaledCanvas((220, 150))
    c.ellipse((5, 12, 29, 75), (19, 128, 183, 15))

    # Feet and telescoping posts.
    for x in (39, 167):
        c.rect((9, 18, 35), (x, 93, 15, 37), radius=4)
        c.rect((37, 216, 203), (x + 4, 98, 7, 29), radius=3)
        c.rect((11, 21, 39), (x - 12, 124, 39, 12), radius=5)
        c.rect((249, 94, 91), (x - 7, 127, 29, 4), radius=2)

    # Outer casing and inset sign plate.
    c.rect((7, 15, 31), (10, 21, 200, 100), radius=15)
    c.rect((41, 229, 212), (13, 24, 194, 94), radius=13)
    c.rect((16, 29, 50), (19, 30, 182, 82), radius=10)
    c.rect((255, 224, 145), (25, 36, 170, 70), radius=7)

    # Bold diagonal warning pattern, clipped visually by the inset border.
    stripe_colors = ((247, 86, 92), (255, 190, 56))
    # Clip at the inset face.  The heavy rounded border below conceals the
    # rectangular clip corners, giving the stripes a precise rounded finish.
    c.surface.set_clip(c.r((27, 38, 166, 66)))
    for index, x in enumerate(range(-1, 207, 35)):
        color = stripe_colors[index % 2]
        c.polygon(color, ((x, 100), (x + 23, 100), (x + 72, 42), (x + 49, 42)))
    c.surface.set_clip(None)
    # Repaint the border so stripe ends are clean and the silhouette remains crisp.
    c.rect((17, 31, 52), (22, 33, 176, 76), width=7, radius=9)
    c.rect((221, 255, 245), (27, 38, 166, 66), width=2, radius=6)

    # Central transit chevrons communicate "clear this obstacle" at a glance.
    c.polygon((12, 25, 45), ((82, 51), (103, 71), (82, 91), (91, 99), (121, 71), (91, 43)))
    c.polygon((225, 255, 247), ((93, 51), (113, 71), (93, 91), (101, 98), (130, 71), (101, 44)))

    for x in (28, 183):
        c.ellipse((7, 18, 34), (x - 6, 14, 18, 18))
        c.ellipse((95, 255, 231), (x - 3, 17, 12, 12))
        c.ellipse((224, 255, 248), (x, 19, 5, 5))
    c.line((232, 255, 249), (28, 28), (190, 28), 2, rounded=False)
    return c.finish()


def build_coin() -> pygame.Surface:
    """Create a faceted token whose silhouette also recolors cleanly in-game."""

    c = ScaledCanvas((96, 96))

    # A subtle alpha halo survives scaling but never becomes an opaque square.
    for radius, alpha in ((46, 12), (42, 20), (38, 30)):
        c.ellipse((255, 184, 53, alpha), (48 - radius, 48 - radius, radius * 2, radius * 2))

    outer = []
    inner = []
    for index in range(12):
        angle = -math.pi / 2 + index * math.tau / 12
        outer.append((48 + math.cos(angle) * 35, 48 + math.sin(angle) * 35))
        inner.append((48 + math.cos(angle) * 27, 48 + math.sin(angle) * 27))
    c.polygon((102, 53, 14), tuple((x + 3, y + 4) for x, y in outer))
    c.polygon((255, 177, 42), outer)
    c.polygon((255, 226, 108), inner)
    c.line((255, 248, 190), (27, 29), (59, 17), 3)

    # Lightning-shaped embossing keeps the token readable at thumbnail size.
    c.polygon((190, 92, 20), ((51, 24), (34, 51), (46, 50), (40, 73), (65, 42), (52, 43)))
    c.polygon((255, 252, 204), ((50, 29), (39, 47), (50, 46), (46, 62), (60, 45), (49, 47)))
    c.ellipse((255, 255, 225), (29, 26, 7, 5))
    return c.finish()


def _vertical_gradient(surface: pygame.Surface, stops: Sequence[tuple[float, Color]]) -> None:
    """Fill an opaque surface by linearly interpolating ordered color stops."""

    height = surface.get_height()
    ordered = sorted(stops, key=lambda item: item[0])
    for y in range(height):
        position = y / max(1, height - 1)
        left = ordered[0]
        right = ordered[-1]
        for first, second in zip(ordered, ordered[1:]):
            if first[0] <= position <= second[0]:
                left, right = first, second
                break
        span = max(1e-9, right[0] - left[0])
        amount = max(0.0, min(1.0, (position - left[0]) / span))
        color = tuple(
            round(left[1][channel] + (right[1][channel] - left[1][channel]) * amount)
            for channel in range(3)
        )
        pygame.draw.line(surface, color, (0, y), (surface.get_width(), y))


def _window_grid(
    surface: pygame.Surface,
    rect: pygame.Rect,
    rng: random.Random,
    color_a: Color,
    color_b: Color,
    density: float,
) -> None:
    margin_x, margin_y = 10, 16
    for y in range(rect.top + margin_y, rect.bottom - 9, 24):
        for x in range(rect.left + margin_x, rect.right - 8, 18):
            if rng.random() > density:
                continue
            color = color_a if rng.random() < 0.70 else color_b
            pygame.draw.rect(surface, color, (x, y, 6, 10), border_radius=1)


def build_skyline() -> pygame.Surface:
    """Create a layered 16:9 neon city with a clear central track horizon."""

    width, height = DELIVERY_SIZES["skyline.png"]
    surface = pygame.Surface((width, height))
    rng = random.Random(20260728)
    _vertical_gradient(
        surface,
        (
            (0.00, (9, 20, 49)),
            (0.30, (27, 59, 91)),
            (0.52, (48, 95, 111)),
            (0.67, (238, 126, 97)),
            (1.00, (21, 34, 51)),
        ),
    )

    # Long, thin atmospheric bands add depth without obscuring gameplay.
    haze = pygame.Surface((width, height), pygame.SRCALPHA)
    for rect, color in (
        ((0, 245, width, 44), (135, 221, 218, 22)),
        ((0, 332, width, 58), (255, 211, 157, 27)),
        ((0, 405, width, 80), (255, 109, 103, 18)),
    ):
        pygame.draw.ellipse(haze, color, rect)
    surface.blit(haze, (0, 0))

    # Distant city: intentionally low contrast so foreground sprites stay readable.
    x = -15
    while x < width:
        building_width = rng.randint(48, 102)
        building_height = rng.randint(120, 280)
        top = 560 - building_height
        body = rng.choice(((30, 53, 73), (35, 61, 76), (40, 55, 75)))
        pygame.draw.rect(surface, body, (x, top, building_width, building_height))
        if rng.random() < 0.35:
            pygame.draw.line(
                surface,
                (89, 217, 210),
                (x + building_width // 2, top),
                (x + building_width // 2, top - rng.randint(28, 70)),
                3,
            )
        _window_grid(
            surface,
            pygame.Rect(x, top, building_width, building_height),
            rng,
            (247, 197, 105),
            (78, 202, 211),
            0.38,
        )
        x += building_width + rng.randint(7, 18)

    # Darker towers at the sides frame the vanishing point instead of covering it.
    towers = (
        (0, 246, 245, 600, (18, 32, 54)),
        (255, 302, 180, 544, (22, 38, 60)),
        (1490, 282, 180, 564, (21, 35, 58)),
        (1680, 208, 240, 638, (15, 29, 50)),
    )
    for index, (x, y, w, h, color) in enumerate(towers):
        pygame.draw.rect(surface, color, (x, y, w, h))
        cap = [(x + 18, y), (x + w // 2, y - 48 - index * 8), (x + w - 18, y)]
        pygame.draw.polygon(surface, color, cap)
        accent = (42, 225, 213) if index % 2 == 0 else (247, 91, 106)
        pygame.draw.line(surface, accent, (x + w - 12, y + 18), (x + w - 12, y + h - 18), 5)
        _window_grid(
            surface,
            pygame.Rect(x, y, w, h),
            rng,
            (255, 197, 95),
            (65, 213, 220),
            0.50,
        )

    # Elevated transit lines and perspective light rails reinforce the setting.
    pygame.draw.rect(surface, (12, 25, 42), (0, 565, width, 37))
    pygame.draw.rect(surface, (48, 222, 209), (0, 565, width, 4))
    pygame.draw.rect(surface, (247, 96, 102), (0, 597, width, 3))
    for x in range(60, width, 155):
        pygame.draw.polygon(
            surface,
            (18, 31, 48),
            ((x, 600), (x + 25, 600), (x + 51, 1080), (x + 1, 1080)),
        )

    # The lower field is mostly neutral because metro_motion draws its own tracks.
    pygame.draw.rect(surface, (22, 35, 49), (0, 846, width, 234))
    for y, color in ((862, (48, 78, 87)), (900, (33, 55, 67)), (1010, (13, 25, 39))):
        pygame.draw.rect(surface, color, (0, y, width, 4))

    # Symmetric station lights guide the eye towards the track horizon.
    vanishing_x, vanishing_y = width // 2, 570
    for side in (-1, 1):
        for step in range(7):
            depth = step / 6
            x = vanishing_x + side * int(190 + depth * 720)
            y = int(vanishing_y + depth * 390)
            radius = 5 + int(depth * 9)
            pygame.draw.circle(surface, (35, 231, 214), (x, y), radius + 5)
            pygame.draw.circle(surface, (222, 255, 247), (x, y), radius)

    return surface


def save_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    assets = {
        "runner.png": build_runner(),
        "barrier.png": build_barrier(),
        "coin.png": build_coin(),
        "skyline.png": build_skyline(),
    }
    for filename, surface in assets.items():
        pygame.image.save(surface, ASSET_DIR / filename)


def verify_assets() -> None:
    for filename, expected_size in DELIVERY_SIZES.items():
        path = ASSET_DIR / filename
        loaded = pygame.image.load(path)
        if loaded.get_size() != expected_size:
            raise RuntimeError(f"{filename}: expected {expected_size}, got {loaded.get_size()}")
        wants_alpha = filename != "skyline.png"
        has_alpha = bool(loaded.get_flags() & pygame.SRCALPHA)
        if wants_alpha != has_alpha:
            raise RuntimeError(f"{filename}: unexpected alpha mode")
        mode = "RGBA transparent" if has_alpha else "RGB opaque"
        print(f"{filename:12} {expected_size[0]:4}x{expected_size[1]:4}  {mode}")


def main() -> None:
    pygame.init()
    try:
        save_assets()
        verify_assets()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
