from __future__ import annotations

"""Generate original 2-D runner placeholder artwork."""

from pathlib import Path

from PIL import Image, ImageDraw


ASSET_DIR = Path(__file__).resolve().parent / "assets"


def runner() -> None:
    image = Image.new("RGBA", (160, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((52, 8, 108, 64), fill=(252, 196, 95), outline=(28, 30, 36), width=5)
    draw.rounded_rectangle((40, 58, 120, 148), radius=18, fill=(35, 197, 155), outline=(28, 30, 36), width=6)
    draw.line((56, 76, 20, 126), fill=(252, 196, 95), width=18)
    draw.line((104, 76, 140, 126), fill=(252, 196, 95), width=18)
    draw.line((60, 142, 42, 210), fill=(45, 48, 58), width=22)
    draw.line((100, 142, 118, 210), fill=(45, 48, 58), width=22)
    image.save(ASSET_DIR / "runner.png")


def barrier() -> None:
    image = Image.new("RGBA", (220, 150), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 20, 212, 130), radius=12, fill=(245, 181, 52), outline=(25, 28, 34), width=7)
    for x in range(-40, 240, 54):
        draw.polygon([(x, 130), (x + 28, 130), (x + 90, 20), (x + 62, 20)], fill=(35, 39, 48))
    image.save(ASSET_DIR / "barrier.png")


def coin() -> None:
    image = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 88, 88), fill=(250, 194, 45), outline=(255, 237, 140), width=8)
    draw.ellipse((28, 20, 50, 72), fill=(255, 232, 122))
    image.save(ASSET_DIR / "coin.png")


def skyline() -> None:
    image = Image.new("RGB", (1000, 700), (111, 196, 214))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 430, 1000, 700), fill=(67, 76, 83))
    colors = [(54, 75, 85), (75, 92, 98), (45, 65, 76), (87, 104, 106)]
    for index, x in enumerate(range(0, 1000, 105)):
        height = 170 + (index * 47) % 170
        draw.rectangle((x, 430 - height, x + 82, 430), fill=colors[index % len(colors)])
        for row in range(430 - height + 24, 414, 38):
            for col in range(x + 16, x + 70, 28):
                draw.rectangle((col, row, col + 11, row + 17), fill=(244, 220, 133))
    image.save(ASSET_DIR / "skyline.png", quality=92)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    runner()
    barrier()
    coin()
    skyline()
    print(f"Created game assets in {ASSET_DIR}")


if __name__ == "__main__":
    main()
