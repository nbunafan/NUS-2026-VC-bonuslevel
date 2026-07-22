# Metro Motion Bonus Prototype

This is an original three-lane runner inspired by the interaction pattern of endless
runner games. It does not copy Subway Surfers artwork, names, characters or source code.

The left 60% of the window is the game. The right 40% displays the player and the YOLOv8
pose skeleton. The pose model runs in a worker thread so game rendering stays responsive.

## Controls

| Player movement | Game action |
|---|---|
| Jump upward | Jump over a yellow barrier |
| Extend/wave left wrist beyond left shoulder | Move one lane left |
| Extend/wave right wrist beyond right shoulder | Move one lane right |
| Crouch with bent knees | Roll under a red overhead obstacle |

Stand neutrally and keep the full body visible during the initial calibration. Press `C`
to recalibrate after moving the phone. Keyboard fallback: arrows or WASD, Space to jump,
`R` to restart and Esc to exit.

## Run with a local or DroidCam camera

Install the one Bonus-specific dependency once:

```powershell
python -m pip install -r bonus_runner\requirements_bonus.txt
```

```powershell
cd "D:\document\NUS材料\phase2 VC\final project"
python bonus_runner\metro_motion.py --camera 0
```

If DroidCam appears as a second Windows camera, use `--camera 1` or `--camera 2`.

## Run with a phone IP camera

Connect the phone and computer to the same Wi-Fi network, start an IP camera app, and use
its MJPEG/video URL. A common Android IP Webcam URL is:

```powershell
python bonus_runner\metro_motion.py --camera "http://192.168.1.8:8080/video"
```

Use the exact URL shown by the phone app. For an unmirrored rear camera add `--no-mirror`.
On a slower laptop, reduce inference cost with `--imgsz 320`.

## Architecture

- `pose_controller.py`: scale-normalized gesture rules, calibration, edge triggers and cooldowns.
- `metro_motion.py`: phone/webcam worker, YOLO inference, split-screen UI and runner game.
- `create_assets.py`: deterministic original bitmap assets used by the prototype.

## Next phase

The next phase should add a calibration screen with adjustable thresholds, action logging,
obstacle-specific tutorials, sound, better animation, a difficulty curve and a short Bonus
Level presentation report with latency and recognition tests.
