# Metro Motion Bonus Runner

This is an original three-lane runner inspired by the interaction pattern of endless
runner games. It does not copy Subway Surfers artwork, names, characters or source code.

The left 60% of the window is the game. The right 40% displays the player and the YOLOv8
pose skeleton. The pose model runs in a worker thread so game rendering stays responsive.

## Pose controls

| Player movement | Game action |
|---|---|
| Raise both wrists above their corresponding shoulders | Jump over a barrier or collect a blue coin |
| Extend/wave left wrist beyond left shoulder | Move one lane left |
| Extend/wave right wrist beyond right shoulder | Move one lane right |

The jump gesture fires on the first reliable pose frame. Holding both hands above the
shoulders continues to request jumps, so the next jump can begin after landing without
lowering the arms first. Jump recognition has priority over left/right gestures, so a wide
two-hand pose cannot accidentally change lane.

Initial body-scale calibration uses 18 frames. Jump wrist confidence accepts `0.25` while
lane gestures retain the stricter threshold, balancing response time and false triggers.

Keyboard fallback: arrows or WASD, Space to jump, `P` to pause, `M` to mute sound,
`C` to recalibrate, `R` to reset and Esc to exit.

## Game rules

- Score increases gradually with distance; running speed is based on distance rather than
  score, so a valuable pickup cannot cause a sudden difficulty spike.
- Coins and correctly jumped barriers extend a four-second combo. The multiplier becomes
  `x2` at 5 combo, `x3` at 10 and `x4` at 20.
- A successful barrier jump awards a discrete `PERFECT DODGE` bonus. A collision costs 40
  points, clears the combo and grants 1.15 seconds of protection from repeated hits.
- Objectives rotate between five pickups, three barrier clears and 250 metres. Each
  completed objective grants a separate score bonus.
- The pace moves through `CRUISE`, `FLOW`, `RUSH` and `HYPER`. Later waves may contain two
  objects, but at least one safe lane always remains.

## NUS milestone

- Ground coins are orange and award 1 coin value plus 25 score.
- Elevated blue coins require a jump and award 3 coin value plus 75 score.
- Collect three orange pickups and three blue pickups in one run to unlock the prominent
  animated `N U S` milestone. The short 2.8-second celebration pauses hazards while pose
  capture continues, then resumes the same run automatically.

Stand neutrally and keep the upper body and hips visible during initial calibration. Press
`C` after moving the phone or changing the camera position.

## Run with a local or DroidCam camera

Install the runner dependencies once:

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
The default inference size is `320` for lower latency on a laptop. Use `--imgsz 256` on a
slower CPU, or `--imgsz 416` when recognition detail is more important than response time.

## Architecture

- `pose_controller.py`: body-scale calibration, two-hand jump state machine and edge-triggered
  lane gestures.
- `metro_motion.py`: camera worker, YOLO inference, pure game rules, procedural sound effects,
  split-screen UI, particles and perspective rendering.
- `create_assets.py`: deterministic 4x supersampled original character, barrier, coin and city
  assets. Run `python bonus_runner\create_assets.py` to rebuild them.
- `render_preview.py`: deterministic UI renderer that does not open YOLO or a camera.

## Preview and tests

```powershell
cd "D:\document\NUS材料\phase2 VC\final project\bonus_runner"
python render_preview.py metro_motion_preview.png
python render_preview.py metro_motion_start_preview.png --start-screen
python -m unittest -v test_pose_controller.py test_metro_lifecycle.py
```
