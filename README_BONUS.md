# Bonus Level

This directory contains the complete pose-controlled dance and runner work.

## Main programs

```powershell
cd "D:\document\NUS材料\phase2 VC\final project\bonuslevel"

# Just Dance scoring, keyframe preview, Fruit Ninja, and video audio
python danceapp.py

# Pose extraction and stickman scoring experiment
python stickman_game.py

# Pose-controlled three-lane runner
python bonus_runner\metro_motion.py --camera 0
```

For a phone camera stream:

```powershell
python bonus_runner\metro_motion.py --camera "http://PHONE_IP:8080/video"
```

## Dependencies

- `pose_utils.py`: shared pose representation and `MainDancerTracker`.
- `danceapp.py`: analyses reference poses once in five-second groups and reuses the cached
  group plan during playback.
- `yolov8n-pose.pt`: model used by all Bonus entry points.
- `models/yolov8n-pose.pt`: compatibility copy for the default `PoseEstimator` path.
- `dance_example_1.mp4`: default dance reference video.
- `bonus_runner/assets`: runner sprites and background.
- `.audio_cache`: extracted reference-video audio cache.

The project-wide `requirements.txt`, run guide, and defense guide are copied here so this
directory can be presented or moved independently without changing the other project levels.
