# SWS3026 Visual Computing Project

This folder now contains runnable code for the beginner, expert, and bonus tracks.

## Environment

Run all commands from this project folder. OpenCV may fail to load XML/YAML models from an absolute path containing non-ASCII characters, so changing into this directory first is important.

```powershell
cd "D:\document\NUS材料\phase2 VC\final project"
python -m pip install -r requirements.txt
```

If Ultralytics cannot write its user cache, use:

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"
```

## Beginner Level

Webcam display only:

```powershell
python starter.py
```

Webcam with the original and improved pipelines in one window:

```powershell
python face_keypoints_webcam.py
```

The program opens in side-by-side mode. The left panel is the original Haar + LBF pipeline;
the right panel is the improved hybrid pipeline. Press `1` for original only, `2` for improved
only, `3` for side-by-side, or `Space` to cycle through the modes. Press `q` to quit. Running
this command does not launch `face_detector_comparison.py`.

To compare the three face detectors on identical webcam frames:

```powershell
python face_detector_comparison.py
```

Use `1` for the normal-face test, `2` for the mask test, and `3` while covering one eye.
The letter shortcuts `n`, `m`, and `e` remain available.
Changing condition or pressing `s` writes detection rate, landmark success rate, detector
latency, and total pipeline latency to `outputs/face_detector_comparison.csv`. Press `r` to
reset the current sample and `q` to quit.

The comparison intentionally uses the same OpenCV LBF 68-point estimator after every face
detector. This isolates the effect of Haar, MediaPipe, and Dlib instead of mixing detector
quality with different landmark models. MediaPipe is expected to be more robust to partial
occlusion because its learned BlazeFace model combines several facial cues. Dlib HOG uses a
different gradient representation and is useful as an independent fallback, although it can
still struggle with strong pose changes or large occlusions. No detector is guaranteed to
recognise every masked face, so the hybrid mode combines detector diversity with a short
temporal hold rather than relying on one method alone.

On Windows with Python 3.12, the project uses `dlib-bin`, a precompiled distribution that
provides the normal `dlib` Python module. MediaPipe is pinned to `0.10.21` because later
releases remove the Solutions API used by `FaceDetection`. NumPy and OpenCV are also pinned
to compatible versions because MediaPipe 0.10.21 requires NumPy 1.x, while the newest OpenCV
5 package requires NumPy 2.x.

To evaluate one classifier independently (the file shows its complete configuration):

```powershell
python model_comparison_suite\individual\lightgbm_leaf31.py
python model_comparison_suite\individual\random_forest.py
python model_comparison_suite\individual\mlp.py
python model_comparison_suite\individual\ensemble_lgbm_rf.py
```

To aggregate all landmark classifiers, ensembles, augmented models, pretrained FER and
ViT references into one comparison folder:

```powershell
python model_comparison_suite\aggregate_results.py
```

The consolidated CSV/JSON report, charts, confusion matrices and per-class reports are
stored under `model_comparison_suite\results`. See `model_comparison_suite\README.md`
for the complete individual-model list.

## Expert Level

Train and evaluate the expression classifier from 478 MediaPipe Face Mesh landmarks:

```powershell
python 478points\train_expression_classifier.py --train-per-class 1200 --test-per-class 300 --pose-copies 1
```

For a slower, fuller run over the whole FER split:

```powershell
python 478points\train_expression_classifier.py --train-per-class 0 --test-per-class 0 --pose-copies 1
```

Outputs are written to:

- `478points/outputs/expression_landmark_classifier.joblib`
- `478points/outputs/metrics.json`
- `478points/outputs/classification_report.txt`
- `478points/outputs/confusion_matrix.csv`
- `478points/outputs/confusion_matrix.png`

Current measured accuracy on the full FER-2013 split:

```text
accuracy = 0.5595
macro F1 = 0.5233
test samples = 6853
prediction latency = 0.182 ms
```

Run webcam expression prediction and effects:

```powershell
python 478points\expression_effects_webcam.py
```

The webcam currently runs in recognition-only mode: it shows the face box, a readable subset
of the 478 detected landmarks, and
the predicted expression in the top-left corner. Confidence values, timing values, and visual
effects are hidden so classification behaviour can be checked without distractions.

Face Mesh jointly detects the face and estimates a dense 3-D mesh. This removes the previous
LBF dependency from the Expert webcam path and is more tolerant of moderate head rotation.
Synthetic yaw and pitch augmentation is applied to landmark coordinates during training.

The program requests 1280x720 MJPG capture at 30 FPS, enables autofocus when supported, and
applies mild display-only sharpening. You can request Full HD or disable sharpening with:

```powershell
python 478points\expression_effects_webcam.py --width 1920 --height 1080 --fps 30
python 478points\expression_effects_webcam.py --no-sharpen
```

## Bonus Level

Run the enhanced Just Dance style application:

```powershell
python danceapp.py
```

The optimized UI refreshes on Tk's main thread at 30 Hz while capture and pose inference run
in workers. Reference-video inference is sampled at about 8 FPS and webcam inference every
third frame; the most recent skeleton is reused between inference frames. A shared model lock
prevents concurrent YOLO calls from competing for CPU/GPU resources. `YOLO_CONFIG_DIR` is set
inside the program before Ultralytics is imported, so no PowerShell environment command is
required.

The app uses:

- YOLOv8 pose keypoints for the reference video and webcam.
- Largest-person selection when multiple people appear.
- Translation/scale-normalized keypoint comparison.
- A short reference pose buffer for simple lag compensation.
- Per-frame score, average score, combo, and feedback labels.
- A small wrist-hit bonus activity where orange circles can be sliced for extra points.

The original starter GUI is preserved as:

```powershell
python danceapp_starter.py
```

## Notes For Presentation

Facial expression recognition from landmarks is fast and real-time friendly, but it discards texture information such as wrinkles, teeth, shadows, and eye details. In FER-2013, some classes are visually ambiguous even with full images; using only landmark geometry makes fear/sad/neutral especially difficult. Normalizing landmarks around the face center improves robustness to translation and scale, while class-balanced tree ensembles help with the dataset imbalance.

For the dance task, the score is based on normalized body pose distance rather than raw pixel distance, so the player can stand at a different size or position from the reference dancer. The reference buffer compares the current webcam pose against several recent reference poses, which handles small human reaction delays.
