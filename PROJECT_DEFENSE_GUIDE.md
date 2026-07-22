# Visual Computing Project: Code Defense Guide

## 1. Project Architecture

The project deliberately keeps alternative methods separate because they answer different
questions in the assignment:

- `478points`: active Expert Level solution. It classifies expressions from 478 MediaPipe
  Face Mesh landmarks and deploys the best RF+MLP ensemble.
- `68points`: archived 68-point LBF experiments and the full classical-model comparison.
- Root Beginner files: face detection and landmark robustness comparisons.
- Root pretrained-model files: image-based FER and ViT baselines. They are comparisons, not
  the official keypoint-only Expert solution.
- `bonus_runner`: pose-controlled three-lane endless runner.

## 2. Active 478-Point Expert Pipeline

### `478points/high_density_face_mesh.py`

This is the shared feature contract. `HighDensityFaceMesh.extract` converts BGR pixels to a
478-point mesh. `normalize_mesh` centers the mesh on the nose, divides by eye distance, and
removes roll. It retains z coordinates and therefore preserves useful yaw information.
`augment_pose` rotates training clouds around yaw and pitch axes, addressing FER-2013's
frontal-face bias without giving image pixels to the classifier.

### `478points/train_expression_classifier.py`

This trains the single LightGBM baseline. FER images are enhanced and enlarged only to help
MediaPipe estimate landmarks. Cached 1,434-dimensional keypoint vectors are the classifier's
actual input. Training augmentation is never applied to the held-out test set. The script
exports accuracy, macro-F1, latency, a class report, and a confusion matrix.

### `478points/train_six_478_classifiers.py`

This compares LightGBM, Random Forest, ExtraTrees, MLP, LightGBM+RF, and RF+MLP on identical
cached samples. Checkpoints make the experiment resumable. Equal soft voting is fixed without
optimizing weights on test labels. The highest result is RF+MLP at 57.97% accuracy.

### `478points/expression_effects_webcam.py`

This is the deployed program. One mesh is converted to one feature row; RF and MLP probabilities
are averaged; six recent probability vectors are averaged to reduce flicker. RF uses one
inference thread because batch-of-one latency (about 15.5 ms) is lower than multithreaded
latency and meets the assignment's 30 ms classifier requirement.

## 3. Beginner Level Files

- `starter.py`: minimal webcam entry point supplied/retained for the first task.
- `face_keypoints_webcam.py`: displays original and improved face pipelines in one window.
- `face_detector_comparison.py`: applies Haar, MediaPipe, and Dlib to identical frames so
  detection-rate and latency comparisons are fair.
- `robust_face_detectors.py`: detector adapters, box clipping, hybrid priority selection, and
  short temporal box holding. MediaPipe is preferred; Dlib and Haar are independent fallbacks.
- `vc_face_utils.py`: shared 68-point data structures, LBF loading, landmark normalization,
  legacy feature extraction, result selection, and drawing.
- `beginner_fer_comparison.py`: compares the Beginner keypoint pipeline with pretrained FER.

The important defense distinction is detection versus landmark fitting. A detector may find a
side face while the old LBF fitter still fails. This motivated the Expert pipeline's direct
Face Mesh replacement.

## 4. Archived 68-Point Expert Experiments

### Shared training code

- `68points/expression_modeling.py`: converts 68 coordinates into aligned coordinates,
  regional distances, anchor distances, and contour deltas; also defines legacy ensembles.
- `68points/train_expression_classifier_augmented.py`: extracts LBF landmarks, preprocesses
  FER crops, augments landmarks, compares candidates, and writes failure analysis.
- `68points/augmented_classifier_common.py`: shared cache loading, evaluation, reports, and
  the RF+MLP probability wrapper.
- `68points/train_augmented_random_forest.py`: Random Forest experiment.
- `68points/train_augmented_mlp.py`: standardized multilayer perceptron experiment.
- `68points/train_augmented_random_forest_mlp.py`: mixed RF+MLP experiment.
- `68points/expression_effects_webcam_augmented.py`: legacy augmented-model webcam launcher.
- `68points/expert_fer_comparison.py`: side-by-side keypoint classifier versus image FER.

### Comparison suite

- `comparison_common.py`: owns fixed train/test loading, evaluation metrics, model persistence,
  latency measurement, and per-model output paths.
- `aggregate_results.py`: reads all saved metrics and creates ranked CSV/JSON/Markdown reports,
  accuracy and latency charts, and combined confusion-matrix figures.
- `individual/_bootstrap.py`: adds suite and project directories to Python's import path.
- `individual/deployed_lightgbm_full.py`: records the full-data deployed LightGBM result.
- `individual/ensemble_extra_hist.py`: ExtraTrees + histogram boosting soft vote.
- `individual/ensemble_lgbm_rf.py`: LightGBM + Random Forest soft vote.
- `individual/ensemble_lightgbm_pair.py`: two LightGBM configurations combined.

The remaining `individual` files each contain one estimator configuration and call the common
runner. Their filenames identify the changed algorithm or hyperparameter:

- `calibrated_linear_svm.py`: linear SVM probabilities obtained through calibration.
- `extra_trees_raw.py`: ExtraTrees on the original landmark representation.
- `extra_trees_enhanced.py`: ExtraTrees on engineered geometry.
- `extra_trees_f40.py`: ExtraTrees with a 40% feature subsample.
- `extra_trees_leaf1.py`: ExtraTrees allowing one sample per leaf.
- `hist_gradient_boosting.py`: baseline histogram gradient boosting.
- `hist_gradient_leaf15.py`, `hist_gradient_leaf63.py`: leaf-size variants.
- `lightgbm_leaf31.py`, `lightgbm_leaf63.py`: LightGBM leaf-count variants.
- `linear_discriminant.py`: Linear Discriminant Analysis baseline.
- `logistic_regression.py`: multinomial linear baseline.
- `mlp.py`: neural-network baseline.
- `random_forest.py`: bagged decision-tree baseline.
- `rbf_svm.py`: nonlinear radial-basis SVM baseline.

## 5. Pretrained Image Baselines

- `pretrained_fer_common.py`: loads the pretrained FER CNN and standardizes its class output.
- `evaluate_pretrained_fer.py`: evaluates FER on the FER-2013 test folders and writes metrics.
- `vit_expression_common.py`: locally loads `trpakov/vit-face-expression` through Transformers.
- `evaluate_vit_face_expression.py`: batches test images through ViT and creates its report.
- `README_PRETRAINED_FER.md` and requirements files document optional dependencies.

FER and ViT receive cropped face pixels, so they are valid external baselines but not compliant
replacements for the Expert instruction to classify from keypoints.

## 6. Bonus Runner

- `bonus_runner/pose_controller.py`: converts MediaPipe body landmarks into calibrated jump,
  crouch, left, and right events. Thresholds are relative to body scale rather than pixels.
- `bonus_runner/metro_motion.py`: game loop, camera worker, split-screen rendering, lane state,
  obstacle/coin updates, collision logic, score, and pose-command integration.
- `bonus_runner/create_assets.py`: generates the project's original 2-D placeholder artwork.
- `bonus_runner/test_pose_controller.py`: synthetic landmark tests for gesture state transitions.
- `bonus_runner/__init__.py`: marks the directory as an importable Python package.

## 7. Dance Reference Application

- `danceapp_starter.py`: minimal provided dance interface and baseline pose comparison.
- `danceapp.py`: completed pose extraction, normalization, recent-reference buffering, scoring,
  and GUI integration.
- `danceapp_enhanced.py`: retained enhanced variant; currently substantially duplicates
  `danceapp.py`, so describe it as an experimental copy rather than a separate algorithm.

The score compares normalized skeleton geometry, not absolute pixels. A recent reference-pose
buffer compensates for small human reaction delays by selecting the best nearby alignment.

## 8. Results to Quote

- 478-point extraction rate on the test split: 95.47%.
- Single LightGBM accuracy: 55.95%.
- RF+MLP accuracy: 57.97%; macro-F1: 54.11%.
- RF+MLP classifier latency: approximately 15.5 ms for one frame.
- The 30 ms requirement concerns classifier prediction; camera capture, Face Mesh, and display
  are separate pipeline costs and should be reported separately when profiling end-to-end FPS.

## 9. Common Defense Questions

**Why use more than 68 points?** Dense Face Mesh represents more mouth, eye, cheek, and 3-D
shape detail and avoids the old separate frontal LBF fitting stage.

**Why normalize?** Translation, face size, and roll are nuisance variables. Removing them lets
the classifier focus on expression geometry.

**Why is accuracy not comparable to image ViT?** ViT uses texture, wrinkles, teeth, and shading;
the Expert model is intentionally restricted to keypoint geometry.

**Why use an ensemble?** RF and MLP make different errors. Averaging calibrated class
probabilities improved held-out accuracy without increasing classifier latency beyond 30 ms.

**Why keep 68-point results?** They provide a controlled historical baseline demonstrating the
effect of changing landmark representation while retaining classical classifiers.
