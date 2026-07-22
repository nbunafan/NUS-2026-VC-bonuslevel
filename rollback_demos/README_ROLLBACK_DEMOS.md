# Rollback demos

These programs deliberately restore one legacy problem at a time for comparison footage.
They do not modify the production `danceapp.py`.

```powershell
cd "D:\document\NUS材料\phase2 VC\final project\bonuslevel"

python rollback_demos\rollback_coordinate_mismatch.py
python rollback_demos\rollback_no_dancer_lock.py
python rollback_demos\rollback_continuous_scoring.py
python rollback_demos\rollback_random_fruit.py
python rollback_demos\rollback_fixed_interval_keyframes.py
```

- `coordinate_mismatch`: scores raw normalized-image coordinates without centering or body
  scaling. Keyframe extraction remains normalized so this demo isolates the scoring defect.
- `no_dancer_lock`: selects the largest detected person independently on each inference frame.
- `continuous_scoring`: adds `10%` of the current similarity every inferred frame.
- `random_fruit`: selects left/right randomly and never converts the opposite side into a bomb.
- `fixed_interval_keyframes`: captures the current pose every `0.7 seconds`, even during a
  transition or while the dancer is stationary, without measuring motion magnitude.

Use the normal `danceapp.py` to record the improved behavior with the same camera and reference
video. The rollback windows intentionally retain the same visual layout for a fair comparison.
