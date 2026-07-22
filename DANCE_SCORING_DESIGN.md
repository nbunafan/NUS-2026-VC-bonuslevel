# Dance Motion Lab Scoring Design

## Goals

The score should reward important dance poses, allow reaction time, and keep Fruit Ninja fun
without letting random fruit placement force the player to perform the wrong dance movement.
It therefore uses two independent point channels: **Dance** and **Fruit**.

## 1. Pose normalization and similarity

Shoulders, elbows, wrists, hips, knees, and ankles are translated to their joint center and
divided by shoulder/hip width. This removes camera position and player size. For all mutually
visible joints, mean Euclidean distance `d` is converted to similarity:

```text
similarity = 100 * exp(-1.35 * d)
```

## 2. A 2.5-second action preview

Every sampled reference pose is scheduled with:

```text
due_time = reference_display_time + 2.5 seconds
```

The player sees the reference pose and the compact `NEXT MOVE` skeleton first. The camera pose
is compared with it only when its due time enters a +/-0.45 second window. This is equivalent
to giving the player 2.5 seconds to react, without claiming that a statistical model predicts
an unseen pose.

The reference panel always displays the original video. Extracted future keyframes appear in a
separate cue lane at the lower right. New cards enter from the right and move toward the `NOW`
line on the left; crossing that line opens the scoring window.

## 3. Sparse keyframe scoring

Reference-pose motion is measured as normalized joint displacement from the preceding sampled
pose. A motion peak above `0.16`, separated from the previous peak by at least `0.7 s`, becomes
a keyframe. This prevents the same action from being scored repeatedly.

At a due keyframe:

```text
displayed score = 0.80 * keyframe similarity + 0.20 * continuity similarity
dance points += keyframe similarity
```

If keyframe similarity is at least 70, dance combo increases and awards up to 20 additional
points. Between keyframes, continuity contributes only `2.5%` of its similarity per sampled
camera pose for feedback only and does not add points. Important poses therefore fully
determine the Dance total; matching every ordinary frame cannot inflate the score.

## 4. Fruit placement and movement conflict

The upcoming reference pose determines which wrist is extended furthest from the shoulder
center. Fruit has a 75% probability of appearing on that side. This supports, rather than
contradicts, the expected dance gesture.

An object spawned on the opposite side is displayed as a bomb. It is intentionally unsafe to
slice because chasing it would move the player away from the expected choreography.

```text
correct-side fruit = 20 base points with Fruit Combo
opposite-side bomb = -(25 + 5 * min(current combo, 5)), capped at -50
```

Ignoring a bomb has no penalty. Slicing one resets Fruit Combo and subtracts up to 50 Fruit
points, with a zero floor. Bombs never reduce Dance points or reset Dance Combo.

## 5. Fruit combo

A hit within 2.2 seconds of the preceding hit continues Fruit Combo:

```text
multiplier = 1 + 0.30 * min(combo - 1, 8)
fruit points = round(20 * multiplier)
```

This rises from 20 points to a maximum of 68 points per fruit. Multiple consecutive
fruit can therefore exceed one dance keyframe, as intended, but missing an expired fruit resets
only Fruit Combo. It does not reset Dance Combo or remove Dance points.

## 6. Displayed totals

The header displays cumulative `Dance`, cumulative `Fruit`, and current Fruit Combo separately.
This makes the scoring explainable during presentation and prevents a large fruit bonus from
being mistaken for pose-recognition accuracy.

Keyframe results also trigger a short visual burst on the player view:

```text
90-100  PERFECT
80-89   AMAZING
65-79   NICE
50-64   GOOD
0-49    KEEP GOING
```

The burst uses outlined text, a short scale pulse, radial rays, and fading particles. It is
triggered only once per keyframe, so continuous-frame scoring cannot cause distracting flicker.
