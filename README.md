# RQI-Net — Day 1 Log

## Environment (verified working, pinned)

```
mediapipe==0.10.21
numpy<2
pandas
matplotlib
scikit-learn
```

`pip install -r requirements.txt` into a fresh venv installs clean with no
dependency conflicts and `mp.solutions.pose` (the legacy, simpler API) works.

**Why pinned:** current mediapipe (1.0.x) removed `mp.solutions.pose`
entirely — only the new Tasks API remains, which needs a separately
downloaded `.task` model file and a different call pattern. If the spec's
`pose_estimator.py`-style code is meant to run as-is, 0.10.21 is the version
to use. Upgrading to 1.0.x later means rewriting `pose_estimator.py` against
`mediapipe.tasks.python.vision.PoseLandmarker`.

Also: `opencv-python` isn't installed directly — mediapipe pulls in
`opencv-contrib-python` as a dependency, which provides `cv2` already.
Adding `opencv-python` on top reintroduces a numpy>=2 conflict, so it's
deliberately left out of requirements.txt.

## Dataset: REHAB24-6 — findings

- 65 recordings, 10 subjects, 6 rehab exercises, correct + incorrect reps,
  supervised by a physiotherapist.
- Captured with 2 RGB cameras + a 16-camera motion-capture rig →
  RGB video, 2D **and** 3D skeleton ground truth, temporal segmentation of
  1,072 individual repetitions, binary correctness labels.
- Published by Masaryk University (Fakulta informatiky), tied to their
  "VisioTherapy" project. Paper: Černek, Sedmidubský, Budíková, *REHAB24-6:
  Physical Therapy Dataset for Analyzing Pose Estimation Methods*, SISAP 2025.
- **No public direct-download link surfaced in search.** The dataset's DOI
  record exists but doesn't expose a file link from what's indexed. Next
  step: check the SISAP/Springer chapter and the Information Systems
  journal article (ScienceDirect, DOI 10.1016/j.is.2025.102579) for a data
  availability statement, or email the authors (Sedmidubský / Budíková,
  Faculty of Informatics, Masaryk University) directly.
- One secondary paper (Tang et al. 2025, using REHAB24-6 for an unrelated
  LLM-feedback study) describes it as IMU-based with 26 joints — this
  contradicts the dataset's own publisher page (RGB + mocap skeletons) and
  looks like an error in that paper. Going with the primary source.

### Exercise-list mismatch — resolved

REHAB24-6's 6 exercises are: Arm Abduction, Arm VW (a V-then-W arm shape
drill), (Inclined) Push-ups, Leg Abduction, Leg Lunge, Squats. No equivalent
of Arm Flexion (forward arm raise) exists in the dataset.

**Decision: scope narrowed to 2 exercises — Squat + Arm Abduction.** Arm
Flexion dropped. `exercise_analyzer.py`, `rqi.py`, and `feedback.py` are
built against this 2-exercise scope from here on.

## Segmentation.csv — confirmed schema (Day 2)

Semicolon-delimited (`sep=";"`, not the pandas default comma):

```
video_id;repetition_number;exercise_id;person_id;first_frame;last_frame;
cam17_orientation;mocap_erroneous;exercise_subtype;lights_on;
extra_person_in_cam17;extra_person_in_cam18;correctness
```

1072 rows. `exercise_id` 1 = Arm Abduction, 6 = Squat (matches the `Ex1`/
`Ex6` video folders). `exercise_subtype` for arm abduction is `"left arm"`
/ `"right arm"` — **reps are single-arm, not bilateral** — so
`exercise_analyzer.analyze_arm_abduction()` now takes a `side` param.
Squat has no side label (bilateral), so `analyze_squat()` auto-picks
whichever side MediaPipe tracked with higher landmark visibility.

`mocap_erroneous` and `extra_person_in_cam17/18` are quality flags —
`dataset_loader.py` drops erroneous-mocap rows by default.

**Unverified, worth checking before trusting results at scale**: some
subjects' mocap `.npy` files are 120fps, others 30fps, but all videos
seen so far are 30fps. If `first_frame`/`last_frame` were authored
against 120fps mocap for some subjects, slicing the 30fps video with
those numbers would be off by ~4x for that subject. Sanity-check one
rep's `last_frame` against the video's actual frame count
(`cv2.CAP_PROP_FRAME_COUNT`) before running the full extraction.

## New modules (Day 2)

- `dataset_loader.py` — parses `Segmentation.csv`, filters to the
  2-exercise scope, drops erroneous-mocap reps, resolves each rep to its
  video path + frame range + (for arm abduction) which side.
- `pose_estimator.extract_frame_range_landmarks()` — pulls just one
  repetition's frames out of an untrimmed video via `first_frame`/
  `last_frame`; `rotate_90=True` for the Camera18 "-transposed" files.
- `run_extraction.py` — runs the full pipeline (loader → pose extraction
  → `exercise_analyzer`) over every rep, writes `results/rep_metrics.csv`
  with computed scores next to the ground-truth `correctness` label.
  Resumable (skips rows already in the output file on re-run). Only
  tested against a synthetic CSV matching the real schema + missing
  video files so far (confirms the logic and error-handling don't
  crash) — **not yet run against real REHAB24-6 video.** Run with
  `--limit 20` first on Kaggle to confirm it works end-to-end before
  the full ~300+ rep run.
