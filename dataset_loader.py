"""
dataset_loader.py

Parses REHAB24-6's Segmentation.csv and resolves each repetition to its
video file, so pose_estimator.py / exercise_analyzer.py can run per-rep
against real labeled data.

Schema confirmed against Segmentation.txt (the dataset's own column docs)
and the real CSV (1072 rows, semicolon-delimited -- sep=";", not comma):

  video_id;repetition_number;exercise_id;person_id;first_frame;last_frame;
  cam17_orientation;mocap_erroneous;exercise_subtype;lights_on;
  extra_person_in_cam17;extra_person_in_cam18;correctness

Scope: only exercise_id 1 (Arm Abduction) and 6 (Squat) are in play here.

CAMERA SELECTION (per Segmentation.txt):
cam17_orientation is 'front' / 'half-profile' / 'profile' and VARIES PER
REP -- it is not constant, despite what earlier spot-checks suggested.
Camera18 sits orthogonal to Camera17, so its orientation is implied:
    cam17 'front'        -> cam18 'profile'
    cam17 'half-profile'  -> cam18 'half-profile'
    cam17 'profile'       -> cam18 'front'

Squat (knee/hip flexion) needs a profile view to be geometrically
meaningful -- a front view foreshortens the flexion angle. Arm abduction
is a frontal-plane movement and needs a front view. So this loader picks
WHICHEVER CAMERA gives the better view, per rep, rather than defaulting
to Camera17 for everything. Camera18 files are "-transposed" (rotated 90
degrees at capture) -- callers get a `rotate_90` flag telling
pose_estimator.extract_frame_range_landmarks() whether to compensate.

exercise_subtype distinguishes sides for some exercises. For exercise_id
1 (Arm Abduction) specifically: the real data is 100% "right arm" (no
"left arm" rows exist at all) -- side mapping is kept generic rather than
hardcoded in case that changes, but don't expect any left-arm reps.
Squat (exercise_id 6) has no subtype (bilateral) -- side picked by
whichever side MediaPipe tracks with higher visibility, in
exercise_analyzer.py.

mocap_erroneous and extra_person_in_cam17/18 are quality flags --
erroneous-mocap rows are dropped by default.

UNVERIFIED ASSUMPTION -- Segmentation.txt doesn't document frame-rate,
and some subjects' mocap .npy files are 120fps vs 30fps video for the
same subject. If first_frame/last_frame were authored against 120fps
mocap for some subjects, slicing the 30fps video with those numbers is
wrong by ~4x for that subject. Sanity-check one rep's last_frame against
the video's actual frame count (cv2.CAP_PROP_FRAME_COUNT) before trusting
results at scale.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

EXERCISE_FOLDERS = {1: "Ex1", 6: "Ex6"}
EXERCISE_NAMES = {1: "arm_abduction", 6: "squat"}
SUBTYPE_TO_SIDE = {"left arm": "left", "right arm": "right"}

# Desired camera orientation per exercise, best to worst.
ORIENTATION_PREFERENCE = {
    "squat": ["profile", "half-profile", "front"],
    "arm_abduction": ["front", "half-profile", "profile"],
}

CAM17_TO_CAM18_ORIENTATION = {
    "front": "profile",
    "half-profile": "half-profile",
    "profile": "front",
}


@dataclass
class RepRecord:
    video_path: str
    exercise_id: int
    exercise_name: str
    video_id: str
    person_id: str
    repetition_number: int
    first_frame: int
    last_frame: int
    correctness: int  # 1 = correct, 0 = incorrect (assumed -- verify on a known rep)
    exercise_subtype: str
    cam17_orientation: str
    side: Optional[str]  # "left"/"right" for arm abduction; None for squat (bilateral)
    camera: str          # "Camera17" or "Camera18" -- whichever best matches the view this exercise needs
    camera_orientation: str  # the orientation actually used (from the preference list)
    rotate_90: bool       # True if `camera` is Camera18 (the "-transposed" files)


def _pick_camera(exercise_name: str, cam17_orientation: str) -> tuple:
    """
    Returns (camera, orientation_used, rotate_90) -- whichever of
    Camera17/Camera18 best matches this exercise's preferred viewing
    angle, given cam17's recorded orientation and the stated
    camera17<->camera18 orientation mapping.
    """
    cam17_o = (cam17_orientation or "").strip().lower()
    cam18_o = CAM17_TO_CAM18_ORIENTATION.get(cam17_o)

    preference = ORIENTATION_PREFERENCE[exercise_name]

    def rank(orientation):
        try:
            return preference.index(orientation)
        except ValueError:
            return len(preference)  # unknown orientation -- worst rank

    cam17_rank = rank(cam17_o)
    cam18_rank = rank(cam18_o) if cam18_o else len(preference)

    if cam18_rank < cam17_rank:
        return "Camera18", cam18_o, True
    return "Camera17", cam17_o, False  # ties go to Camera17 (no rotation needed)


def load_segmentation(
    csv_path: str,
    videos_root: str,
    fps_suffix: str = "30fps",
    drop_erroneous_mocap: bool = True,
    drop_multi_person: bool = True,
    camera_override: Optional[str] = None,
) -> list[RepRecord]:
    """
    Read Segmentation.csv, filter to the project's 2-exercise scope, and
    resolve each repetition to its real video file path -- auto-picking
    Camera17 or Camera18 per rep based on which gives the better view
    for that exercise (see module docstring).

    camera_override: force "Camera17" or "Camera18" for every rep instead
    of auto-selecting (mainly for debugging / comparing against the
    naive approach).
    """
    df = pd.read_csv(csv_path, sep=";")

    missing = {
        "video_id", "person_id", "repetition_number", "exercise_id",
        "first_frame", "last_frame", "correctness", "cam17_orientation",
    } - set(df.columns)
    if missing:
        raise ValueError(
            f"Segmentation.csv is missing expected columns: {missing}. "
            f"Got: {df.columns.tolist()}. Did the schema change, or is sep wrong?"
        )

    df = df[df["exercise_id"].isin(EXERCISE_FOLDERS.keys())]

    if drop_erroneous_mocap and "mocap_erroneous" in df.columns:
        df = df[df["mocap_erroneous"] == 0]

    if drop_multi_person:
        for col in ("extra_person_in_cam17", "extra_person_in_cam18"):
            if col in df.columns:
                df = df[df[col] == 0]

    records = []
    skipped_unknown_side = 0
    for _, row in df.iterrows():
        ex_id = int(row["exercise_id"])
        ex_name = EXERCISE_NAMES[ex_id]
        folder = EXERCISE_FOLDERS[ex_id]
        video_id = str(row["video_id"])
        cam17_orientation = str(row.get("cam17_orientation", ""))

        if camera_override:
            camera = camera_override
            orientation_used = cam17_orientation if camera == "Camera17" else \
                CAM17_TO_CAM18_ORIENTATION.get(cam17_orientation.strip().lower(), "")
            rotate_90 = camera == "Camera18"
        else:
            camera, orientation_used, rotate_90 = _pick_camera(ex_name, cam17_orientation)

        suffix = "-transposed" if camera == "Camera18" else ""
        filename = f"{video_id}-{camera}-{fps_suffix}{suffix}.mp4"
        video_path = str(Path(videos_root) / folder / filename)

        subtype = str(row.get("exercise_subtype", "")).strip().lower()
        side = None
        if ex_id == 1:  # arm abduction -- single-arm, need the side
            side = SUBTYPE_TO_SIDE.get(subtype)
            if side is None:
                skipped_unknown_side += 1
                continue  # can't score an arm-abduction rep without knowing which arm

        records.append(
            RepRecord(
                video_path=video_path,
                exercise_id=ex_id,
                exercise_name=ex_name,
                video_id=video_id,
                person_id=str(row["person_id"]),
                repetition_number=int(row["repetition_number"]),
                first_frame=int(row["first_frame"]),
                last_frame=int(row["last_frame"]),
                correctness=int(row["correctness"]),
                exercise_subtype=str(row.get("exercise_subtype", "")),
                cam17_orientation=cam17_orientation,
                side=side,
                camera=camera,
                camera_orientation=orientation_used,
                rotate_90=rotate_90,
            )
        )

    if skipped_unknown_side:
        print(
            f"[dataset_loader] Warning: skipped {skipped_unknown_side} arm-abduction "
            f"rows with an unrecognized exercise_subtype (not 'left arm'/'right arm')."
        )

    return records


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python dataset_loader.py <Segmentation.csv> <videos_root>")
        sys.exit(1)

    recs = load_segmentation(sys.argv[1], sys.argv[2])
    print(f"Loaded {len(recs)} reps (exercise_id in {{1, 6}}) after filtering.")
    by_ex = {}
    for r in recs:
        by_ex.setdefault(r.exercise_name, []).append(r)
    for name, items in by_ex.items():
        correct = sum(1 for r in items if r.correctness == 1)
        cam_counts = {}
        for r in items:
            cam_counts[r.camera] = cam_counts.get(r.camera, 0) + 1
        print(f"  {name}: {len(items)} reps, {correct} correct / {len(items) - correct} incorrect, camera split: {cam_counts}")
    if recs:
        print("Example record:", recs[0])
