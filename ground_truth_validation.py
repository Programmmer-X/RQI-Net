"""
ground_truth_validation.py

Validates the spec's hardcoded angle thresholds against REHAB24-6's
ground-truth 3D mocap skeletons, bypassing MediaPipe entirely. This
answers one question: do these target ranges actually separate
correct from incorrect repetitions in real data?

    Squat:          knee 70-100 deg, hip 60-120 deg
    Arm Abduction:  elevation 160-180 deg, elbow 170-180 deg

RUN THIS ON KAGGLE (needs the REHAB24-6 npy files). Paths below assume
the same layout as videos/Ex{n}/... — adjust DATA_ROOT and the npy
path pattern if the real layout differs; I haven't verified the exact
3d_joints/ subfolder structure myself.

Joint-angle convention matches angle_calculator.joint_angle, just fed
3D (x, y, z) mocap points instead of 2D MediaPipe landmarks.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from angle_calculator import joint_angle

DATA_ROOT = Path("/kaggle/input/datasets/mohamedkhapiry/rehab24-6")
SEGMENTATION_CSV = DATA_ROOT / "Segmentation.csv"
JOINTS_3D_DIR = DATA_ROOT / "3d_joints"  # VERIFY: adjust if the real folder name differs

# 26-joint BVH-style skeleton, index order per joints_names.txt
JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Neck", "Head", "Head_end",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHand_end",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHand_end",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBase_end",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBase_end",
]
J = {name: i for i, name in enumerate(JOINT_NAMES)}

EX_ARM_ABDUCTION = 1
EX_SQUAT = 6


def load_npy(video_id: str, exercise_id: int) -> np.ndarray:
    """Load the (frames, 26, 4) mocap array for one person+exercise recording."""
    path = JOINTS_3D_DIR / f"Ex{exercise_id}" / f"{video_id}-120fps.npy"
    return np.load(path)


def xyz(frame: np.ndarray, joint_name: str) -> np.ndarray:
    """frame: (26, 4) array for one timestep. Returns (x, y, z)."""
    return frame[J[joint_name], :3]


def squat_angles(frame: np.ndarray) -> tuple[float, float]:
    """Returns (knee_angle, hip_angle) for one frame, using the right leg."""
    knee = joint_angle(
        xyz(frame, "RightUpLeg"), xyz(frame, "RightLeg"), xyz(frame, "RightFoot")
    )
    hip = joint_angle(
        xyz(frame, "Spine"), xyz(frame, "Hips"), xyz(frame, "RightUpLeg")
    )
    return knee, hip


def arm_abduction_angles(frame: np.ndarray) -> tuple[float, float]:
    """Returns (elevation_angle, elbow_angle) for one frame, right arm only."""
    elevation = joint_angle(
        xyz(frame, "Hips"), xyz(frame, "RightArm"), xyz(frame, "RightHand")
    )
    elbow = joint_angle(
        xyz(frame, "RightArm"), xyz(frame, "RightForeArm"), xyz(frame, "RightHand")
    )
    return elevation, elbow


def analyze_exercise(seg: pd.DataFrame, exercise_id: int, angle_fn, angle_names):
    rows = seg[
        (seg["exercise_id"] == exercise_id) & (seg["mocap_erroneous"] == 0)
    ]
    records = []
    for _, row in rows.iterrows():
        try:
            arr = load_npy(row["video_id"], exercise_id)
        except FileNotFoundError:
            continue  # path pattern likely needs adjusting — see note at top

        first, last = int(row["first_frame"]), int(row["last_frame"])
        last = min(last, arr.shape[0] - 1)
        if first >= last:
            continue

        segment = arr[first : last + 1]
        angle_series = [angle_fn(segment[t]) for t in range(segment.shape[0])]
        angle_series = np.array(angle_series)  # (T, 2)

        records.append(
            {
                "video_id": row["video_id"],
                "repetition_number": row["repetition_number"],
                "correctness": int(row["correctness"]),
                f"mean_{angle_names[0]}": angle_series[:, 0].mean(),
                f"mean_{angle_names[1]}": angle_series[:, 1].mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame, angle_col: str, target_low: float, target_high: float):
    for label, name in [(1, "correct"), (0, "incorrect")]:
        subset = df[df["correctness"] == label][angle_col]
        if subset.empty:
            continue
        in_range = subset.between(target_low, target_high).mean() * 100
        print(
            f"  {name:9s} (n={len(subset):3d}): "
            f"mean={subset.mean():6.1f}  std={subset.std():5.1f}  "
            f"in-target[{target_low}-{target_high}]={in_range:5.1f}%"
        )


if __name__ == "__main__":
    seg = pd.read_csv(SEGMENTATION_CSV)

    print("=" * 60)
    print("SQUAT (Ex6) — knee 70-100 deg target, hip 60-120 deg target")
    print("=" * 60)
    squat_df = analyze_exercise(seg, EX_SQUAT, squat_angles, ("knee", "hip"))
    print(f"Loaded {len(squat_df)} repetitions.")
    print("\nKnee angle:")
    summarize(squat_df, "mean_knee", 70, 100)
    print("\nHip angle:")
    summarize(squat_df, "mean_hip", 60, 120)

    print()
    print("=" * 60)
    print("ARM ABDUCTION (Ex1) — elevation 160-180 deg, elbow 170-180 deg")
    print("=" * 60)
    arm_df = analyze_exercise(
        seg, EX_ARM_ABDUCTION, arm_abduction_angles, ("elevation", "elbow")
    )
    print(f"Loaded {len(arm_df)} repetitions.")
    print("\nElevation angle:")
    summarize(arm_df, "mean_elevation", 160, 180)
    print("\nElbow angle:")
    summarize(arm_df, "mean_elbow", 170, 180)

    squat_df.to_csv("/kaggle/working/squat_ground_truth_angles.csv", index=False)
    arm_df.to_csv("/kaggle/working/arm_abduction_ground_truth_angles.csv", index=False)
    print("\nSaved per-rep angle CSVs to /kaggle/working/")
