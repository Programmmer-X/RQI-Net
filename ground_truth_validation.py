import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from angle_calculator import joint_angle

DATA_ROOT = Path("/kaggle/input/datasets/mohamedkhapiry/rehab24-6")
SEGMENTATION_CSV = DATA_ROOT / "Segmentation.csv"
JOINTS_3D_DIR = DATA_ROOT / "3d_joints"
MARKERS_3D_DIR = DATA_ROOT / "3d_markers"

EX_ARM_ABDUCTION = 1
EX_SQUAT = 6

JOINT_NAMES_26 = [
    "Hips", "Spine", "Spine1", "Neck", "Head", "Head_end",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHand_end",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHand_end",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBase_end",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBase_end",
]
J26 = {name: i for i, name in enumerate(JOINT_NAMES_26)}

MARKER_NAMES_41 = [
    "BackLeft", "BackRight", "BackTop", "Chest", "HeadFront", "HeadSide", "HeadTop",
    "LAnkleOut", "LElbowOut", "LHandOut", "LHeel", "LKneeOut", "LShin",
    "LShoulderBack", "LShoulderTop", "LThigh", "LToeIn", "LToeOut", "LToeTip",
    "LUArmHigh", "LWristIn", "LWristOut",
    "RAnkleOut", "RElbowOut", "RHandOut", "RHeel", "RKneeOut", "RShin",
    "RShoulderBack", "RShoulderTop", "RThigh", "RToeIn", "RToeOut", "RToeTip",
    "RUArmHigh", "RWristIn", "RWristOut",
    "WaistLBack", "WaistLFront", "WaistRBack", "WaistRFront",
]
M41 = {name: i for i, name in enumerate(MARKER_NAMES_41)}


def _pt(frame, idx_map, name):
    return frame[idx_map[name], :3]


def load_source(video_id, exercise_id):
    joints_path = JOINTS_3D_DIR / f"Ex{exercise_id}" / f"{video_id}-120fps.npy"
    if joints_path.exists():
        return "joints", np.load(joints_path)
    markers_path = MARKERS_3D_DIR / f"Ex{exercise_id}" / f"{video_id}-120fps.npy"
    if markers_path.exists():
        return "markers", np.load(markers_path)
    raise FileNotFoundError(f"No npy for {video_id} (Ex{exercise_id})")


def get_virtual_points(source_type, frame):
    if source_type == "joints":
        g = lambda name: _pt(frame, J26, name)
        return {
            "hip_right": g("RightUpLeg"), "knee_right": g("RightLeg"),
            "ankle_right": g("RightFoot"), "shoulder_right": g("RightArm"),
            "elbow_right": g("RightForeArm"), "wrist_right": g("RightHand"),
            "shoulder_center": (g("LeftShoulder") + g("RightShoulder")) / 2,
            "pelvis_center": g("Hips"),
            "neck": g("Neck"), "spine1": g("Spine1"),
        }
    elif source_type == "markers":
        m = lambda name: _pt(frame, M41, name)
        r_sh = (m("RShoulderBack") + m("RShoulderTop")) / 2
        l_sh = (m("LShoulderBack") + m("LShoulderTop")) / 2
        return {
            "hip_right": (m("WaistRBack") + m("WaistRFront")) / 2,
            "knee_right": m("RKneeOut"), "ankle_right": m("RAnkleOut"),
            "shoulder_right": r_sh, "elbow_right": m("RElbowOut"),
            "wrist_right": (m("RWristIn") + m("RWristOut")) / 2,
            "shoulder_center": (l_sh + r_sh) / 2,
            "pelvis_center": (m("WaistLBack") + m("WaistLFront") + m("WaistRBack") + m("WaistRFront")) / 4,
            "neck": m("Chest"), "spine1": m("Chest"),  # no direct marker — approximated
        }
    raise ValueError(source_type)


def squat_angles(p):
    knee = joint_angle(p["hip_right"], p["knee_right"], p["ankle_right"])
    hip = joint_angle(p["shoulder_center"], p["hip_right"], p["knee_right"])
    return knee, hip


def arm_elevation_candidates(p):
    """
    Six candidate definitions of 'arm elevation', all as the angle at the
    shoulder between some trunk-side reference and the upper-arm vector
    (shoulder_right -> elbow_right). Returns a dict so all can be compared
    against the same reps at once.
    """
    sh, el = p["shoulder_right"], p["elbow_right"]
    trunk_dir = p["neck"] - p["pelvis_center"]  # whole-trunk axis, midline-ish
    world_up = np.array([0.0, 1.0, 0.0])        # y confirmed vertical in this dataset

    return {
        "pelvis_ref": joint_angle(p["pelvis_center"], sh, el),           # original/baseline
        "neck_ref": joint_angle(p["neck"], sh, el),
        "spine1_ref": joint_angle(p["spine1"], sh, el),
        "shoulder_center_ref": joint_angle(p["shoulder_center"], sh, el),
        "trunk_vector_ref": joint_angle(sh + trunk_dir, sh, el),         # neck-pelvis direction, anchored at shoulder
        "world_vertical_ref": joint_angle(sh + world_up, sh, el),        # fixed vertical, trunk-lean-proof
    }


def elbow_angle(p):
    return joint_angle(p["shoulder_right"], p["elbow_right"], p["wrist_right"])


def read_segmentation(path):
    df = pd.read_csv(path)
    if "video_id" not in df.columns:
        df = pd.read_csv(path, sep=";")
    if "video_id" not in df.columns:
        raise ValueError(f"Could not parse {path}")
    return df


def analyze_squat(seg):
    rows = seg[(seg["exercise_id"] == EX_SQUAT) & (seg["mocap_erroneous"] == 0)]
    records = []
    for _, row in rows.iterrows():
        try:
            source_type, arr = load_source(row["video_id"], EX_SQUAT)
        except FileNotFoundError:
            continue
        first, last = int(row["first_frame"]), int(row["last_frame"])
        last = min(last, arr.shape[0] - 1)
        if first >= last:
            continue
        segment = arr[first:last + 1]
        series = np.array([squat_angles(get_virtual_points(source_type, segment[t])) for t in range(segment.shape[0])])
        peak_idx = series[:, 0].argmin()  # deepest knee bend
        records.append({
            "video_id": row["video_id"], "correctness": int(row["correctness"]),
            "peak_knee": series[peak_idx, 0], "peak_hip": series[peak_idx, 1],
        })
    return pd.DataFrame.from_records(records)


def analyze_arm_abduction_candidates(seg):
    rows = seg[(seg["exercise_id"] == EX_ARM_ABDUCTION) & (seg["mocap_erroneous"] == 0)]
    candidate_names = ["pelvis_ref", "neck_ref", "spine1_ref", "shoulder_center_ref",
                        "trunk_vector_ref", "world_vertical_ref"]
    records = []
    for _, row in rows.iterrows():
        try:
            source_type, arr = load_source(row["video_id"], EX_ARM_ABDUCTION)
        except FileNotFoundError:
            continue
        first, last = int(row["first_frame"]), int(row["last_frame"])
        last = min(last, arr.shape[0] - 1)
        if first >= last:
            continue
        segment = arr[first:last + 1]

        series = {name: [] for name in candidate_names}
        elbow_series = []
        for t in range(segment.shape[0]):
            pts = get_virtual_points(source_type, segment[t])
            cands = arm_elevation_candidates(pts)
            for name in candidate_names:
                series[name].append(cands[name])
            elbow_series.append(elbow_angle(pts))

        rec = {"video_id": row["video_id"], "correctness": int(row["correctness"])}
        for name in candidate_names:
            arr_s = np.array(series[name])
            rec[f"peak_{name}"] = arr_s.max()  # each definition peaks on its own terms
        rec["elbow_at_overall_peak"] = np.array(elbow_series)[np.array(series["pelvis_ref"]).argmax()]
        records.append(rec)
    return pd.DataFrame.from_records(records), candidate_names


def cohens_d(a, b):
    n1, n0 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.std() ** 2 + (n0 - 1) * b.std() ** 2) / (n1 + n0 - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def summarize_simple(df, col, lo, hi):
    for label, name in [(1, "correct"), (0, "incorrect")]:
        subset = df[df["correctness"] == label][col]
        if subset.empty:
            continue
        in_range = subset.between(lo, hi).mean() * 100
        print(f"  {name:9s} (n={len(subset):3d}): mean={subset.mean():6.1f}  std={subset.std():5.1f}  in-target[{lo}-{hi}]={in_range:5.1f}%")


def compare_candidates(df, candidate_names):
    results = []
    for name in candidate_names:
        col = f"peak_{name}"
        correct = df[df["correctness"] == 1][col]
        incorrect = df[df["correctness"] == 0][col]
        d = cohens_d(correct, incorrect)
        results.append((name, d, correct.mean(), incorrect.mean()))
    results.sort(key=lambda r: -abs(r[1]))

    print(f"{'definition':20s} {'cohens_d':>9s} {'correct_mean':>13s} {'incorrect_mean':>15s}")
    for name, d, cm, im in results:
        flag = "  <-- best separation" if name == results[0][0] else ""
        print(f"{name:20s} {d:9.2f} {cm:13.1f} {im:15.1f}{flag}")
    return results


if __name__ == "__main__":
    seg = read_segmentation(SEGMENTATION_CSV)

    print("=" * 70); print("SQUAT (Ex6) — unchanged from before"); print("=" * 70)
    squat_df = analyze_squat(seg)
    print(f"Loaded {len(squat_df)} reps.")
    print("\nKnee (at deepest point):"); summarize_simple(squat_df, "peak_knee", 70, 100)
    print("\nHip (at deepest point):"); summarize_simple(squat_df, "peak_hip", 60, 120)

    print(); print("=" * 70)
    print("ARM ABDUCTION (Ex1) — comparing 6 elevation definitions by effect size")
    print("=" * 70)
    arm_df, candidate_names = analyze_arm_abduction_candidates(seg)
    print(f"Loaded {len(arm_df)} reps.\n")
    compare_candidates(arm_df, candidate_names)

    squat_df.to_csv("/kaggle/working/squat_ground_truth_angles.csv", index=False)
    arm_df.to_csv("/kaggle/working/arm_abduction_candidate_angles.csv", index=False)
    print("\nSaved CSVs to /kaggle/working/")