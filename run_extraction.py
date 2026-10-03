"""
run_extraction.py

Ties dataset_loader -> pose_estimator -> exercise_analyzer together:
for every repetition in REHAB24-6's manifest, pull the frame range out
of the untrimmed video, run MediaPipe, score it, and record the result
next to the ground-truth correctness label.

Meant to run on Kaggle, where the dataset is actually attached. Writes
incrementally to results/rep_metrics.csv so a crash partway through
doesn't lose completed work -- safe to re-run (skips rows already in
the output file).

Usage (Kaggle):
    !python run_extraction.py \\
        --csv /kaggle/input/datasets/mohamedkhapiry/rehab24-6/Segmentation.csv \\
        --videos-root /kaggle/input/datasets/mohamedkhapiry/rehab24-6/videos \\
        --out results/rep_metrics.csv \\
        --limit 20   # start small -- drop this flag once it looks right
"""

import argparse
import csv
import os
import sys
import time

from dataset_loader import load_segmentation
from exercise_analyzer import analyze_arm_abduction, analyze_squat
from pose_estimator import extract_frame_range_landmarks

FIELDNAMES = [
    "video_id", "exercise_name", "repetition_number", "side",
    "first_frame", "last_frame", "n_frames_detected",
    "ground_truth_correctness",
    # squat fields (blank for arm_abduction rows)
    "depth_score", "posture_score", "stability_score",
    "mean_knee_angle", "mean_hip_angle",
    # arm abduction fields (blank for squat rows)
    "rom_score", "elbow_score", "smoothness_score",
    "mean_elevation_angle", "mean_elbow_angle",
    "error",
]


def already_done(out_path: str) -> set:
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["video_id"], row["exercise_name"], row["repetition_number"]))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Segmentation.csv")
    parser.add_argument("--videos-root", required=True, help="Path to the videos/ folder")
    parser.add_argument("--out", default="results/rep_metrics.csv")
    parser.add_argument("--camera", default="Camera17")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N reps (for a quick test run)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    manifest = load_segmentation(args.csv, args.videos_root, camera=args.camera)
    if args.limit:
        manifest = manifest[: args.limit]

    done = already_done(args.out)
    write_header = not os.path.exists(args.out)

    rotate_90 = args.camera == "Camera18"

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        start = time.time()
        for i, rec in enumerate(manifest):
            key = (rec.person_id, rec.exercise_name, str(rec.repetition_number))
            if key in done:
                continue

            row = {
                "video_id": rec.person_id,
                "exercise_name": rec.exercise_name,
                "repetition_number": rec.repetition_number,
                "side": rec.side or "",
                "first_frame": rec.first_frame,
                "last_frame": rec.last_frame,
                "ground_truth_correctness": rec.correctness,
                "error": "",
            }

            try:
                if not os.path.exists(rec.video_path):
                    raise FileNotFoundError(rec.video_path)

                frames = extract_frame_range_landmarks(
                    rec.video_path, rec.first_frame, rec.last_frame, rotate_90=rotate_90
                )
                row["n_frames_detected"] = len(frames)

                if len(frames) == 0:
                    raise ValueError("No pose detected in any frame of this rep")

                if rec.exercise_name == "squat":
                    m = analyze_squat(frames)
                    row.update(
                        depth_score=m.depth_score,
                        posture_score=m.posture_score,
                        stability_score=m.stability_score,
                        mean_knee_angle=m.mean_knee_angle,
                        mean_hip_angle=m.mean_hip_angle,
                    )
                else:  # arm_abduction
                    m = analyze_arm_abduction(frames, side=rec.side)
                    row.update(
                        rom_score=m.rom_score,
                        elbow_score=m.elbow_score,
                        smoothness_score=m.smoothness_score,
                        mean_elevation_angle=m.mean_elevation_angle,
                        mean_elbow_angle=m.mean_elbow_angle,
                    )

            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                row.setdefault("n_frames_detected", 0)

            writer.writerow(row)
            f.flush()

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                print(f"[{i + 1}/{len(manifest)}] {elapsed:.0f}s elapsed", file=sys.stderr)

    print(f"Done. Results in {args.out}")


if __name__ == "__main__":
    main()
