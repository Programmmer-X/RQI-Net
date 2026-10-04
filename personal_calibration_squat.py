"""
personal_calibration_squat.py

Tests per-patient calibration — the strategy used by prior work on this
exact dataset (a smartphone-camera rehab monitoring paper derives a
person-specific threshold from a few known-correct/known-incorrect reps
per patient, rather than a population-generalizing classifier).

This is a different paradigm from the earlier ML baselines, not another
round of feature engineering: leave-one-rep-out WITHIN each subject —
for each rep, build "correct"/"incorrect" centroids from that SAME
subject's OTHER reps (excluding the one being classified), then classify
by nearest centroid. Falls back to the population centroid for a class
if a subject has zero remaining reps of that class after exclusion
(e.g. a subject with only 1 incorrect rep total).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth_validation import SEGMENTATION_CSV, analyze_squat, read_segmentation
from ml_baseline_squat import FEATURE_COLS


def leave_one_out_personal_centroid(df, feature_cols):
    X = StandardScaler().fit_transform(df[feature_cols].values)
    y = df["correctness"].values
    subjects = df["video_id"].values
    idx = np.arange(len(df))

    global_correct = X[y == 1].mean(axis=0)
    global_incorrect = X[y == 0].mean(axis=0)

    preds, scores = [], []
    fallback_count = 0
    for i in range(len(df)):
        others = (subjects == subjects[i]) & (idx != i)
        correct_mask = others & (y == 1)
        incorrect_mask = others & (y == 0)

        c_centroid = X[correct_mask].mean(axis=0) if correct_mask.sum() > 0 else global_correct
        ic_centroid = X[incorrect_mask].mean(axis=0) if incorrect_mask.sum() > 0 else global_incorrect
        if correct_mask.sum() == 0 or incorrect_mask.sum() == 0:
            fallback_count += 1

        d_correct = np.linalg.norm(X[i] - c_centroid)
        d_incorrect = np.linalg.norm(X[i] - ic_centroid)
        preds.append(1 if d_correct < d_incorrect else 0)
        scores.append(d_incorrect - d_correct)  # continuous: higher = more "correct"-like

    return np.array(preds), np.array(scores), y, fallback_count


def main():
    seg = read_segmentation(SEGMENTATION_CSV)
    squat_df = analyze_squat(seg)

    preds, scores, y, fallback_count = leave_one_out_personal_centroid(squat_df, FEATURE_COLS)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, scores)

    print(f"Loaded {len(squat_df)} reps. {fallback_count} needed a global-centroid fallback "
          f"(subject had 0 remaining reps of one class after leave-one-out).\n")
    print("=" * 72)
    print("PER-PATIENT CALIBRATION (leave-one-rep-out nearest centroid, within subject)")
    print("=" * 72)
    print(f"accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  roc_auc={auc:.3f}")
    print(f"\nFor comparison: population models (subject-held-out) scored ~0.50-0.53 AUC — chance level.")


if __name__ == "__main__":
    main()