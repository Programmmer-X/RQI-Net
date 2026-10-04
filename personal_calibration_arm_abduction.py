"""
personal_calibration_arm_abduction.py

Same protocol as personal_calibration_squat.py, applied to Ex1 (Arm
Abduction) using FEATURE_COLS from ml_baseline_arm_abduction.py. Built
leak-free from the start (scaler fit fresh per held-out rep).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth_validation import SEGMENTATION_CSV, analyze_arm_abduction_final, read_segmentation
from ml_baseline_arm_abduction import FEATURE_COLS


def leave_one_out_personal_centroid(df, feature_cols):
    X_raw = df[feature_cols].values
    y = df["correctness"].values
    subjects = df["video_id"].values
    idx = np.arange(len(df))

    preds, scores = [], []
    fallback_count = 0
    for i in range(len(df)):
        train_mask = idx != i
        scaler = StandardScaler().fit(X_raw[train_mask])
        X_train = scaler.transform(X_raw[train_mask])
        x_i = scaler.transform(X_raw[i : i + 1])[0]

        train_subjects = subjects[train_mask]
        train_y = y[train_mask]
        same_subject = train_subjects == subjects[i]

        correct_mask = same_subject & (train_y == 1)
        incorrect_mask = same_subject & (train_y == 0)

        global_correct = X_train[train_y == 1].mean(axis=0)
        global_incorrect = X_train[train_y == 0].mean(axis=0)

        c_centroid = X_train[correct_mask].mean(axis=0) if correct_mask.sum() > 0 else global_correct
        ic_centroid = X_train[incorrect_mask].mean(axis=0) if incorrect_mask.sum() > 0 else global_incorrect
        if correct_mask.sum() == 0 or incorrect_mask.sum() == 0:
            fallback_count += 1

        d_correct = np.linalg.norm(x_i - c_centroid)
        d_incorrect = np.linalg.norm(x_i - ic_centroid)
        preds.append(1 if d_correct < d_incorrect else 0)
        scores.append(d_incorrect - d_correct)

    return np.array(preds), np.array(scores), y, fallback_count


def main():
    seg = read_segmentation(SEGMENTATION_CSV)
    arm_df = analyze_arm_abduction_final(seg)

    preds, scores, y, fallback_count = leave_one_out_personal_centroid(arm_df, FEATURE_COLS)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, scores)

    print(f"Loaded {len(arm_df)} reps. {fallback_count} needed a global-centroid fallback "
          f"(subject had 0 remaining reps of one class after leave-one-out).\n")
    print("=" * 72)
    print("PER-PATIENT CALIBRATION (leave-one-rep-out nearest centroid, within subject)")
    print("Scaler fit fresh per held-out rep — no leakage")
    print("=" * 72)
    print(f"accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  roc_auc={auc:.3f}")


if __name__ == "__main__":
    main()