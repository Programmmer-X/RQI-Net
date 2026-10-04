"""
ml_baseline_arm_abduction.py

Classical ML baseline for Ex1 (Arm Abduction) correctness classification.

Why a model instead of a threshold: univariate testing (ground_truth_validation.py)
found every individual feature negligible (|d| < 0.20) because REHAB24-6
deliberately gives each incorrect rep a DIFFERENT predefined error type
(per the dataset paper — errors are varied per subject, not a single
consistent mistake). No single feature should be expected to separate
correct/incorrect cleanly; a model combining all 5 features can still
work if each feature catches a different error mode.

Compares logistic regression (interpretable, gives coefficients) against
random forest (likely higher ceiling, less interpretable), using
GroupKFold on video_id so no subject's reps leak across train/test folds
— with only 13 subjects for Ex1, this matters a lot more than usual.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth_validation import SEGMENTATION_CSV, analyze_arm_abduction_final, read_segmentation
from ml_utils import evaluate_model

FEATURE_COLS = ["peak_elevation", "peak_elbow", "max_trunk_lean", "mean_trunk_lean", "elevation_rom"]


def main():
    seg = read_segmentation(SEGMENTATION_CSV)
    arm_df = analyze_arm_abduction_final(seg)

    X = arm_df[FEATURE_COLS].values
    y = arm_df["correctness"].values
    groups = arm_df["video_id"].values

    n_subjects = len(set(groups))
    n_splits = min(5, n_subjects)
    print(f"Loaded {len(arm_df)} reps across {n_subjects} subjects. GroupKFold(n_splits={n_splits}), grouped by video_id.\n")

    print("Feature correlation matrix:")
    print(arm_df[FEATURE_COLS].corr().round(2))
    print()

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
    }

    # X is RAW here — evaluate_model scales internally, per fold, no leakage.
    results = [evaluate_model(name, model, X, y, groups, n_splits) for name, model in models.items()]
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)

    print("=" * 72)
    print("CROSS-VALIDATED COMPARISON (out-of-fold, grouped by subject)")
    print("=" * 72)
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    winner = results_df.iloc[0]["model"]
    print(f"\nWinner (by F1): {winner}")

    # Fit on full data for interpretability — separate from the CV evaluation above.
    print("\n" + "=" * 72)
    print("INTERPRETABILITY (fit on full data, not the CV estimate)")
    print("=" * 72)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # full-data fit for interpretability only — no held-out evaluation here, so no leakage concern
    lr_full = LogisticRegression(max_iter=1000).fit(X_scaled, y)
    print("\nLogistic regression coefficients (standardized features — magnitude = importance):")
    for name, coef in sorted(zip(FEATURE_COLS, lr_full.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:20s} {coef:+.3f}")

    rf_full = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42).fit(X_scaled, y)
    print("\nRandom forest feature importances:")
    for name, imp in sorted(zip(FEATURE_COLS, rf_full.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:20s} {imp:.3f}")

    results_df.to_csv("/kaggle/working/arm_abduction_ml_baseline_results.csv", index=False)
    print("\n" + "=" * 72)
    print("REDUCED FEATURE SET (dropping mean_trunk_lean — collinear with max_trunk_lean)")
    print("=" * 72)
    reduced_cols = [c for c in FEATURE_COLS if c != "mean_trunk_lean"]
    X_reduced = arm_df[reduced_cols].values  # RAW — evaluate_model scales internally
    reduced_results = [
        evaluate_model(f"{n}_reduced", m, X_reduced, y, groups, n_splits)
        for n, m in {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
        }.items()
    ]
    reduced_df = pd.DataFrame(reduced_results).sort_values("f1", ascending=False).reset_index(drop=True)
    print(reduced_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    lr_reduced = LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(X_reduced), y)
    print("\nLR coefficients WITHOUT mean_trunk_lean:")
    for name, coef in sorted(zip(reduced_cols, lr_reduced.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:20s} {coef:+.3f}")
    print("\nSaved results to /kaggle/working/arm_abduction_ml_baseline_results.csv")


if __name__ == "__main__":
    main()