"""
ml_baseline_squat.py

Classical ML baseline for Ex6 (Squat) correctness classification, mirroring
ml_baseline_arm_abduction.py for paper consistency. Only 2 features exist
for squat so far (peak_knee, peak_hip) vs Ex1's 5 — squat already showed
real univariate signal (correct reps bend consistently deeper), so a
richer feature set wasn't built; add more (e.g. stability/jitter, ROM)
only if this baseline underperforms.

Note: squat is class-imbalanced (134 correct / 61 incorrect, ~69% correct)
— accuracy alone would be misleading; F1/precision/recall matter more here
than for Ex1's balanced 90/88 split.
"""
import sys
from pathlib import Path
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth_validation import SEGMENTATION_CSV, analyze_squat, read_segmentation
from ml_utils import evaluate_model

FEATURE_COLS = ["peak_knee", "peak_hip", "trunk_lean_at_peak", "max_trunk_lean"]
# valgus_at_peak dropped — r=0.98 with peak_knee, the x,y projection still
# carries the same depth signal (y dominates both), not independent info


def main():
    seg = read_segmentation(SEGMENTATION_CSV)
    squat_df = analyze_squat(seg)

    X = squat_df[FEATURE_COLS].values
    y = squat_df["correctness"].values
    groups = squat_df["video_id"].values

    n_subjects = len(set(groups))
    n_splits = min(5, n_subjects)
    n_correct, n_incorrect = int(y.sum()), int((1 - y).sum())
    print(f"Loaded {len(squat_df)} reps across {n_subjects} subjects "
          f"({n_correct} correct / {n_incorrect} incorrect). GroupKFold(n_splits={n_splits}).\n")


    print("Feature correlation matrix:")
    print(squat_df[FEATURE_COLS].corr().round(2))
    print()

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
    }
    # X is RAW here — evaluate_model scales internally, per fold, no leakage.
    results = [evaluate_model(name, model, X, y, groups, n_splits) for name, model in models.items()]
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    print("=" * 72)
    print("CROSS-VALIDATED COMPARISON (out-of-fold, grouped by subject)")
    print("=" * 72)
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWinner (by ROC-AUC — F1 is unreliable here under 68/32 class imbalance): {results_df.iloc[0]['model']}")

    # Final check before accepting these results: per-subject z-score each
    # feature. Tests whether the weak signal above is masked by between-
    # subject variance (2/9 subjects showed reversed correctness direction
    # earlier) rather than genuinely absent. Uses only each subject's own
    # feature mean/std — no label information, so no leakage.
    normalized_df = squat_df.copy()
    for col in FEATURE_COLS:
        normalized_df[col] = squat_df.groupby("video_id")[col].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0
        )
    X_norm = np.nan_to_num(normalized_df[FEATURE_COLS].values, nan=0.0)

    norm_results = [
        evaluate_model(f"{name}_subject_normalized", model, X_norm, y, groups, n_splits)
        for name, model in models.items()
    ]
    norm_df = pd.DataFrame(norm_results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 72)
    print("PER-SUBJECT NORMALIZED FEATURES (z-scored within each subject)")
    print("=" * 72)
    print(norm_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 72)
    print("INTERPRETABILITY (fit on full data, not the CV estimate)")
    print("=" * 72)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # full-data fit for interpretability only — no held-out evaluation here, so no leakage concern
    lr_full = LogisticRegression(max_iter=1000).fit(X_scaled, y)
    print("\nLogistic regression coefficients (standardized features):")
    for name, coef in sorted(zip(FEATURE_COLS, lr_full.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:12s} {coef:+.3f}")

    rf_full = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42).fit(X_scaled, y)
    print("\nRandom forest feature importances:")
    for name, imp in sorted(zip(FEATURE_COLS, rf_full.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:12s} {imp:.3f}")

    results_df.to_csv("/kaggle/working/squat_ml_baseline_results.csv", index=False)
    print("\nSaved results to /kaggle/working/squat_ml_baseline_results.csv")


if __name__ == "__main__":
    main()