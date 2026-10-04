"""
rqi.py

Rehabilitation Quality Index (RQI) scoring, implementing the architecture
validated in VALIDATION_FINDINGS.md:

  - Ex1 (Arm Abduction): population Random Forest (CV ROC-AUC = 0.605).
    Personal calibration was tested and REJECTED for this exercise
    (ROC-AUC = 0.486, below the population result) — do not use it here.
  - Ex6 (Squat): per-patient calibration (leave-one-out ROC-AUC = 0.556,
    vs. population ROC-AUC ~0.526-0.528). Population models were tested
    and are NOT used for squat scoring.

RQI is a continuous score in [0, 1], higher = better. This module never
emits a binary correct/incorrect/pass/fail label — only a continuous
score, a human-readable quality band derived from that score, and
rule-based feedback.

Reuses existing repository functions rather than re-deriving features:
  - ground_truth_validation.read_segmentation / analyze_arm_abduction_final
    / analyze_squat for feature extraction.
  - FEATURE_COLS from ml_baseline_arm_abduction.py and ml_baseline_squat.py
    for the exact validated feature sets.
No validation script is modified; this file only imports from them.

Known limitation, stated plainly rather than hidden: Ex1's Random Forest
has a validated ROC-AUC of 0.605 — well above chance but not a strong
classifier. Its predict_proba() output is used as RQI per the validated
architecture decision, but should be read as a moderate-confidence signal,
not a precise probability.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth_validation import (
    SEGMENTATION_CSV,
    analyze_arm_abduction_final,
    analyze_squat,
    read_segmentation,
)
from ml_baseline_arm_abduction import FEATURE_COLS as ARM_FEATURE_COLS
from ml_baseline_squat import FEATURE_COLS as SQUAT_FEATURE_COLS

# Quality bands over the continuous RQI score. These bucket boundaries are
# illustrative (chosen to match the examples in the design spec), NOT
# independently validated against an accuracy-optimal cutoff — no such
# cutoff was established during validation. Treat as a display convenience,
# not a certified threshold.
QUALITY_BANDS = [
    (0.85, "Excellent"),
    (0.65, "Good"),
    (0.45, "Borderline"),
    (0.0, "Poor"),
]

# Feedback trigger distance, in standard deviations from the correct-class
# mean, computed from the real training data (not an arbitrary constant).
FEEDBACK_SD_THRESHOLD = 0.5


def _quality_label(score: float) -> str:
    """Maps a continuous RQI score to a human-readable quality band."""
    for lower_bound, label in QUALITY_BANDS:
        if score >= lower_bound:
            return label
    return "Poor"


@dataclass
class _SquatCalibration:
    """Per-patient squat calibration state."""
    scaler: StandardScaler
    correct_centroid: np.ndarray
    incorrect_centroid: np.ndarray
    used_fallback_correct: bool = False
    used_fallback_incorrect: bool = False


@dataclass
class _FeatureReference:
    """Mean/std of each feature among CORRECT reps in the training population — used for feedback thresholds."""
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)


class RQIScorer:
    """
    Rehabilitation Quality Index scorer for Arm Abduction (Ex1) and Squat
    (Ex6), implementing the two different validated scoring paradigms for
    each exercise (see module docstring).

    Typical usage:
        scorer = RQIScorer()
        scorer.fit_arm_model()
        scorer.fit_squat_calibration(patient_id="P001", calibration_reps=df)

        result = scorer.score_rep("arm_abduction", rep_features=features)
        result = scorer.score_rep("squat", rep_features=features, patient_id="P001")
    """

    def __init__(self) -> None:
        self._arm_pipeline: Optional[Pipeline] = None
        self._arm_feature_reference: Optional[_FeatureReference] = None

        self._squat_calibrations: Dict[str, _SquatCalibration] = {}
        self._squat_population_correct: Optional[np.ndarray] = None
        self._squat_population_incorrect: Optional[np.ndarray] = None
        self._squat_population_scaler: Optional[StandardScaler] = None
        self._squat_feature_reference: Optional[_FeatureReference] = None

    # ------------------------------------------------------------------
    # Arm Abduction (Ex1) — population Random Forest
    # ------------------------------------------------------------------

    def fit_arm_model(self, model_path: Optional[str] = None) -> None:
        """
        Fits (or loads) the validated Ex1 Random Forest.

        Uses the exact hyperparameters already validated in
        ml_baseline_arm_abduction.py (n_estimators=200, max_depth=4,
        random_state=42) — not a new search. If model_path is given and
        exists, loads the persisted pipeline instead of refitting (the
        repository does not yet ship a persisted model; pass model_path
        to cache one after the first fit for faster reuse).

        Also computes per-feature mean/std among correct reps, used by
        generate_feedback() — this is real data from the training
        population, not an invented threshold.
        """
        if model_path is not None and Path(model_path).exists():
            self._arm_pipeline = joblib.load(model_path)
        else:
            seg = read_segmentation(SEGMENTATION_CSV)
            arm_df = analyze_arm_abduction_final(seg)

            X = arm_df[ARM_FEATURE_COLS].values
            y = arm_df["correctness"].values

            # Same Pipeline(StandardScaler, model) pattern already
            # validated via ml_utils.evaluate_model — scaling and model
            # fit together, fit once on the full dataset for deployment
            # (mirrors the "INTERPRETABILITY" full-data fit already done
            # in ml_baseline_arm_abduction.py, not a new training recipe).
            self._arm_pipeline = make_pipeline(
                StandardScaler(),
                RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
            )
            self._arm_pipeline.fit(X, y)

            self._arm_feature_reference = self._compute_feature_reference(arm_df, ARM_FEATURE_COLS)

            if model_path is not None:
                joblib.dump(self._arm_pipeline, model_path)

    def score_arm_rep(self, rep_features: Dict[str, float]) -> float:
        """
        RQI for one Arm Abduction repetition: P(correct) from the
        validated Random Forest. In [0, 1], higher = better, by
        construction (it IS a probability).

        rep_features must contain all of ARM_FEATURE_COLS:
        peak_elevation, peak_elbow, max_trunk_lean, mean_trunk_lean,
        elevation_rom.
        """
        if self._arm_pipeline is None:
            raise RuntimeError("Call fit_arm_model() before scoring.")
        self._validate_features(rep_features, ARM_FEATURE_COLS)

        x = np.array([[rep_features[col] for col in ARM_FEATURE_COLS]])
        probability_correct = self._arm_pipeline.predict_proba(x)[0, 1]
        return float(np.clip(probability_correct, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Squat (Ex6) — per-patient calibration
    # ------------------------------------------------------------------

    def fit_squat_calibration(self, patient_id: str, calibration_reps: pd.DataFrame) -> None:
        """
        Calibration phase for one patient's squat scoring: builds
        correct/incorrect centroids from a few of THIS patient's known
        reps (physio-supervised — some deliberately correct, some
        deliberately incorrect, matching the REHAB24-6 protocol this
        architecture was validated against).

        calibration_reps must have columns = SQUAT_FEATURE_COLS plus
        "correctness" (0/1). If calibration_reps lacks reps of one
        class entirely, falls back to the POPULATION centroid for that
        class (same fallback rule used during validation; triggered for
        1/191 reps there — expect it to be rare, not the norm).
        """
        if self._squat_population_correct is None:
            self._fit_squat_population_reference()

        X_raw = calibration_reps[SQUAT_FEATURE_COLS].values
        y = calibration_reps["correctness"].values

        scaler = StandardScaler().fit(X_raw)
        X_scaled = scaler.transform(X_raw)

        correct_mask = y == 1
        incorrect_mask = y == 0

        used_fallback_correct = correct_mask.sum() == 0
        used_fallback_incorrect = incorrect_mask.sum() == 0

        correct_centroid = (
            X_scaled[correct_mask].mean(axis=0)
            if not used_fallback_correct
            else scaler.transform(self._squat_population_scaler.inverse_transform([self._squat_population_correct]))[0]
        )
        incorrect_centroid = (
            X_scaled[incorrect_mask].mean(axis=0)
            if not used_fallback_incorrect
            else scaler.transform(self._squat_population_scaler.inverse_transform([self._squat_population_incorrect]))[0]
        )

        self._squat_calibrations[patient_id] = _SquatCalibration(
            scaler=scaler,
            correct_centroid=correct_centroid,
            incorrect_centroid=incorrect_centroid,
            used_fallback_correct=used_fallback_correct,
            used_fallback_incorrect=used_fallback_incorrect,
        )

    def score_squat_rep(self, patient_id: str, rep_features: Dict[str, float]) -> float:
        """
        RQI for one Squat repetition, via the validated per-patient
        nearest-centroid distance ratio:

            score = distance_incorrect / (distance_correct + distance_incorrect)

        Closer to the correct centroid -> distance_correct shrinks ->
        score rises toward 1. Clamped to [0, 1].

        Requires fit_squat_calibration(patient_id, ...) to have been
        called first for this patient.
        """
        if patient_id not in self._squat_calibrations:
            raise RuntimeError(
                f"No calibration for patient '{patient_id}'. Call fit_squat_calibration() first."
            )
        self._validate_features(rep_features, SQUAT_FEATURE_COLS)

        calibration = self._squat_calibrations[patient_id]
        x_raw = np.array([[rep_features[col] for col in SQUAT_FEATURE_COLS]])
        x_scaled = calibration.scaler.transform(x_raw)[0]

        distance_correct = float(np.linalg.norm(x_scaled - calibration.correct_centroid))
        distance_incorrect = float(np.linalg.norm(x_scaled - calibration.incorrect_centroid))

        denominator = distance_correct + distance_incorrect
        if denominator == 0:
            # Degenerate case: rep is exactly equidistant-zero from both
            # centroids (only possible if correct_centroid == incorrect_centroid).
            return 0.5

        score_raw = distance_incorrect / denominator
        return float(np.clip(score_raw, 0.0, 1.0))

    def _fit_squat_population_reference(self) -> None:
        """Lazily computes population-level squat centroids (fallback use only) and feedback reference stats."""
        seg = read_segmentation(SEGMENTATION_CSV)
        squat_df = analyze_squat(seg)

        X_raw = squat_df[SQUAT_FEATURE_COLS].values
        y = squat_df["correctness"].values

        self._squat_population_scaler = StandardScaler().fit(X_raw)
        X_scaled = self._squat_population_scaler.transform(X_raw)

        self._squat_population_correct = X_scaled[y == 1].mean(axis=0)
        self._squat_population_incorrect = X_scaled[y == 0].mean(axis=0)

        self._squat_feature_reference = self._compute_feature_reference(squat_df, SQUAT_FEATURE_COLS)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_feature_reference(df: pd.DataFrame, feature_cols: List[str]) -> _FeatureReference:
        """Mean/std of each feature among CORRECT reps — the reference distribution feedback is compared against."""
        correct = df[df["correctness"] == 1]
        return _FeatureReference(
            means={col: float(correct[col].mean()) for col in feature_cols},
            stds={col: float(correct[col].std()) for col in feature_cols},
        )

    def generate_feedback(self, exercise: str, rep_features: Dict[str, float]) -> List[str]:
        """
        Rule-based feedback: compares rep_features against the correct-
        class mean/std from the training population (real data, computed
        in fit_arm_model() / fit_squat_calibration()), not arbitrary
        constants. A feature more than FEEDBACK_SD_THRESHOLD standard
        deviations from the correct-class mean, in the "wrong" direction,
        triggers its message.
        """
        if exercise == "arm_abduction":
            return self._generate_arm_feedback(rep_features)
        elif exercise == "squat":
            return self._generate_squat_feedback(rep_features)
        raise ValueError(f"Unknown exercise: {exercise!r}")

    def _generate_arm_feedback(self, rep_features: Dict[str, float]) -> List[str]:
        if self._arm_feature_reference is None:
            raise RuntimeError("Call fit_arm_model() before generating feedback.")
        ref = self._arm_feature_reference
        feedback: List[str] = []

        if self._is_low(rep_features["peak_elevation"], ref, "peak_elevation"):
            feedback.append("Raise arm higher.")
        if self._is_low(rep_features["peak_elbow"], ref, "peak_elbow"):
            feedback.append("Keep elbow straighter.")
        if self._is_high(rep_features["max_trunk_lean"], ref, "max_trunk_lean"):
            feedback.append("Reduce trunk compensation.")

        return feedback

    def _generate_squat_feedback(self, rep_features: Dict[str, float]) -> List[str]:
        if self._squat_feature_reference is None:
            self._fit_squat_population_reference()
        ref = self._squat_feature_reference
        feedback: List[str] = []

        # Validated finding: correct squats have a SMALLER (more flexed)
        # knee angle. "peak_knee large" = too little flexion = shallow.
        if self._is_high(rep_features["peak_knee"], ref, "peak_knee"):
            feedback.append("Increase squat depth.")
        if self._is_high(rep_features["trunk_lean_at_peak"], ref, "trunk_lean_at_peak"):
            feedback.append("Keep torso more upright.")
        if self._is_high(rep_features["max_trunk_lean"], ref, "max_trunk_lean"):
            feedback.append("Reduce forward lean.")

        return feedback

    @staticmethod
    def _is_low(value: float, ref: _FeatureReference, feature: str) -> bool:
        mean, std = ref.means[feature], ref.stds[feature]
        if std == 0:
            return False
        return (mean - value) / std > FEEDBACK_SD_THRESHOLD

    @staticmethod
    def _is_high(value: float, ref: _FeatureReference, feature: str) -> bool:
        mean, std = ref.means[feature], ref.stds[feature]
        if std == 0:
            return False
        return (value - mean) / std > FEEDBACK_SD_THRESHOLD

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def score_rep(
        self,
        exercise: str,
        rep_features: Dict[str, float],
        patient_id: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Scores one repetition and returns the full result dict:

            {
                "exercise": "squat",
                "rqi_score": 0.74,
                "quality": "Good",
                "feedback": ["Increase squat depth", "Reduce forward lean"],
            }

        exercise: "arm_abduction" or "squat".
        patient_id: required for "squat" (per-patient calibration);
            ignored for "arm_abduction" (population model, no patient state).
        """
        if exercise == "arm_abduction":
            score = self.score_arm_rep(rep_features)
        elif exercise == "squat":
            if patient_id is None:
                raise ValueError("patient_id is required for squat scoring (per-patient calibration).")
            score = self.score_squat_rep(patient_id, rep_features)
        else:
            raise ValueError(f"Unknown exercise: {exercise!r}")

        return {
            "exercise": exercise,
            "rqi_score": round(score, 3),
            "quality": _quality_label(score),
            "feedback": self.generate_feedback(exercise, rep_features),
        }

    @staticmethod
    def _validate_features(rep_features: Dict[str, float], required_cols: List[str]) -> None:
        missing = [c for c in required_cols if c not in rep_features]
        if missing:
            raise ValueError(f"rep_features is missing required columns: {missing}")


if __name__ == "__main__":
    # Example usage. Feature extraction (read_segmentation,
    # analyze_arm_abduction_final, analyze_squat) requires the REHAB24-6
    # dataset — this block only runs end-to-end on Kaggle, same as the
    # rest of this repository.
    scorer = RQIScorer()

    scorer.fit_arm_model()
    example_arm_rep = {
        "peak_elevation": 95.0,
        "peak_elbow": 150.0,
        "max_trunk_lean": 12.0,
        "mean_trunk_lean": 6.0,
        "elevation_rom": 40.0,
    }
    print(scorer.score_rep("arm_abduction", example_arm_rep))

    seg = read_segmentation(SEGMENTATION_CSV)
    squat_df = analyze_squat(seg)
    example_patient_id = squat_df["video_id"].iloc[0]
    patient_reps = squat_df[squat_df["video_id"] == example_patient_id]
    scorer.fit_squat_calibration(example_patient_id, patient_reps)

    example_squat_rep = {
        "peak_knee": 140.0,
        "peak_hip": 150.0,
        "trunk_lean_at_peak": 15.0,
        "max_trunk_lean": 18.0,
    }
    print(scorer.score_rep("squat", example_squat_rep, patient_id=example_patient_id))