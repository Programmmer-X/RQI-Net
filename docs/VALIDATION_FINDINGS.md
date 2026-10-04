# RQI-Net — Ground-Truth Validation Findings

All results below are from the leakage-corrected pipeline (`ml_utils.py`
scaling fixed to fit per-fold, not on the full dataset before splitting;
`personal_calibration_*.py` scaling fixed to fit per held-out rep).

## 1. Dataset Summary

| Property | Value |
|---|---|
| Dataset | REHAB24-6 |
| Exercises in scope | Ex1 (Arm Abduction), Ex6 (Squat) |
| Ex1 reps / subjects | 178 / 13 |
| Ex6 reps / subjects | 191 / 9 (130 correct / 61 incorrect) |

## 2. Arm-Abduction (Ex1) Validation Findings

**Cross-validated comparison (GroupKFold, grouped by subject), full feature set:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.596 | 0.632 | 0.478 | 0.544 | 0.605 |
| Logistic Regression | 0.511 | 0.515 | 0.556 | 0.535 | 0.520 |

Winner by F1 and by ROC-AUC: Random Forest. Both models near chance;
Random Forest is the stronger of the two but still weak (AUC 0.605).

**Reduced feature set** (`mean_trunk_lean` dropped — collinear with `max_trunk_lean`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (reduced) | 0.584 | 0.608 | 0.500 | 0.549 | 0.601 |
| Logistic Regression (reduced) | 0.483 | 0.490 | 0.522 | 0.505 | 0.510 |

Dropping the collinear feature did not materially change performance.

**Interpretation**: performance is near chance level. No clinically useful
discrimination demonstrated from population-level models for Ex1.

## 3. Squat (Ex6) Validation Findings

**Cross-validated comparison (GroupKFold, grouped by subject):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.686 | 0.684 | 1.000 | 0.812 | 0.528 |
| Random Forest | 0.660 | 0.692 | 0.900 | 0.783 | 0.526 |

Winner by ROC-AUC: Logistic Regression — but the margin (0.528 vs 0.526) is
not meaningful; both are at chance level. Logistic Regression's recall of
1.000 combined with precision ≈ the dataset's correct-class base rate
(130/191 ≈ 0.68) indicates a near-constant "always predict correct"
prediction pattern rather than learned discrimination — F1 (0.812) looks
strong only because of this, which is why ROC-AUC, not F1, is the metric
of record under this class imbalance.

**Interpretation**: classification performance remains close to chance
level despite the higher F1, which is an artifact of class imbalance
rather than evidence of discrimination.

## 4. Personal-Calibration Findings

Leave-one-rep-out nearest-centroid classification within each subject.

| Exercise | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Squat | 0.487 | 0.682 | 0.462 | 0.550 | 0.556 |
| Arm Abduction | 0.489 | 0.494 | 0.456 | 0.474 | 0.486 |

Squat's personal-calibration ROC-AUC (0.556) is modestly above its
population-model ROC-AUC (0.526–0.528). Arm Abduction's personal-
calibration ROC-AUC (0.486) is at or slightly below chance, and below its
own population-model ROC-AUC (0.605) — personal calibration did not help
Ex1 in this verification.

## 5. Key Design Conclusions

- Population models perform near chance level for both exercises (best
  AUC: Ex1 Random Forest 0.605; Squat ≈ 0.526–0.528).
- Per-patient calibration shows only weak signal, and only for one of the
  two exercises (Squat 0.556; Ex1 0.486).
- Results do not support fixed/global angle thresholds for either
  exercise.
- Results support retaining a patient-specific calibration architecture
  for Squat; Ex1 calibration performance is not yet established as
  better than the population approach.
- RQI should remain a continuous score rather than a binary classifier —
  supported directly by Squat's degenerate Logistic Regression behavior
  above (recall=1.000, F1 inflated by class imbalance), which shows a
  hard binary decision is unreliable at this performance level.

## 6. Limitations

- Small subject counts (9 for Squat, 13 for Ex1) limit the statistical
  power of all cross-validated and leave-one-out estimates above; no
  confidence intervals are included in this document because none were
  computed in the supplied verification outputs.
- Squat's class imbalance (130 correct / 61 incorrect) makes F1 an
  unreliable ranking metric, as demonstrated directly by the Logistic
  Regression result in Section 3.
- Personal calibration required a global-centroid fallback for subjects
  with zero remaining reps of one class after leave-one-out (1 fallback
  for Squat, 5 for Arm Abduction) — performance for those specific reps
  is not purely personal-calibration-derived.
- Personal calibration does not yet show benefit for Arm Abduction
  (ROC-AUC 0.486, below its own population-model result) — the
  architecture decision in Section 5 to favor per-patient calibration is
  supported for Squat only, not uniformly across both exercises.
- Earlier single-feature exploratory screening (alternative elevation
  definitions; elbow, trunk-lean, and ROM univariate tests) preceded this
  verification pass and is not restated here, per the instruction to use
  only the outputs supplied for this update. Those tests used no
  scaler/cross-validation and are not affected by the leakage bug; see
  repository history for those results.

## 7. Traceability: Architecture Decisions → Evidence

| Decision | Evidence |
|---|---|
| Reject fixed/global angle thresholds | Ex1 population ROC-AUC 0.605 (best case); Squat population ROC-AUC 0.526–0.528 — both near chance (Sections 2–3) |
| Use ROC-AUC, not F1, as the ranking metric | Squat Logistic Regression: F1=0.812 (looks strong) vs ROC-AUC=0.528 (chance), driven by recall=1.000 under 130/61 class imbalance (Section 3) |
| RQI as continuous score, not binary classifier | Same Squat Logistic Regression result — a binary decision rule at this performance level is unreliable (Sections 3, 5) |
| Retain per-patient calibration architecture for Squat | Personal-calibration ROC-AUC 0.556 > population ROC-AUC 0.526–0.528 (Section 4) |
| Do not yet extend calibration claim to Arm Abduction | Personal-calibration ROC-AUC 0.486 < population ROC-AUC 0.605 for Ex1 (Section 4) |
| Calibration protocol requires both correct and incorrect reps per patient | Fallback to global centroid triggered when a subject lacked reps of one class (1/191 Squat, 5/178 Ex1) (Section 4, Limitations) |