"""
ml_utils.py — shared by ml_baseline_arm_abduction.py and ml_baseline_squat.py.

evaluate_model wraps the model in a StandardScaler+model Pipeline internally,
so the scaler is fit fresh on each fold's TRAINING data only, via
cross_val_predict. Callers must pass RAW (unscaled) X — scaling before
calling this function (then passing pre-scaled X in) leaks test-fold
statistics into the scaling applied to held-out samples. This was a real
bug in earlier versions of the baseline scripts, fixed here.
"""
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_model(name, model, X, y, groups, n_splits):
    """X must be RAW (unscaled) features — scaling happens inside the pipeline, per fold."""
    pipeline = make_pipeline(StandardScaler(), model)
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = cross_val_predict(pipeline, X, y, cv=gkf, groups=groups, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, cv=gkf, groups=groups, method="predict_proba")[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }