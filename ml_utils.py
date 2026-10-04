"""
ml_utils.py — shared by ml_baseline_arm_abduction.py and ml_baseline_squat.py.
"""
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(name, model, X, y, groups, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = cross_val_predict(model, X, y, cv=gkf, groups=groups, method="predict")
    y_proba = cross_val_predict(model, X, y, cv=gkf, groups=groups, method="predict_proba")[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }