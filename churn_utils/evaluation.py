"""Evaluation helpers for churn models."""

import numpy as np
from sklearn.metrics import roc_auc_score, classification_report


def evaluate_classifier(y_true, y_proba, threshold: float = 0.5):
    """Print classification report and return ROC-AUC."""
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC: {auc:.3f}")
    print("Classification report @ threshold", threshold)
    print(classification_report(y_true, y_pred))
    return auc


def lift_top_decile(y_true, y_proba, top_fraction: float = 0.1):
    """Compute lift in the top X% of predicted risk."""
    n = len(y_true)
    k = int(np.floor(top_fraction * n))

    order = np.argsort(-y_proba)
    top_idx = order[:k]

    baseline_rate = y_true.mean()
    top_rate = y_true[top_idx].mean()

    lift = top_rate / baseline_rate if baseline_rate > 0 else np.nan
    print(f"Baseline churn rate: {baseline_rate:.2%}")
    print(f"Top {top_fraction:.0%} segment churn rate: {top_rate:.2%}")
    print(f"Lift: {lift:.2f}x")
    return lift
