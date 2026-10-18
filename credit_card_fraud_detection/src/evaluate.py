"""
Evaluation Module for Credit Card Fraud Detection.
Calculates PR-AUC, ROC-AUC, Precision, Recall, F1, F2-score,
confusion matrices, and cost-calibrated decision thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)


def evaluate_model_performance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    model_name: str = "Model"
) -> dict:
    """
    Evaluates a trained classifier on test data with standard fraud detection metrics.
    """
    # Probability estimation
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-decision))
    else:
        y_proba = model.predict(X_test)

    # Class predictions at specified threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except Exception:
        pr_auc = 0.0

    return {
        "model_name": model_name,
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "f2_score": float(f2),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "probabilities": y_proba
    }


def find_optimal_threshold(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 200.0,
    cost_fp: float = 15.0
) -> dict:
    """
    Evaluates thresholds from 0.01 to 0.99 to find:
    1. Maximum F1-score threshold.
    2. Maximum F2-score threshold (recall-heavy).
    3. Minimum financial cost threshold.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1.0
    best_f1_thresh = 0.5
    best_f2 = -1.0
    best_f2_thresh = 0.5
    min_cost = float('inf')
    best_cost_thresh = 0.5

    records = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
        total_cost = (fn * cost_fn) + (fp * cost_fp)

        records.append({
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "f2": float(f2),
            "cost": float(total_cost)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = t
        if f2 > best_f2:
            best_f2 = f2
            best_f2_thresh = t
        if total_cost < min_cost:
            min_cost = total_cost
            best_cost_thresh = t

    return {
        "best_f1_threshold": float(best_f1_thresh),
        "best_f1": float(best_f1),
        "best_f2_threshold": float(best_f2_thresh),
        "best_f2": float(best_f2),
        "best_cost_threshold": float(best_cost_thresh),
        "min_cost": float(min_cost),
        "records": records
    }
