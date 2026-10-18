"""
Training Pipeline for Credit Card Fraud Detection.
Trains and compares Logistic Regression, Random Forest, and XGBoost,
evaluating both Class Weighting and SMOTE imbalance handling techniques.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.data_loader import load_credit_card_data
from src.preprocessor import prepare_train_test_data
from src.evaluate import evaluate_model_performance, find_optimal_threshold


def train_and_compare_models(data_path: str = None, samples: int = 100000, models_dir: str = None):
    """
    Executes end-to-end model training, comparison, and serialization.
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    print("\n=======================================================")
    print("STEP 1: Loading Dataset")
    print("=======================================================")
    df = load_credit_card_data(data_path=data_path, fallback_samples=samples)

    print("\n=======================================================")
    print("STEP 2: Data Preprocessing & Leakage-Free Train-Test Split")
    print("=======================================================")
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(df, test_size=0.2, random_state=42)

    # Save fitted preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)

    # Also save test set slice for evaluation and app benchmarking
    test_data_path = os.path.join(models_dir, "test_data.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_data_path)

    # Calculate class weighting ratio for XGBoost
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(1, n_pos)
    print(f"[Training] Class balance ratio: {n_neg:,} negative / {n_pos:,} positive -> scale_pos_weight: {scale_pos_weight:.2f}")

    print("\n=======================================================")
    print("STEP 3: Preparing SMOTE Resampled Training Set (Leakage-Free)")
    print("=======================================================")
    # Apply SMOTE strictly to training data only!
    # Sampling strategy 0.1 oversamples fraud to 10% of majority class
    smote = SMOTE(sampling_strategy=0.1, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE] Original training shape: {X_train.shape}, Frauds: {y_train.sum():,}")
    print(f"[SMOTE] Resampled training shape: {X_train_smote.shape}, Frauds: {y_train_smote.sum():,}")

    models_config = {
        "Logistic Regression (Class Weighted)": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "X": X_train,
            "y": y_train,
            "file": "logistic_regression.joblib"
        },
        "Random Forest (Class Weighted)": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
            "X": X_train,
            "y": y_train,
            "file": "random_forest.joblib"
        },
        "XGBoost (Cost-Sensitive Weighted)": {
            "model": XGBClassifier(scale_pos_weight=scale_pos_weight, max_depth=5, learning_rate=0.1, n_estimators=100, eval_metric="logloss", random_state=42),
            "X": X_train,
            "y": y_train,
            "file": "xgboost.joblib"
        },
        "Logistic Regression + SMOTE": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "X": X_train_smote,
            "y": y_train_smote,
            "file": "logistic_regression_smote.joblib"
        },
        "XGBoost + SMOTE": {
            "model": XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, eval_metric="logloss", random_state=42),
            "X": X_train_smote,
            "y": y_train_smote,
            "file": "xgboost_smote.joblib"
        }
    }

    metrics_results = {}
    best_pr_auc = -1.0
    best_model_name = ""

    print("\n=======================================================")
    print("STEP 4: Training & Evaluating Models")
    print("=======================================================")

    for name, cfg in models_config.items():
        print(f"\n--- Training {name} ---")
        clf = cfg["model"]
        clf.fit(cfg["X"], cfg["y"])

        # Serialize trained model
        save_path = os.path.join(models_dir, cfg["file"])
        joblib.dump(clf, save_path)
        print(f"[Model] Saved to {save_path}")

        # Evaluate on untouched test set
        eval_dict = evaluate_model_performance(clf, X_test, y_test, threshold=0.5, model_name=name)
        opt_thresh = find_optimal_threshold(y_test.values, eval_dict["probabilities"])

        eval_dict["optimal_thresholds"] = {
            "best_f1_threshold": opt_thresh["best_f1_threshold"],
            "best_f1": opt_thresh["best_f1"],
            "best_f2_threshold": opt_thresh["best_f2_threshold"],
            "best_f2": opt_thresh["best_f2"],
            "best_cost_threshold": opt_thresh["best_cost_threshold"],
            "min_cost": opt_thresh["min_cost"]
        }

        # Exclude raw probabilities array for JSON serialization
        json_eval = {k: v for k, v in eval_dict.items() if k != "probabilities"}
        metrics_results[name] = json_eval

        print(f"Results for {name}:")
        print(f"  Precision: {eval_dict['precision']:.4f} | Recall: {eval_dict['recall']:.4f} | F1: {eval_dict['f1_score']:.4f} | F2: {eval_dict['f2_score']:.4f}")
        print(f"  ROC-AUC:   {eval_dict['roc_auc']:.4f} | PR-AUC: {eval_dict['pr_auc']:.4f}")
        print(f"  Optimal F1 Threshold: {opt_thresh['best_f1_threshold']:.2f} (F1 = {opt_thresh['best_f1']:.4f})")
        print(f"  Confusion Matrix: TP={eval_dict['confusion_matrix']['tp']}, FP={eval_dict['confusion_matrix']['fp']}, FN={eval_dict['confusion_matrix']['fn']}, TN={eval_dict['confusion_matrix']['tn']}")

        if eval_dict['pr_auc'] > best_pr_auc:
            best_pr_auc = eval_dict['pr_auc']
            best_model_name = name

    # Record best model reference
    metrics_results["_best_model"] = best_model_name
    metrics_file = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_results, f, indent=2)
    print(f"\n[CHAMPION] Best Performing Model (by PR-AUC): {best_model_name} (PR-AUC = {best_pr_auc:.4f})")

    # Link primary production model
    best_cfg = models_config[best_model_name]
    best_model_copy = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_cfg["model"], best_model_copy)
    print(f"[Model] Saved champion model copy to {best_model_copy}")

    return metrics_results


if __name__ == "__main__":
    train_and_compare_models(samples=100000)
