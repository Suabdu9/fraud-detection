"""
trainer.py
==========
Model training, evaluation, and persistence utilities.
"""

import logging
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
    classification_report, roc_auc_score
)

logger = logging.getLogger(__name__)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def get_models(dataset: str = "fraud") -> dict:
    """Return dict of model name → estimator for a given dataset."""
    if dataset == "fraud":
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight="balanced",
                random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=10, eval_metric="aucpr",
                random_state=42, n_jobs=-1, verbosity=0),
            "LightGBM": LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
        }
    else:  # creditcard
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=578, eval_metric="aucpr",
                random_state=42, n_jobs=-1, verbosity=0),
            "LightGBM": LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
        }


def evaluate(model, X_test, y_test) -> dict:
    """Compute all evaluation metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    return {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "auc_pr": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": cm,
        "report": classification_report(y_test, y_pred),
    }


def cross_val_evaluate(model, X, y, k: int = 5) -> dict:
    """Run stratified k-fold CV and return mean ± std of key metrics."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scoring = ["f1", "average_precision", "roc_auc"]
    results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    return {
        "cv_f1_mean": results["test_f1"].mean(),
        "cv_f1_std": results["test_f1"].std(),
        "cv_auc_pr_mean": results["test_average_precision"].mean(),
        "cv_auc_pr_std": results["test_average_precision"].std(),
        "cv_roc_auc_mean": results["test_roc_auc"].mean(),
        "cv_roc_auc_std": results["test_roc_auc"].std(),
    }


def save_model(model, name: str, dataset: str):
    """Save fitted model to models/ directory."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{dataset}_{name}.joblib"
    joblib.dump(model, path)
    logger.info("Model saved: %s", path)
    return path


def load_model(name: str, dataset: str):
    """Load a saved model from models/ directory."""
    path = MODELS_DIR / f"{dataset}_{name}.joblib"
    return joblib.load(path)
