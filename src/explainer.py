"""
explainer.py
============
SHAP-based model explainability utilities.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)


def get_shap_explainer(model, X_background, model_type: str = "tree"):
    """Create appropriate SHAP explainer for the model type."""
    if model_type == "tree":
        return shap.TreeExplainer(model)
    elif model_type == "linear":
        return shap.LinearExplainer(model, X_background)
    else:
        return shap.KernelExplainer(model.predict_proba, X_background[:100])


def shap_summary_plot(shap_values, X, title: str, save_path=None):
    """Generate and optionally save a SHAP summary (beeswarm) plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def shap_bar_plot(shap_values, X, title: str, max_display: int = 15, save_path=None):
    """Generate SHAP mean absolute importance bar plot."""
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, plot_type="bar",
                      max_display=max_display, show=False)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def shap_force_plot_single(explainer, shap_values, X, idx: int, label: str, save_path=None):
    """Generate a SHAP force plot for a single prediction."""
    shap.initjs()
    plot = shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X.iloc[idx],
        matplotlib=True,
        show=False,
    )
    plt.title(f"Force Plot — {label}", fontsize=11)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    return plot


def find_example_indices(y_test, y_pred, y_proba, n=1):
    """Return indices of TP, FP, FN examples for force plot analysis."""
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

    # Pick highest-confidence examples
    tp = tp_idx[np.argsort(y_proba[tp_idx])[-n:]]
    fp = fp_idx[np.argsort(y_proba[fp_idx])[-n:]]
    fn = fn_idx[np.argsort(y_proba[fn_idx])[:n]]  # lowest proba among FN

    return {"TP": tp, "FP": fp, "FN": fn}
