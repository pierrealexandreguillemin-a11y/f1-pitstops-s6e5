"""Evaluation metrics + calibration plots for f1-pitstops-s6e5."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute AUC, log-loss, Brier score for binary classification."""
    return {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "logloss": float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7))),
        "brier": float(brier_score_loss(y_true, y_pred)),
    }


def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration",
    save_path: Path | None = None,
) -> None:
    """Reliability diagram - calibrated model lies on y=x."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
