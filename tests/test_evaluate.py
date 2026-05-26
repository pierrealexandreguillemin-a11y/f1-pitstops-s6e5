"""Tests for src/evaluate.py - metrics + calibration."""

from __future__ import annotations

import numpy as np


def test_compute_auc_and_logloss_returns_dict():
    from src.evaluate import compute_metrics

    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    metrics = compute_metrics(y_true, y_pred)
    assert "auc" in metrics
    assert "logloss" in metrics
    assert "brier" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["logloss"] >= 0.0
    assert 0.0 <= metrics["brier"] <= 1.0


def test_compute_metrics_perfect_classifier_auc_is_1():
    from src.evaluate import compute_metrics

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["auc"] == 1.0
