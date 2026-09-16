"""
Tests for Phase 16 evaluation metrics and evaluation pipeline.
"""

import os
import numpy as np
import pytest

from evaluation.metrics import compute_binary_metrics, compute_multilabel_metrics
from evaluation.evaluate import run_evaluation


def test_binary_metrics_perfect_score():
    """Test binary metrics with perfect predictions."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    m = compute_binary_metrics(y_true, y_pred)

    assert m["tp"] == 2
    assert m["tn"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["specificity"] == 1.0
    assert m["f1"] == 1.0


def test_binary_metrics_zero_division():
    """Test binary metrics handle zero-division safely."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    m = compute_binary_metrics(y_true, y_pred)

    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["specificity"] == 1.0  # TN / (TN + FP) = 4 / 4 = 1.0


def test_multilabel_metrics_macro_micro():
    """Test multi-label metrics calculation across multiple findings."""
    y_true = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    y_pred = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 0],
    ])
    y_prob = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.4, 0.7],
        [0.1, 0.1],
    ])
    names = ["finding_a", "finding_b"]

    res = compute_multilabel_metrics(y_true, y_pred, y_prob, names)

    assert "finding_a" in res["per_finding"]
    assert "finding_b" in res["per_finding"]
    assert "macro_avg" in res
    assert "micro_avg" in res

    assert 0.0 <= res["macro_avg"]["f1"] <= 1.0
    assert 0.0 <= res["micro_avg"]["f1"] <= 1.0


def test_evaluation_pipeline_execution(tmp_path):
    """Test full evaluation pipeline execution on test split."""
    out_dir = str(tmp_path / "results")
    res = run_evaluation(split="test", output_dir=out_dir)

    assert res["evaluation_metadata"]["split"] == "test"
    assert "per_finding_metrics" in res
    assert os.path.exists(os.path.join(out_dir, "predictions.csv"))
    assert os.path.exists(os.path.join(out_dir, "metrics.json"))
    assert os.path.exists(os.path.join(out_dir, "evaluation_report.txt"))

    cm_dir = os.path.join(out_dir, "confusion_matrices")
    assert os.path.exists(cm_dir)
    assert len(os.listdir(cm_dir)) == 8
