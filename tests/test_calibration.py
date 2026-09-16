"""
Tests for Phase 17 threshold calibration and sweep pipeline.
"""

import os
import numpy as np
import pytest

from evaluation.calibrate import (
    calibrate_thresholds,
    find_optimal_thresholds,
    run_threshold_sweep,
)


def test_threshold_sweep_metrics():
    """Test running threshold sweep over candidate grid."""
    y_true = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    y_prob = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.4, 0.7],
        [0.1, 0.1],
    ])
    names = ["finding_a", "finding_b"]
    candidates = [0.2, 0.5, 0.8]

    sweep = run_threshold_sweep(y_true, y_prob, names, candidates)

    assert len(sweep) == 3
    assert sweep[0]["threshold"] == 0.2
    assert "macro_f1" in sweep[0]
    assert "per_finding" in sweep[0]


def test_find_optimal_thresholds_selection():
    """Test optimal threshold extraction logic."""
    names = ["finding_a", "finding_b"]
    sweep = [
        {
            "threshold": 0.2,
            "macro_f1": 0.5,
            "per_finding": {"finding_a": {"f1": 0.6}, "finding_b": {"f1": 0.4}},
        },
        {
            "threshold": 0.3,
            "macro_f1": 0.8,
            "per_finding": {"finding_a": {"f1": 0.8}, "finding_b": {"f1": 0.8}},
        },
        {
            "threshold": 0.4,
            "macro_f1": 0.6,
            "per_finding": {"finding_a": {"f1": 0.5}, "finding_b": {"f1": 0.7}},
        },
    ]

    opt_macro, opt_pf = find_optimal_thresholds(sweep, names)

    assert opt_macro == 0.3
    assert opt_pf["finding_a"] == 0.3
    assert opt_pf["finding_b"] == 0.3


def test_calibration_pipeline_execution(tmp_path):
    """Test running threshold calibration pipeline end-to-end."""
    out_dir = str(tmp_path / "results")
    res = calibrate_thresholds(output_dir=out_dir)

    assert "calibration_metadata" in res
    assert os.path.exists(os.path.join(out_dir, "threshold_sweep.csv"))
    assert os.path.exists(os.path.join(out_dir, "calibration_results.json"))
    assert os.path.exists(os.path.join(out_dir, "calibration_report.txt"))
