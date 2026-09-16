"""
Threshold calibration and sweep module for PubMedCLIP prototype.

Sweeps threshold candidates on the validation split ('val') to find optimal
decision thresholds (overall macro F1 and per-finding), then evaluates the
calibrated thresholds on the held-out test split ('test') to prevent data leakage.
"""

import csv
import json
import logging
import os
import sys
from typing import Any

import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.inference.analyzer import get_prompt_texts
from app.models.pubmedclip import pubmedclip_model
from data.evaluation.dataset import EvaluationDataset
from evaluation.metrics import compute_multilabel_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    finding_names: list[str],
    threshold_candidates: list[float],
) -> list[dict[str, Any]]:
    """Run threshold sweep over a range of threshold candidates.

    Args:
        y_true: Ground truth binary matrix (N, K).
        y_prob: Continuous similarity matrix (N, K).
        finding_names: List of finding names.
        threshold_candidates: List of threshold values to test.

    Returns:
        List of dictionaries with metrics for each threshold candidate.
    """
    sweep_results = []

    for th in threshold_candidates:
        y_pred = (y_prob >= th).astype(int)
        metrics = compute_multilabel_metrics(y_true, y_pred, y_prob, finding_names)

        sweep_results.append({
            "threshold": round(float(th), 4),
            "macro_f1": metrics["macro_avg"]["f1"],
            "macro_precision": metrics["macro_avg"]["precision"],
            "macro_recall": metrics["macro_avg"]["recall"],
            "macro_specificity": metrics["macro_avg"]["specificity"],
            "micro_f1": metrics["micro_avg"]["f1"],
            "micro_precision": metrics["micro_avg"]["precision"],
            "micro_recall": metrics["micro_avg"]["recall"],
            "per_finding": {
                name: {
                    "precision": metrics["per_finding"][name]["precision"],
                    "recall": metrics["per_finding"][name]["recall"],
                    "f1": metrics["per_finding"][name]["f1"],
                }
                for name in finding_names
            },
        })

    return sweep_results


def find_optimal_thresholds(
    sweep_results: list[dict[str, Any]], finding_names: list[str]
) -> tuple[float, dict[str, float]]:
    """Determine optimal overall macro threshold and per-finding optimal thresholds.

    Args:
        sweep_results: Output from run_threshold_sweep.
        finding_names: List of finding names.

    Returns:
        Tuple of (optimal_macro_threshold, per_finding_optimal_thresholds_dict).
    """
    # 1. Best overall macro threshold (highest macro F1)
    best_macro_entry = max(sweep_results, key=lambda x: x["macro_f1"])
    optimal_macro_threshold = best_macro_entry["threshold"]

    # 2. Per-finding optimal thresholds (highest per-finding F1)
    per_finding_optimal: dict[str, float] = {}
    for name in finding_names:
        best_entry = max(sweep_results, key=lambda x: x["per_finding"][name]["f1"])
        per_finding_optimal[name] = best_entry["threshold"]

    return optimal_macro_threshold, per_finding_optimal


def calibrate_thresholds(
    output_dir: str = os.path.join("data", "evaluation", "results"),
) -> dict[str, Any]:
    """Execute complete threshold calibration pipeline.

    1. Load PubMedCLIP model and generate embeddings for validation ('val') and test ('test') splits.
    2. Sweep threshold candidates [0.15, 0.40] on 'val' split.
    3. Select optimal overall threshold and per-finding thresholds from 'val' split.
    4. Evaluate uncalibrated (default 0.25) vs calibrated threshold on held-out 'test' split.
    5. Save threshold_sweep.csv, calibration_results.json, and calibration_report.txt.
    """
    logger.info("Loading PubMedCLIP model for threshold calibration...")
    pubmedclip_model.load()
    prompt_texts = get_prompt_texts()
    text_embeddings = pubmedclip_model.get_text_embeddings(prompt_texts)

    dataset_loader = EvaluationDataset()

    # Load Validation Split (for tuning)
    val_samples, val_y_true, finding_names = dataset_loader.load_samples(split="val")
    val_prob_list = []
    for _, pil_img in val_samples:
        img_emb = pubmedclip_model.get_image_embedding(pil_img)
        sims = pubmedclip_model.compute_similarity(img_emb, text_embeddings)
        val_prob_list.append(sims.cpu().tolist())
    val_y_prob = np.array(val_prob_list, dtype=float)

    # Load Test Split (for unbiased evaluation)
    test_samples, test_y_true, _ = dataset_loader.load_samples(split="test")
    test_prob_list = []
    for _, pil_img in test_samples:
        img_emb = pubmedclip_model.get_image_embedding(pil_img)
        sims = pubmedclip_model.compute_similarity(img_emb, text_embeddings)
        test_prob_list.append(sims.cpu().tolist())
    test_y_prob = np.array(test_prob_list, dtype=float)

    # Define Candidate Grid: 0.15 to 0.40 with step 0.01
    candidates = [round(x, 2) for x in np.arange(0.15, 0.41, 0.01).tolist()]

    # 1. Sweep on Validation Split
    logger.info("Running threshold sweep on validation split (N=%d)...", len(val_samples))
    val_sweep = run_threshold_sweep(val_y_true, val_y_prob, finding_names, candidates)

    # 2. Select Optimal Thresholds from Validation
    opt_macro_th, opt_per_finding_th = find_optimal_thresholds(val_sweep, finding_names)
    default_th = settings.CONFIDENCE_THRESHOLD

    logger.info("Optimal Macro Threshold (Val Split): %.4f", opt_macro_th)
    for name, th_val in opt_per_finding_th.items():
        logger.info("  Optimal Threshold for %-18s: %.4f", name, th_val)

    # 3. Evaluate Uncalibrated vs Calibrated on HELD-OUT Test Split
    test_pred_default = (test_y_prob >= default_th).astype(int)
    test_metrics_default = compute_multilabel_metrics(test_y_true, test_pred_default, test_y_prob, finding_names)

    test_pred_calibrated = (test_y_prob >= opt_macro_th).astype(int)
    test_metrics_calibrated = compute_multilabel_metrics(test_y_true, test_pred_calibrated, test_y_prob, finding_names)

    # Evaluate per-finding calibrated predictions on Test
    test_pred_per_finding = np.zeros_like(test_y_true)
    for j, name in enumerate(finding_names):
        test_pred_per_finding[:, j] = (test_y_prob[:, j] >= opt_per_finding_th[name]).astype(int)
    test_metrics_per_finding_calibrated = compute_multilabel_metrics(
        test_y_true, test_pred_per_finding, test_y_prob, finding_names
    )

    os.makedirs(output_dir, exist_ok=True)

    # 4. Save threshold_sweep.csv
    sweep_csv_path = os.path.join(output_dir, "threshold_sweep.csv")
    with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "macro_precision", "macro_recall", "macro_specificity", "macro_f1", "micro_f1"])
        for entry in val_sweep:
            writer.writerow([
                entry["threshold"],
                entry["macro_precision"],
                entry["macro_recall"],
                entry["macro_specificity"],
                entry["macro_f1"],
                entry["micro_f1"],
            ])

    # 5. Save calibration_results.json
    calibration_output = {
        "calibration_metadata": {
            "validation_samples": len(val_samples),
            "test_samples": len(test_samples),
            "default_threshold": default_th,
            "optimal_macro_threshold": opt_macro_th,
            "optimal_per_finding_thresholds": opt_per_finding_th,
            "disclaimer": "Research prototype similarity threshold calibration. NOT clinically validated probabilities.",
        },
        "validation_sweep": val_sweep,
        "test_comparison": {
            "uncalibrated_default": {
                "threshold": default_th,
                "macro_avg": test_metrics_default["macro_avg"],
                "micro_avg": test_metrics_default["micro_avg"],
            },
            "calibrated_macro": {
                "threshold": opt_macro_th,
                "macro_avg": test_metrics_calibrated["macro_avg"],
                "micro_avg": test_metrics_calibrated["micro_avg"],
            },
            "calibrated_per_finding": {
                "thresholds": opt_per_finding_th,
                "macro_avg": test_metrics_per_finding_calibrated["macro_avg"],
                "micro_avg": test_metrics_per_finding_calibrated["micro_avg"],
            },
        },
    }

    results_json_path = os.path.join(output_dir, "calibration_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(calibration_output, f, indent=2)

    # 6. Save calibration_report.txt
    report_path = os.path.join(output_dir, "calibration_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CHEST X-RAY FINDING ANALYZER — THRESHOLD CALIBRATION REPORT (PHASE 17)\n")
        f.write("=" * 80 + "\n\n")

        f.write("CALIBRATION METHODOLOGY:\n")
        f.write("  1. Swept threshold candidates in range [0.15, 0.40] on Validation Split (N=%d).\n" % len(val_samples))
        f.write("  2. Selected optimal threshold maximizing Macro F1 score on Validation Split.\n")
        f.write("  3. Evaluated uncalibrated default (%.2f) vs calibrated threshold (%.4f) on held-out Test Split (N=%d).\n\n" % (default_th, opt_macro_th, len(test_samples)))

        f.write("OPTIMAL THRESHOLDS (SELECTION FROM VAL SPLIT):\n")
        f.write(f"  Default Threshold       : {default_th:.4f}\n")
        f.write(f"  Optimal Macro Threshold : {opt_macro_th:.4f}\n")
        f.write("  Optimal Per-Finding Thresholds:\n")
        for name, th_val in opt_per_finding_th.items():
            f.write(f"    - {name:<18}: {th_val:.4f}\n")
        f.write("\n")

        f.write("HELD-OUT TEST SET EVALUATION COMPARISON:\n")
        f.write("  Config                  | Threshold | Precision | Recall   | Specificity | F1 Score\n")
        f.write("  ------------------------+-----------+-----------+----------+-------------+---------\n")
        f.write(f"  Uncalibrated (Default)  | {default_th:<9.4f} | {test_metrics_default['macro_avg']['precision']:<9.4f} | {test_metrics_default['macro_avg']['recall']:<8.4f} | {test_metrics_default['macro_avg']['specificity']:<11.4f} | {test_metrics_default['macro_avg']['f1']:<8.4f}\n")
        f.write(f"  Calibrated (Macro)      | {opt_macro_th:<9.4f} | {test_metrics_calibrated['macro_avg']['precision']:<9.4f} | {test_metrics_calibrated['macro_avg']['recall']:<8.4f} | {test_metrics_calibrated['macro_avg']['specificity']:<11.4f} | {test_metrics_calibrated['macro_avg']['f1']:<8.4f}\n")
        f.write(f"  Calibrated (Per-Finding)| Dynamic   | {test_metrics_per_finding_calibrated['macro_avg']['precision']:<9.4f} | {test_metrics_per_finding_calibrated['macro_avg']['recall']:<8.4f} | {test_metrics_per_finding_calibrated['macro_avg']['specificity']:<11.4f} | {test_metrics_per_finding_calibrated['macro_avg']['f1']:<8.4f}\n\n")

        f.write("MEDICAL DISCLAIMER & TERMINOLOGY:\n")
        f.write("  Prototype similarity scores are normalized vector dot-products and do NOT represent\n")
        f.write("  clinically validated probabilities of disease presence. Threshold calibration optimizes\n")
        f.write("  decision boundaries for prototype classification and is NOT a clinical diagnosis.\n")

    logger.info("Saved calibration report to '%s'", report_path)
    return calibration_output


def print_calibration_summary(calibration_res: dict[str, Any]) -> None:
    """Print formatted terminal output for threshold calibration."""
    print("\n" + "=" * 80)
    print("CHEST X-RAY FINDING ANALYZER — PHASE 17 THRESHOLD CALIBRATION SUMMARY")
    print("=" * 80)
    meta = calibration_res["calibration_metadata"]
    test_comp = calibration_res["test_comparison"]

    print(f"Validation Samples: {meta['validation_samples']} | Test Samples: {meta['test_samples']}")
    print(f"Default Threshold: {meta['default_threshold']} -> Optimal Macro Threshold: {meta['optimal_macro_threshold']}")
    print("-" * 80)

    print("Per-Finding Optimal Thresholds (Selected from Validation Split):")
    for name, th_val in meta["optimal_per_finding_thresholds"].items():
        print(f"  - {name:<18}: {th_val:.4f}")

    print("\nHELD-OUT TEST SET EVALUATION COMPARISON:")
    print(f"{'Configuration':<24} | {'Threshold':<9} | {'Precision':<9} | {'Recall':<8} | {'Specificity':<11} | {'F1 Score':<8}")
    print("-" * 80)

    def_m = test_comp["uncalibrated_default"]["macro_avg"]
    cal_m = test_comp["calibrated_macro"]["macro_avg"]
    pf_m = test_comp["calibrated_per_finding"]["macro_avg"]

    print(f"{'Uncalibrated (Default)':<24} | {meta['default_threshold']:<9.4f} | {def_m['precision']:<9.4f} | {def_m['recall']:<8.4f} | {def_m['specificity']:<11.4f} | {def_m['f1']:<8.4f}")
    print(f"{'Calibrated (Macro)':<24} | {meta['optimal_macro_threshold']:<9.4f} | {cal_m['precision']:<9.4f} | {cal_m['recall']:<8.4f} | {cal_m['specificity']:<11.4f} | {cal_m['f1']:<8.4f}")
    print(f"{'Calibrated (Per-Finding)':<24} | {'Dynamic':<9} | {pf_m['precision']:<9.4f} | {pf_m['recall']:<8.4f} | {pf_m['specificity']:<11.4f} | {pf_m['f1']:<8.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    res = calibrate_thresholds()
    print_calibration_summary(res)
