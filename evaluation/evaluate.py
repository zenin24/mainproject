"""
Evaluation script for PubMedCLIP Chest X-ray Finding Analyzer prototype.

Runs zero-shot inference on evaluation dataset using exact production pipeline,
computes binary & multi-label evaluation metrics, saves predictions.csv, metrics.json,
evaluation_report.txt, and confusion matrices.
"""

import csv
import json
import logging
import os
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.inference.analyzer import get_prompt_texts
from app.models.pubmedclip import pubmedclip_model
from data.evaluation.dataset import EvaluationDataset
from evaluation.metrics import compute_multilabel_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_confusion_matrix_image(
    tp: int, fp: int, tn: int, fn: int, finding_name: str, output_path: str
) -> None:
    """Generate a clean visual confusion matrix image using PIL."""
    img = Image.new("RGB", (400, 320), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((20, 15), f"Confusion Matrix: {finding_name}", fill=(20, 25, 40))

    # Matrix Headers
    draw.text((160, 45), "Pred Positive", fill=(80, 80, 100))
    draw.text((270, 45), "Pred Negative", fill=(80, 80, 100))
    draw.text((20, 95), "Actual Positive", fill=(80, 80, 100))
    draw.text((20, 195), "Actual Negative", fill=(80, 80, 100))

    # Boxes (TP: green tint, FP: red tint, FN: orange tint, TN: blue tint)
    # TP
    draw.rectangle([140, 75, 240, 155], fill=(220, 245, 220), outline=(100, 180, 100), width=2)
    draw.text((170, 95), f"TP: {tp}", fill=(20, 80, 20))

    # FN
    draw.rectangle([250, 75, 350, 155], fill=(255, 235, 220), outline=(220, 140, 80), width=2)
    draw.text((280, 95), f"FN: {fn}", fill=(140, 60, 20))

    # FP
    draw.rectangle([140, 175, 240, 255], fill=(255, 225, 225), outline=(220, 100, 100), width=2)
    draw.text((170, 195), f"FP: {fp}", fill=(140, 20, 20))

    # TN
    draw.rectangle([250, 175, 350, 255], fill=(220, 235, 255), outline=(100, 140, 220), width=2)
    draw.text((280, 195), f"TN: {tn}", fill=(20, 40, 140))

    # Footer note
    draw.text((20, 280), "PubMedCLIP Zero-Shot Evaluation", fill=(120, 120, 140))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


def run_evaluation(
    split: str = "test",
    output_dir: str = os.path.join("data", "evaluation", "results"),
) -> dict[str, Any]:
    """Run full evaluation pipeline on designated split.

    Args:
        split: 'test', 'val', or 'all'.
        output_dir: Output directory path for evaluation artifacts.

    Returns:
        Structured evaluation results dictionary.
    """
    logger.info("Initializing PubMedCLIP model for evaluation...")
    pubmedclip_model.load()
    device_str = str(pubmedclip_model.device)
    logger.info("PubMedCLIP running on device: %s", device_str)

    # Load dataset via Phase 15 loader (ensures identical preprocessing)
    dataset_loader = EvaluationDataset()
    samples, y_true, finding_names = dataset_loader.load_samples(split=split)

    if not samples:
        raise ValueError(f"No samples found in split '{split}'")

    logger.info("Loaded %d images for split '%s'. Generating embeddings...", len(samples), split)

    # 1. Text embeddings for controlled medical prompts
    prompt_texts = get_prompt_texts()
    text_embeddings = pubmedclip_model.get_text_embeddings(prompt_texts)

    # 2. Run inference over evaluation images
    raw_predictions: list[dict[str, Any]] = []
    y_prob_list: list[list[float]] = []

    for img_id, pil_img in samples:
        img_emb = pubmedclip_model.get_image_embedding(pil_img)
        sims = pubmedclip_model.compute_similarity(img_emb, text_embeddings)
        scores = sims.cpu().tolist()
        y_prob_list.append(scores)

    y_prob = np.array(y_prob_list, dtype=float)

    # 3. Apply prototype threshold (fixed at settings.CONFIDENCE_THRESHOLD)
    threshold = settings.CONFIDENCE_THRESHOLD
    y_pred = (y_prob >= threshold).astype(int)

    # 4. Save raw predictions CSV
    os.makedirs(output_dir, exist_ok=True)
    pred_csv_path = os.path.join(output_dir, "predictions.csv")

    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id",
            "split",
            "finding",
            "ground_truth",
            "similarity_score",
            "threshold_used",
            "prediction",
            "above_threshold",
        ])

        for i, (img_id, _) in enumerate(samples):
            for j, finding in enumerate(finding_names):
                gt = int(y_true[i, j])
                score = round(float(y_prob[i, j]), 4)
                pred = int(y_pred[i, j])
                writer.writerow([
                    img_id,
                    split,
                    finding,
                    gt,
                    score,
                    threshold,
                    pred,
                    pred == 1,
                ])

    logger.info("Saved raw evaluation predictions to '%s'", pred_csv_path)

    # 5. Calculate Metrics
    metrics_res = compute_multilabel_metrics(y_true, y_pred, y_prob, finding_names)

    # Add metadata
    evaluation_output = {
        "evaluation_metadata": {
            "dataset": "NIH_ChestXray14_Standard_Subset",
            "split": split,
            "sample_count": len(samples),
            "model_checkpoint": settings.MODEL_NAME,
            "device": device_str,
            "threshold_used": threshold,
            "disclaimer": "Research prototype evaluation. Not for clinical diagnosis.",
        },
        "per_finding_metrics": metrics_res["per_finding"],
        "macro_average": metrics_res["macro_avg"],
        "micro_average": metrics_res["micro_avg"],
    }

    # 6. Save machine-readable JSON
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_output, f, indent=2)
    logger.info("Saved machine-readable metrics to '%s'", metrics_json_path)

    # 7. Generate Confusion Matrices
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    for finding, m in metrics_res["per_finding"].items():
        cm_path = os.path.join(cm_dir, f"{finding}_confusion_matrix.png")
        generate_confusion_matrix_image(
            tp=m["tp"],
            fp=m["fp"],
            tn=m["tn"],
            fn=m["fn"],
            finding_name=finding,
            output_path=cm_path,
        )

    # 8. Generate Human-Readable Report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CHEST X-RAY FINDING ANALYZER — MODEL EVALUATION REPORT (PHASE 16)\n")
        f.write("=" * 80 + "\n\n")

        f.write("EVALUATION METADATA:\n")
        f.write(f"  Model Checkpoint : {settings.MODEL_NAME}\n")
        f.write(f"  Compute Device   : {device_str}\n")
        f.write(f"  Evaluation Split : {split} (N = {len(samples)})\n")
        f.write(f"  Threshold Used   : {threshold} (Fixed Prototype Threshold)\n")
        f.write("  Disclaimer       : Research prototype evaluation. Not for clinical diagnosis.\n\n")

        f.write("PER-FINDING METRICS:\n")
        header = f"{'Finding':<18} | {'Positives':<9} | {'Precision':<9} | {'Recall':<8} | {'Specificity':<11} | {'F1 Score':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")

        for finding, m in metrics_res["per_finding"].items():
            roc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
            pr_str = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "N/A"
            f.write(
                f"{finding:<18} | {m['total_positives']:<9} | {m['precision']:<9.4f} | "
                f"{m['recall']:<8.4f} | {m['specificity']:<11.4f} | {m['f1']:<8.4f} | "
                f"{roc_str:<8} | {pr_str:<8}\n"
            )

        f.write("\nAGGREGATED MULTI-LABEL METRICS:\n")
        macro = metrics_res["macro_avg"]
        micro = metrics_res["micro_avg"]
        f.write(f"  Macro Average  -> Precision: {macro['precision']:.4f}, Recall: {macro['recall']:.4f}, Specificity: {macro['specificity']:.4f}, F1: {macro['f1']:.4f}\n")
        f.write(f"  Micro Average  -> Precision: {micro['precision']:.4f}, Recall: {micro['recall']:.4f}, Specificity: {micro['specificity']:.4f}, F1: {micro['f1']:.4f}\n")

    logger.info("Saved human-readable evaluation report to '%s'", report_path)
    return evaluation_output


def print_summary_table(results: dict[str, Any]) -> None:
    """Print clean formatted markdown evaluation summary table."""
    print("\n" + "=" * 80)
    print("CHEST X-RAY FINDING ANALYZER — PHASE 16 EVALUATION SUMMARY")
    print("=" * 80)
    meta = results["evaluation_metadata"]
    print(f"Device: {meta['device']} | Model: {meta['model_checkpoint']} | Split: {meta['split']} (N={meta['sample_count']})")
    print(f"Prototype Threshold Used: {meta['threshold_used']}")
    print("-" * 80)

    print(f"{'Finding':<18} | {'Positives':<9} | {'Precision':<9} | {'Recall':<8} | {'Specificity':<11} | {'F1':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8}")
    print("-" * 85)

    for finding, m in results["per_finding_metrics"].items():
        roc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        pr_str = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "N/A"
        print(
            f"{finding:<18} | {m['total_positives']:<9} | {m['precision']:<9.4f} | "
            f"{m['recall']:<8.4f} | {m['specificity']:<11.4f} | {m['f1']:<8.4f} | "
            f"{roc_str:<8} | {pr_str:<8}"
        )

    print("-" * 85)
    macro = results["macro_average"]
    micro = results["micro_average"]
    print(f"Macro Average      | -         | {macro['precision']:<9.4f} | {macro['recall']:<8.4f} | {macro['specificity']:<11.4f} | {macro['f1']:<8.4f} | -        | -")
    print(f"Micro Average      | -         | {micro['precision']:<9.4f} | {micro['recall']:<8.4f} | {micro['specificity']:<11.4f} | {micro['f1']:<8.4f} | -        | -")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    split_arg = sys.argv[1] if len(sys.argv) > 1 else "test"
    res = run_evaluation(split=split_arg)
    print_summary_table(res)
