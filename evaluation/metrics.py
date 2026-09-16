"""
Metrics calculation module for multi-label chest X-ray evaluation.

Computes TP, FP, TN, FN, Precision, Recall/Sensitivity, Specificity, F1,
ROC-AUC, PR-AUC, Macro Average, and Micro Average.
"""

from typing import Any
import numpy as np


def compute_binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None
) -> dict[str, Any]:
    """Compute binary classification metrics for a single finding label.

    Args:
        y_true: Ground truth binary array (0 or 1).
        y_pred: Predicted binary array (0 or 1).
        y_prob: Continuous similarity/probability scores.

    Returns:
        Dictionary containing TP, FP, TN, FN, Precision, Recall, Specificity, F1,
        ROC-AUC, and PR-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total_positives = tp + fn
    total_negatives = tn + fp

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall / Sensitivity: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity: TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1 Score: 2 * P * R / (P + R)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # ROC-AUC & PR-AUC calculation if continuous scores provided
    roc_auc = _compute_roc_auc(y_true, y_prob) if y_prob is not None else None
    pr_auc = _compute_pr_auc(y_true, y_prob) if y_prob is not None else None

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_positives": total_positives,
        "total_negatives": total_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
    }


def _compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Compute ROC-AUC score safely. Requires at least one positive and one negative sample."""
    if len(np.unique(y_true)) < 2:
        return None  # Cannot compute ROC-AUC if only 1 class is present

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_prob))
    except ImportError:
        # Pure NumPy fallback via Mann-Whitney U statistic formulation
        pos_scores = y_prob[y_true == 1]
        neg_scores = y_prob[y_true == 0]
        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        if n_pos == 0 or n_neg == 0:
            return None
        # Count pairs where pos_score > neg_score
        greater = np.sum(pos_scores[:, None] > neg_scores[None, :])
        equal = np.sum(pos_scores[:, None] == neg_scores[None, :])
        auc = (greater + 0.5 * equal) / (n_pos * n_neg)
        return float(auc)
    except Exception:
        return None


def _compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Compute Precision-Recall AUC score safely."""
    if np.sum(y_true == 1) == 0:
        return None

    try:
        from sklearn.metrics import precision_recall_curve, auc
        p, r, _ = precision_recall_curve(y_true, y_prob)
        return float(auc(r, p))
    except ImportError:
        # Pure NumPy trapezoidal rule fallback for PR-AUC
        thresholds = np.sort(np.unique(y_prob))[::-1]
        precisions = [1.0]
        recalls = [0.0]
        for th in thresholds:
            preds = (y_prob >= th).astype(int)
            m = compute_binary_metrics(y_true, preds)
            precisions.append(m["precision"])
            recalls.append(m["recall"])
        precisions_arr = np.array(precisions)
        recalls_arr = np.array(recalls)
        # Sort by recall
        order = np.argsort(recalls_arr)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if trapz_fn is not None:
            return float(trapz_fn(precisions_arr[order], recalls_arr[order]))
        return None
    except Exception:
        return None


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    finding_names: list[str],
) -> dict[str, Any]:
    """Compute per-finding, macro-average, and micro-average multi-label metrics.

    Args:
        y_true: Ground truth binary matrix (N_samples, N_findings).
        y_pred: Predicted binary matrix (N_samples, N_findings).
        y_prob: Continuous similarity matrix (N_samples, N_findings).
        finding_names: List of finding class names.

    Returns:
        Structured metrics dictionary.
    """
    per_finding: dict[str, dict[str, Any]] = {}
    precisions, recalls, specificities, f1s = [], [], [], []
    roc_aucs, pr_aucs = [], []

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    for i, name in enumerate(finding_names):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        col_prob = y_prob[:, i] if y_prob is not None else None

        metrics = compute_binary_metrics(col_true, col_pred, col_prob)
        per_finding[name] = metrics

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_tn += metrics["tn"]
        total_fn += metrics["fn"]

        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        specificities.append(metrics["specificity"])
        f1s.append(metrics["f1"])

        if metrics["roc_auc"] is not None:
            roc_aucs.append(metrics["roc_auc"])
        if metrics["pr_auc"] is not None:
            pr_aucs.append(metrics["pr_auc"])

    # Macro Average (unweighted mean across findings)
    macro_precision = round(float(np.mean(precisions)), 4) if precisions else 0.0
    macro_recall = round(float(np.mean(recalls)), 4) if recalls else 0.0
    macro_specificity = round(float(np.mean(specificities)), 4) if specificities else 0.0
    macro_f1 = round(float(np.mean(f1s)), 4) if f1s else 0.0
    macro_roc_auc = round(float(np.mean(roc_aucs)), 4) if roc_aucs else None
    macro_pr_auc = round(float(np.mean(pr_aucs)), 4) if pr_aucs else None

    # Micro Average (pooled across all instances and classes)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    micro_f1 = (
        (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "per_finding": per_finding,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "specificity": macro_specificity,
            "f1": macro_f1,
            "roc_auc": macro_roc_auc,
            "pr_auc": macro_pr_auc,
        },
        "micro_avg": {
            "tp": total_tp,
            "fp": total_fp,
            "tn": total_tn,
            "fn": total_fn,
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "specificity": round(micro_specificity, 4),
            "f1": round(micro_f1, 4),
        },
    }
