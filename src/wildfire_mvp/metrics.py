from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_curve_manual(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    tp = 0
    fp = 0
    positives = max(int(np.sum(y_true == 1)), 1)
    precision = [1.0]
    recall = [0.0]

    for label in y_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision.append(tp / max(tp + fp, 1))
        recall.append(tp / positives)

    return np.asarray(precision, dtype=float), np.asarray(recall, dtype=float)


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall = precision_recall_curve_manual(y_true, y_prob)
    return float(np.trapezoid(precision, recall))


def f1_score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    counts = confusion_counts(y_true, y_prob, threshold)
    precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
    recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    counts = confusion_counts(y_true, y_prob, threshold)
    return counts["tp"] / max(counts["tp"] + counts["fn"], 1)


def precision_score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    counts = confusion_counts(y_true, y_prob, threshold)
    return counts["tp"] / max(counts["tp"] + counts["fp"], 1)


def specificity_score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    counts = confusion_counts(y_true, y_prob, threshold)
    return counts["tn"] / max(counts["tn"] + counts["fp"], 1)


@dataclass(slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 15) -> list[ReliabilityBin]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[ReliabilityBin] = []
    for idx in range(num_bins):
        lo, hi = edges[idx], edges[idx + 1]
        mask = (y_prob >= lo) & (y_prob < hi if idx < num_bins - 1 else y_prob <= hi)
        if np.any(mask):
            confidence = float(np.mean(y_prob[mask]))
            accuracy = float(np.mean(y_true[mask]))
            count = int(np.sum(mask))
        else:
            confidence = 0.0
            accuracy = 0.0
            count = 0
        bins.append(ReliabilityBin(lo, hi, count, confidence, accuracy))
    return bins


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 15) -> float:
    bins = reliability_bins(y_true, y_prob, num_bins)
    total = max(sum(item.count for item in bins), 1)
    ece = 0.0
    for item in bins:
        ece += (item.count / total) * abs(item.accuracy - item.confidence)
    return float(ece)


def spearman_rank_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Arrays must be the same length")
    if len(a) < 2:
        return 0.0

    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks

    ra = rank(a)
    rb = rank(b)
    ra_mean = float(np.mean(ra))
    rb_mean = float(np.mean(rb))
    numerator = float(np.sum((ra - ra_mean) * (rb - rb_mean)))
    denominator = math.sqrt(float(np.sum((ra - ra_mean) ** 2) * np.sum((rb - rb_mean) ** 2)))
    return 0.0 if denominator == 0 else numerator / denominator
