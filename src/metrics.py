
"""Metrics: fixed-FPR threshold, classification metrics and AUROC."""
import numpy as np


def threshold_at_fpr(safe_scores, target_fpr: float) -> float:
    """Choose the lowest threshold whose empirical safe-set FPR is <= target."""
    safe = np.asarray(safe_scores, dtype=float)

    if safe.size == 0:
        raise ValueError("Cannot calibrate threshold without safe scores")
    if not np.all(np.isfinite(safe)):
        raise ValueError("Safe scores must all be finite")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")

    candidates = np.unique(np.append(safe, safe.max() + 1.0))
    candidates.sort()

    for threshold in candidates:
        fpr = float(np.mean(safe >= threshold))
        if fpr <= target_fpr:
            return float(threshold)

    return float(np.nextafter(safe.max(), np.inf))


def confusion(scores, labels, threshold):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must all be finite")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must contain only 0 and 1")

    predictions = scores >= threshold

    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    tn = int(np.sum(~predictions & (labels == 0)))

    return tp, fp, fn, tn


def prf(tp, fp, fn):
    """Return recall, precision and F1; undefined metrics are None."""
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None

    if (
        recall is not None
        and precision is not None
        and recall + precision > 0
    ):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return recall, precision, f1


def auroc(scores, labels):
    """Mann-Whitney AUROC; ties count as 0.5."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must all be finite")

    positive = scores[labels == 1]
    negative = scores[labels == 0]

    if len(positive) == 0 or len(negative) == 0:
        return None

    greater = np.sum(positive[:, None] > negative[None, :])
    equal = np.sum(positive[:, None] == negative[None, :])

    return float(
        (greater + 0.5 * equal) / (len(positive) * len(negative))
    )
