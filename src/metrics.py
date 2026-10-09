"""Metrics: threshold at fixed FPR, recall/precision/F1, AUROC (pure numpy)."""
import numpy as np


def threshold_at_fpr(safe_scores, target_fpr: float) -> float:
    """Smallest threshold t such that FPR (share of safe scores >= t) <= target_fpr.
    Flag an episode as unsafe when score >= t."""
    safe = np.sort(np.asarray(safe_scores, dtype=float))
    for t in np.unique(np.append(safe, safe.max() + 1.0)):
        if np.mean(safe >= t) <= target_fpr:
            return float(t)
    return float(safe.max() + 1.0)


def confusion(scores, labels, threshold):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def prf(tp, fp, fn):
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and precision + recall else float("nan"))
    return recall, precision, f1


def auroc(scores, labels) -> float:
    """Mann-Whitney formulation, ties count 0.5."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))
