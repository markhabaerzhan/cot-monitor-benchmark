"""Statistical tests used for RQ1-RQ3 (see Methodology Card)."""
import numpy as np
from scipy import stats


def rq1_homogeneity(detected_by_cat: dict, total_by_cat: dict) -> dict:
    """Chi-square test of homogeneity on detected/missed counts per category
    (falls back to a note when expected counts are small; use Fisher in Week 7)."""
    cats = list(total_by_cat)
    table = np.array([[detected_by_cat[c], total_by_cat[c] - detected_by_cat[c]] for c in cats])
    if (table.sum(axis=0) == 0).any():
        return {"test": "chi2", "p_value": 1.0, "note": "degenerate table"}
    chi2, p, _, expected = stats.chi2_contingency(table)
    return {"test": "chi2", "chi2": float(chi2), "p_value": float(p),
            "small_expected_counts": bool((expected < 5).any())}


def rq2_binomial(k: int, n: int, target: float) -> dict:
    """One-sided exact binomial test, H1: recall > target."""
    res = stats.binomtest(k, n, target, alternative="greater")
    ci = res.proportion_ci(confidence_level=0.95, method="exact")
    return {"test": "binomial_greater", "k": k, "n": n, "recall": k / n,
            "p_value": float(res.pvalue), "ci95": [float(ci.low), float(ci.high)]}


def rq3_mcnemar(detected_a, detected_b) -> dict:
    """Exact McNemar on paired detections. a = output-only, b = CoT+output.
    b01 = missed by a but caught by b; b10 = the reverse."""
    a, b = np.asarray(detected_a, bool), np.asarray(detected_b, bool)
    b01, b10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    n = b01 + b10
    p = float(stats.binomtest(b01, n, 0.5, alternative="greater").pvalue) if n else 1.0
    return {"test": "mcnemar_exact_one_sided", "b01_only_cot": b01,
            "b10_only_output": b10, "p_value": p}


def paired_bootstrap_diff(detected_a, detected_b, n_resamples: int, seed: int) -> dict:
    a, b = np.asarray(detected_a, float), np.asarray(detected_b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs[i] = b[idx].mean() - a[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"recall_diff": float(b.mean() - a.mean()), "ci95": [float(lo), float(hi)]}
