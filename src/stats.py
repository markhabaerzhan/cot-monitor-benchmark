
"""Statistical tests used for RQ1-RQ3 (see Methodology Card)."""
import numpy as np
from scipy import stats



def rq1_homogeneity(detected_by_cat: dict, total_by_cat: dict) -> dict:
    """Test whether detection rates differ across categories."""
    cats = list(total_by_cat)

    if len(cats) < 2:
        raise ValueError("RQ1 requires at least two categories.")

    table = np.array([
        [
            detected_by_cat[c],
            total_by_cat[c] - detected_by_cat[c],
        ]
        for c in cats
    ], dtype=int)

    if np.any(table < 0):
        raise ValueError("Detected counts cannot exceed total counts.")

    if np.any(table.sum(axis=0) == 0):
        return {
            "test": "not_testable",
            "p_value": 1.0,
            "note": "No detections or no missed cases across categories.",
        }

    chi2, p_chi2, _, expected = stats.chi2_contingency(
        table, correction=False
    )

    if np.any(expected < 5):
        result = stats.fisher_exact(table, method=stats.MonteCarloMethod(
            n_resamples=100_000,
            rng=np.random.default_rng(42),
        ))
        return {
            "test": "fisher_freeman_halton_monte_carlo",
            "p_value": float(result.pvalue),
            "small_expected_counts": True,
            "n_resamples": 100_000,
            "seed": 42,
        }

    return {
        "test": "chi2",
        "chi2": float(chi2),
        "p_value": float(p_chi2),
        "small_expected_counts": False,
    }




def rq2_binomial(
    k: int, n: int, target: float, seed: int = 42,
    n_resamples: int = 2000,
) -> dict:
    """One-sided exact binomial test with a 95% Wilson CI."""
    if n <= 0 or k < 0 or k > n:
        raise ValueError("Require n > 0 and 0 <= k <= n.")
    if not 0 <= target <= 1:
        raise ValueError("target must be between 0 and 1.")

    res = stats.binomtest(k, n, target, alternative="greater")

    ci = res.proportion_ci(
        confidence_level=0.95,
        method="wilson",
    )

    return {
        "test": "binomial_greater",
        "k": k,
        "n": n,
        "recall": k / n,
        "p_value": float(res.pvalue),
        "ci95": [float(ci.low), float(ci.high)],
        "ci_method": "wilson",
    }



def holm_correction(p_values: dict) -> dict:
    """Holm correction for a family of p-values."""
    names = list(p_values)
    if any(not 0 <= p_values[name] <= 1 for name in names):
        raise ValueError("All p-values must be between 0 and 1.")

    m = len(names)
    ordered = sorted(names, key=lambda name: p_values[name])
    adjusted = {}
    running_max = 0.0

    for rank, name in enumerate(ordered):
        value = min(1.0, (m - rank) * p_values[name])
        running_max = max(running_max, value)
        adjusted[name] = running_max

    return adjusted


def rq3_mcnemar(detected_a, detected_b) -> dict:
    """Exact one-sided McNemar test on paired detections."""
    a = np.asarray(detected_a, dtype=bool)
    b = np.asarray(detected_b, dtype=bool)

    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Detection arrays must be 1-D and have equal lengths.")

    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n = b01 + b10

    p = (
        float(stats.binomtest(
            b01, n, 0.5, alternative="greater"
        ).pvalue)
        if n else 1.0
    )

    return {
        "test": "mcnemar_exact_one_sided",
        "b01_only_cot": b01,
        "b10_only_output": b10,
        "p_value": p,
    }


def paired_bootstrap_diff(
    detected_a, detected_b, n_resamples: int, seed: int
) -> dict:
    """Paired bootstrap CI for the difference in detection rates."""
    a = np.asarray(detected_a, dtype=float)
    b = np.asarray(detected_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Inputs must be 1-D arrays of equal length.")
    if len(a) == 0 or n_resamples <= 0:
        raise ValueError("Inputs and n_resamples must be non-empty/positive.")

    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_resamples)

    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs[i] = b[idx].mean() - a[idx].mean()

    lo, hi = np.percentile(diffs, [2.5, 97.5])

    return {
        "recall_diff": float(b.mean() - a.mean()),
        "ci95": [float(lo), float(hi)],
        "ci_method": "paired_percentile_bootstrap",
    }
