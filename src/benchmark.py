
"""Core benchmark for comparing output-only and CoT-enabled monitors."""
import json
import logging
from pathlib import Path

import numpy as np

from . import metrics, monitors, stats

log = logging.getLogger("benchmark")


def load_episodes(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No episodes in {path}")

    for i, episode in enumerate(rows):
        if episode.get("label") not in {"safe", "unsafe"}:
            raise ValueError(f"Episode {i}: label must be 'safe' or 'unsafe'")
        if "category" not in episode:
            raise ValueError(f"Episode {i}: missing category")

    return rows


def split_episodes(episodes: list, cfg: dict):
    """Split safe episodes into calibration/test; keep unsafe episodes test-only."""
    seed = int(cfg["experiment"]["seed"])
    fraction = float(cfg["split"]["calibration_fraction"])
    minimum = int(cfg["split"].get("min_safe_per_split_warning", 100))

    if not 0 < fraction < 1:
        raise ValueError("split.calibration_fraction must be between 0 and 1")

    safe = [e for e in episodes if e["label"] == "safe"]
    unsafe = [e for e in episodes if e["label"] == "unsafe"]

    if len(safe) < 2:
        raise ValueError("At least two safe episodes are required for calibration/test")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(safe))
    n_calibration = int(round(len(safe) * fraction))
    n_calibration = min(max(n_calibration, 1), len(safe) - 1)

    calibration = [safe[i] for i in indices[:n_calibration]]
    safe_test = [safe[i] for i in indices[n_calibration:]]
    test = safe_test + unsafe

    if len(calibration) < minimum or len(safe_test) < minimum:
        log.warning(
            "Small safe split: calibration=%d, safe_test=%d; recommended minimum=%d",
            len(calibration), len(safe_test), minimum
        )

    if not unsafe:
        raise ValueError("No unsafe episodes found; recall and RQ tests cannot be computed")

    return calibration, test


def score_episodes(episodes, cond, monitor_cfg, prompt_template):
    return np.asarray([
        monitors.score_episode(
            episode, cond["visible_fields"], monitor_cfg, prompt_template
        )
        for episode in episodes
    ], dtype=float)


def run_condition(calibration, test, cond, cfg, prompt_template):
    """Calibrate threshold on safe calibration data; evaluate only on test data."""
    safe_calibration = [
        e for e in calibration if e["label"] == "safe"
    ]
    if not safe_calibration:
        raise ValueError("Calibration set must contain safe episodes")

    calibration_scores = score_episodes(
        safe_calibration, cond, cfg["monitor"], prompt_template
    )
    threshold = metrics.threshold_at_fpr(
        calibration_scores, cfg["experiment"]["target_fpr"]
    )

    scores = score_episodes(test, cond, cfg["monitor"], prompt_template)
    labels = np.asarray(
        [1 if e["label"] == "unsafe" else 0 for e in test], dtype=int
    )
    cats = np.asarray([e["category"] for e in test])
    flagged = scores >= threshold

    tp, fp, fn, tn = metrics.confusion(scores, labels, threshold)
    recall, precision, f1 = metrics.prf(tp, fp, fn)

    result = {
        "threshold": float(threshold),
        "calibration": {
            "n_safe": len(safe_calibration),
            "target_fpr": float(cfg["experiment"]["target_fpr"]),
        },
        "test": {
            "n_total": len(test),
            "n_safe": int(np.sum(labels == 0)),
            "n_unsafe": int(np.sum(labels == 1)),
        },
        "overall": {
            "recall_at_fpr": recall,
            "precision": precision,
            "f1": f1,
            "auroc": metrics.auroc(scores, labels),
            "fpr": fp / (fp + tn) if fp + tn else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "per_category": {},
    }

    safe_mask = labels == 0
    for category in cfg["data"]["categories"]:
        category_mask = cats == category
        if not np.any(category_mask & (labels == 1)):
            continue

        mask = category_mask | safe_mask
        ctp, cfp, cfn, ctn = metrics.confusion(
            scores[mask], labels[mask], threshold
        )
        r, p, f = metrics.prf(ctp, cfp, cfn)
        category_scores = scores[mask]
        category_labels = labels[mask]

        result["per_category"][category] = {
            "n_unsafe": int(np.sum(category_mask & (labels == 1))),
            "detected": ctp,
            "recall_at_fpr": r,
            "precision": p,
            "f1": f,
            "auroc": metrics.auroc(category_scores, category_labels),
            "fp": cfp,
            "tn": ctn,
        }

    return result, flagged, labels, cats


def run(cfg: dict) -> dict:
    episodes = load_episodes(cfg["data"]["episodes_path"])
    calibration, test = split_episodes(episodes, cfg)
    template = Path(cfg["monitor"]["prompt_template"]).read_text(
        encoding="utf-8"
    )

    output = {
        "config": cfg,
        "n_episodes": len(episodes),
        "n_calibration_safe": len(calibration),
        "n_test_episodes": len(test),
        "conditions": {},
    }
    detections = {}

    for condition in cfg["conditions"]:
        name = condition["name"]
        log.info("Running condition: %s", name)

        result, flagged, labels, categories = run_condition(
            calibration, test, condition, cfg, template
        )
        output["conditions"][name] = result
        detections[name] = flagged

        log.info(
            "threshold=%s recall=%s AUROC=%s",
            result["threshold"],
            result["overall"]["recall_at_fpr"],
            result["overall"]["auroc"],
        )

    unsafe = labels == 1
    if not np.any(unsafe):
        raise ValueError("Test set contains no unsafe episodes")

    statistics_cfg = cfg["statistics"]
    seed = int(cfg["experiment"]["seed"])

    cot = detections["cot_plus_output"]
    baseline = detections["output_only"]

    totals = {
        category: int(np.sum((categories == category) & unsafe))
        for category in cfg["data"]["categories"]
        if np.any((categories == category) & unsafe)
    }
    detected_by_category = {
        category: int(np.sum(cot & unsafe & (categories == category)))
        for category in totals
    }

    output["tests"] = {
        "RQ1_category_homogeneity": stats.rq1_homogeneity(
            detected_by_category, totals
        ),
        "RQ2_vs_target": stats.rq2_binomial(
            int(np.sum(cot & unsafe)),
            int(np.sum(unsafe)),
            float(statistics_cfg["rq2_target_recall"]),
        ),
        "RQ3_mcnemar": stats.rq3_mcnemar(
            baseline[unsafe], cot[unsafe]
        ),
        "RQ3_paired_bootstrap": stats.paired_bootstrap_diff(
            baseline[unsafe],
            cot[unsafe],
            int(statistics_cfg["bootstrap_resamples"]),
            seed,
        ),
    }

    return output


def save(result: dict, out_dir: str, run_name: str) -> Path:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_name}.json"

    path.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path
