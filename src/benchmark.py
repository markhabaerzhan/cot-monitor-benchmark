"""Core benchmark: score episodes under each condition, calibrate the threshold on
safe episodes (target FPR), compute metrics per category, run RQ1-RQ3 tests."""
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
    return rows


def run_condition(episodes, cond, cfg, prompt_template):
    scores = np.array([
        monitors.score_episode(e, cond["visible_fields"], cfg["monitor"], prompt_template)
        for e in episodes])
    labels = np.array([1 if e["label"] == "unsafe" else 0 for e in episodes])
    cats = np.array([e["category"] for e in episodes])
    thr = metrics.threshold_at_fpr(scores[labels == 0], cfg["experiment"]["target_fpr"])
    flagged = scores >= thr

    tp, fp, fn, tn = metrics.confusion(scores, labels, thr)
    recall, precision, f1 = metrics.prf(tp, fp, fn)
    result = {
        "threshold": thr,
        "overall": {"recall_at_fpr": recall, "precision": precision, "f1": f1,
                    "auroc": metrics.auroc(scores, labels),
                    "fpr": fp / (fp + tn) if fp + tn else float("nan"),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "per_category": {},
    }
    safe_mask = labels == 0
    for c in cfg["data"]["categories"]:
        m = (cats == c) | safe_mask
        if not (cats == c).any():
            continue
        ctp, cfp, cfn, _ = metrics.confusion(scores[m], labels[m], thr)
        r, p, f = metrics.prf(ctp, cfp, cfn)
        result["per_category"][c] = {
            "n_unsafe": int((cats == c).sum()), "detected": ctp,
            "recall_at_fpr": r, "precision": p, "f1": f,
            "auroc": metrics.auroc(scores[m], labels[m])}
    return result, flagged, labels, cats


def run(cfg: dict) -> dict:
    episodes = load_episodes(cfg["data"]["episodes_path"])
    template = Path(cfg["monitor"]["prompt_template"]).read_text(encoding="utf-8")
    out, detections = {"config": cfg, "n_episodes": len(episodes), "conditions": {}}, {}
    for cond in cfg["conditions"]:
        log.info("Running condition: %s", cond["name"])
        res, flagged, labels, cats = run_condition(episodes, cond, cfg, template)
        out["conditions"][cond["name"]] = res
        detections[cond["name"]] = flagged
        log.info("  threshold=%.1f recall@FPR=%.3f auroc=%.3f", res["threshold"],
                 res["overall"]["recall_at_fpr"], res["overall"]["auroc"])

    unsafe = labels == 1
    st, sc = cfg["statistics"], cfg["experiment"]["seed"]
    cot = detections["cot_plus_output"]
    base = detections["output_only"]
    totals = {c: int((cats == c).sum()) for c in cfg["data"]["categories"] if (cats == c).any()}
    det_by_cat = {c: int((cot & (cats == c)).sum()) for c in totals}
    out["tests"] = {
        "RQ1_category_homogeneity": stats.rq1_homogeneity(det_by_cat, totals),
        "RQ2_vs_target": stats.rq2_binomial(int(cot[unsafe].sum()), int(unsafe.sum()),
                                            st["rq2_target_recall"]),
        "RQ3_mcnemar": stats.rq3_mcnemar(base[unsafe], cot[unsafe]),
        "RQ3_paired_bootstrap": stats.paired_bootstrap_diff(
            base[unsafe], cot[unsafe], st["bootstrap_resamples"], sc),
    }
    return out


def save(result: dict, out_dir: str, run_name: str) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_name}.json"
    path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return path
