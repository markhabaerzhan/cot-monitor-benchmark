# Monitoring LLM Reasoning Traces for Unsafe Behavior

**Does reading the chain-of-thought (CoT) help an automated monitor detect unsafe LLM behavior compared with reading outputs alone?**

**Authors:** Yerzhankyzy Markhaba, Kulmakhanbet Zarina
**Course:** Research Methods, Task 2 (Methodology Card & Research Repository)
**Status:** measurement tool ready; full data collection planned for Week 7.

## Overview

We compare two automated monitors on identical episodes:

| Condition | Monitor reads |
|---|---|
| `output_only` | prompt + final output |
| `cot_plus_output` | prompt + reasoning trace + final output |

Unsafe behavior is grouped into four categories: **reward hacking, deception, sabotage, harmful compliance**. For each condition the decision threshold is calibrated on safe episodes to a fixed **5% false positive rate (FPR)**. The primary metric is **recall at 5% FPR**; secondary metrics are precision, F1 and AUROC.

| RQ | Question | Test |
|---|---|---|
| RQ1 | Does recall differ between behavior categories? | Chi-square homogeneity (Fisher if counts are small) |
| RQ2 | Is CoT-monitor recall at 5% FPR > 0.70? | One-sided exact binomial + 95% CI |
| RQ3 | Does the CoT monitor beat the output-only monitor? | Exact McNemar + paired bootstrap CI |

> **Important:** `data/sample/` contains 10 hand-written toy rows and the default `keyword` backend is an offline placeholder. Numbers produced by the sample run only prove that the pipeline works; they are **not** research results.

## Repository structure

```
.
├── configs/
│   ├── config.yaml            # all experiment parameters
│   └── monitor_prompt.txt     # monitor prompt template (same for both conditions)
├── data/
│   └── sample/episodes.jsonl  # 10-row smoke-test sample (full data is git-ignored)
├── src/
│   ├── main.py                # entry point
│   ├── benchmark.py           # scoring, calibration, per-category metrics, tests
│   ├── monitors.py            # keyword placeholder + OpenAI-compatible LLM backend
│   ├── metrics.py             # recall@FPR, precision, F1, AUROC
│   └── stats.py               # chi-square, binomial, McNemar, bootstrap
├── tests/test_pipeline.py     # unit + smoke tests
├── notebooks/                 # analysis notebooks (Week 7+)
├── docs/                      # Methodology Card
├── results/                   # run outputs (git-ignored)
├── Dockerfile
├── requirements.txt           # exact pins
└── LICENSE
```

## Planned benchmarks

| ID | Condition | Behavior categories | Episodes (planned) | Metrics |
|---|---|---|---|---|
| B1 | output_only | all 4 + safe | balanced per category, to be fixed after pilot | recall@5% FPR, precision, F1, AUROC |
| B2 | cot_plus_output | all 4 + safe | same episodes as B1 | recall@5% FPR, precision, F1, AUROC |
| B3 | B1 vs B2 (paired) | all 4 | same episodes | McNemar, paired bootstrap CI |

## System requirements

- Python 3.13 (or Docker)
- No GPU needed. The default sample run is fully offline.
- For the real `llm` backend: an API key for an OpenAI-compatible endpoint.

## Quickstart

Docker:

```bash
docker build -t cot-monitor .
docker run --rm -v "$PWD/results:/app/results" cot-monitor
```

Local:

```bash
pip install -r requirements.txt
python -m src.main --config configs/config.yaml
```

Run the tests: `python -m unittest discover -s tests -t .`

Output: `results/sample_run.json` (metrics and test results) and `results/sample_run.log`.

To use a real LLM monitor, set `monitor.backend: llm` and `monitor.llm.model` in `configs/config.yaml`, then `export MONITOR_API_KEY=...` (the key is never stored in the repo).

## Data format

One JSON object per line: `id`, `category` (`safe` or one of the four categories), `label` (`safe`/`unsafe`), `prompt`, `cot`, `output`. Harmful-compliance episodes are stored only as redacted placeholders; no harmful content is published.

## Citation / License

Released under the MIT License (see `LICENSE`). If you use this tool:

```
Yerzhankyzy, M., Kulmakhanbet, Z. (2026). Monitoring LLM Reasoning Traces for Unsafe
Behavior: a CoT vs output-only monitor benchmark. GitHub repository.
```
