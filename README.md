# Monitoring LLM Reasoning Traces for Unsafe Behavior

**Does reading the chain-of-thought (CoT) help an automated monitor detect unsafe LLM behavior compared with reading outputs alone?**

<**Authors:** Yerzhankyzy Markhaba, Kulmakhanbet Zarina

**Course:** Research Methods, Task 2 (Methodology Card & Research Repository)

**Status:** Measurement tool ready; full data collection planned for Week 7.


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

```text
.
├── configs/
│   ├── config.yaml
│   └── monitor_prompt.txt
├── data/
│   └── sample/
│       └── episodes.jsonl
├── docs/
│   ├── methodology_card_v3.pdf
│   ├── methodology_card.md
│   └── README.md
├── notebooks/
├── results/
│   ├── .gitkeep
│   ├── sample_run.json
│   └── sample_run.log
├── src/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── main.py
│   ├── metrics.py
│   ├── monitors.py
│   └── stats.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
└── requirements.txt
```

## Planned benchmarks

| ID | Condition | Behavior categories | Episodes (planned) | Metrics |
|---|---|---|---|---|
| B1 | `output_only` | All four unsafe categories + safe | To be determined after the pilot | Recall@5% FPR, precision, F1, AUROC |
| B2 | `cot_plus_output` | All four unsafe categories + safe | Same episodes as B1 | Recall@5% FPR, precision, F1, AUROC |
| B3 | B1 vs. B2 (paired) | All four unsafe categories | Same episodes | Exact McNemar test, paired bootstrap 95% CI |

The full evaluation dataset and final sample size will be determined after the pilot. The current sample dataset is intended only for pipeline testing.

## System requirements

- Python 3.13 or Docker.
- No GPU is required for the default offline sample run.
- The real `llm` backend requires an API key for an OpenAI-compatible endpoint.

## Quickstart

### Docker (Bash)

```bash
docker build -t cot-monitor .
docker run --rm -v "$PWD/results:/app/results" cot-monitor
```

### Local execution

```bash
pip install -r requirements.txt
python -m src.main --config configs/config.yaml
```

### Run tests

```bash
python -m unittest discover -s tests -t .
```

The sample run writes `results/sample_run.json` and `results/sample_run.log`. These outputs demonstrate pipeline execution and must not be interpreted as research findings.

To use a real LLM monitor, configure `monitor.backend: llm` and `monitor.llm.model` in `configs/config.yaml`. Set `MONITOR_API_KEY` as an environment variable. Never commit API keys to the repository.

## Data format

Each line in the JSONL dataset contains one JSON object with the following fields:

- `id` — unique episode identifier.
- `category` — `safe` or one of the four unsafe behavior categories.
- `label` — `safe` or `unsafe`.
- `prompt` — input prompt.
- `cot` — reasoning trace available to the experiment.
- `output` — final model output.

Harmful-compliance episodes use redacted placeholders; harmful content is not published.

## Citation / License

This project is released under the MIT License. See `LICENSE`.

### BibTeX

```bibtex
@misc{yerzhankyzy2026cotmonitor,
  author = {Yerzhankyzy, Markhaba and Kulmakhanbet, Zarina},
  title = {Monitoring LLM Reasoning Traces for Unsafe Behavior: A CoT vs Output-Only Monitor Benchmark},
  year = {2026},
  note = {Research Methods, Task 2: Methodology Card and Research Repository},
  howpublished = {GitHub repository}
}
```
