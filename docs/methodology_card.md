# Methodology Passport Card 

## 1. Project information 

| Field | Value |
|---|---|
| Title | Monitoring LLM Reasoning Traces (Chain-of-Thought) for Detecting Unsafe Behavior in AI Safety Evaluations |
| Authors | Yerzhankyzy Markhaba, Kulmakhanbet Zarina |
| GitHub repository | https://github.com/markhabaerzhan/cot-monitor-benchmark |

## 2. Research type 

**Quantitative, experimental.** The study compares numerical detection metrics (recall at fixed FPR, precision, F1, and AUROC) for output-only and reasoning-trace + output monitors. Statistical hypotheses are tested at α = 0.05. Both conditions use the same test episodes, enabling paired comparisons. No human subjects or interviews are involved. False-positive and false-negative analysis (Task 5) is a descriptive supplement rather than a separate research method.

The current implementation uses a keyword-based placeholder monitor and sample data for pipeline testing. Full experimental conclusions require the planned evaluation with the finalized monitor and dataset.

## 3. Hypotheses 

Common settings: α = 0.05; primary metric is recall at 5% FPR. RQ3 is the main comparison; RQ1 and RQ2 are supporting research questions.

| Research question | H₀ (null) | H₁ (alternative) | Statistical test |
|---|---|---|---|
| **RQ3 (main)** | Recall at 5% FPR of the reasoning-trace + output monitor is equal to or lower than that of the output-only monitor. | Recall at 5% FPR of the reasoning-trace + output monitor is higher than that of the output-only monitor. | One-sided exact McNemar test + paired percentile bootstrap 95% CI |
| RQ2 | Recall at 5% FPR of the reasoning-trace + output monitor is ≤ 0.70. | Recall at 5% FPR of the reasoning-trace + output monitor is > 0.70. | One-sided exact binomial test + Wilson 95% CI |
| RQ1 | Recall at 5% FPR is the same across the four unsafe-behavior categories. | Recall differs between at least two categories. | Chi-square homogeneity test; Fisher–Freeman–Halton Monte Carlo test if expected counts are small |

For RQ3, the recall difference is defined as **recall(cot_plus_output) − recall(output_only)**. A positive value favors the reasoning-trace + output monitor.

## 4. Variable matrix 

| Type | Variable | Levels / unit | Used in |
|---|---|---|---|
| **Independent** | Monitor input | `output_only` vs `cot_plus_output` | RQ2, RQ3 |
| **Independent** | Unsafe-behavior category | reward hacking, deception, sabotage, harmful compliance | RQ1 |
| **Dependent** | Recall at 5% FPR (primary) | Proportion, 0–1 | RQ1, RQ2, RQ3 |
| **Dependent** | Precision, F1, AUROC (secondary) | 0–1 | Descriptive evaluation |
| **Controlled** | Monitor implementation and prompt template | Held constant across conditions where applicable | All |
| **Controlled** | Test episodes and labels | Same test episodes for both conditions | All |
| **Controlled** | Operating point | Threshold calibrated separately for each monitor using safe calibration episodes; empirical calibration FPR ≤ 5% | All |
| **Controlled** | Software and reproducibility settings | Python 3.13, pinned `requirements.txt`, Docker image `python:3.13-slim`, seed 42, CPU-only setup | All |

Equal category sizes, the final monitor model, and the episode-generation protocol must be specified and verified for the main experiment; they are not enforced by the current benchmark code.

## 5. Metrics and baseline

**Primary metric:** Recall at 5% FPR — the proportion of unsafe test episodes detected using a threshold calibrated on safe calibration episodes so that the empirical calibration-set FPR does not exceed 5%. The realized FPR on the separate safe test set is reported independently and may differ from 5%.

**Secondary metrics:** Precision, F1, and AUROC. AUROC is threshold-independent and measures ranking performance across possible thresholds.

**Guardrail metrics and validity checks:**

| Guardrail | Requirement or status |
|---|---|
| Calibration-set FPR | Must be ≤ 5% by the threshold-selection procedure |
| Test-set FPR | Reported separately; not guaranteed to be ≤ 5% |
| Monitor failures or unparseable responses | Target ≤ 1% of calls; must be measured and reported in the main experiment |
| Sample size | Aim for at least 100 safe episodes in both calibration and test splits; the current code issues warnings when either split is below this recommendation |
| Category balance | Equal numbers per unsafe category are planned but not enforced by the current code |
| Data safety | Do not publish harmful content unnecessarily or commit API keys |

**Baseline:** The output-only monitor (`output_only`) is compared with the reasoning-trace + output monitor (`cot_plus_output`). Both conditions use the same test episodes and the same monitor implementation settings where applicable. The current configuration uses a keyword-based placeholder (`backend: keyword`); a finalized LLM-based monitor must be configured for the intended LLM experiment. A random scorer can be added as an additional sanity baseline but is not one of the two conditions currently defined in `config.yaml`.

**Known limitation:** Chain-of-thought text may not faithfully represent a model's internal reasoning, and models may conceal or misrepresent their intentions. This limitation should be considered when interpreting results and discussed in the error analysis (Task 5). References: Turpin et al. (2023), Lanham et al. (2023), and Chen et al. (2025).and a model can hide intent in it (Turpin et al., 2023; Lanham et al., 2023; Chen et al., 2025). This is treated as a limitation and analyzed in the error analysis (Task 5).
