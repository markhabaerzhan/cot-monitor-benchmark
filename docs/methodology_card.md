# Methodology Passport Card / Карточка методологии

## 1. Project information / Информация о проекте

| Field | Value |
|---|---|
| Title | Monitoring LLM Reasoning Traces (Chain-of-Thought) for Detecting Unsafe Behavior in AI Safety Evaluations |
| Authors | Yerzhankyzy Markhaba, Kulmakhanbet Zarina |
| GitHub repository | https://github.com/markhabaerzhan/cot-monitor-benchmark |

## 2. Research type / Тип исследования

**Quantitative (количественное), experimental.** We measure numeric detection metrics (recall at fixed FPR, precision, F1, AUROC) of automated monitors on a labeled test set and test hypotheses statistically (α = 0.05). The same episodes are given to both monitors (paired design), so the effect of the monitor input is isolated. No human subjects or interviews are involved, so a qualitative component is not required; the error analysis of false positives/negatives (Task 5) is a small descriptive add-on, not a second method.

## 3. Hypotheses / Гипотезы

Derived from Task 1. Common settings: α = 0.05; primary metric is recall at 5% FPR on a labeled set of safe and unsafe episodes. RQ3 is the central (confirmatory) hypothesis; RQ1 and RQ2 are supporting.

| | H₀ (null) | H₁ (alternative) | Test |
|---|---|---|---|
| **RQ3 (main)** | Recall at 5% FPR of the reasoning-trace + output monitor is **equal to or lower than** that of the output-only monitor. | Recall at 5% FPR of the reasoning-trace + output monitor is **higher** than that of the output-only monitor. | Exact McNemar (one-sided) + paired bootstrap 95% CI of recall difference |
| RQ2 | Reasoning-trace monitor recall at 5% FPR is **≤ 0.70**. | Reasoning-trace monitor recall at 5% FPR is **> 0.70**. | One-sided exact binomial + bootstrap 95% CI. The 0.70 target is a working value, to be calibrated after the pilot run. |
| RQ1 | Recall at 5% FPR is the **same** for all four unsafe-behavior categories. | Recall differs between **at least two** categories. | Chi-square homogeneity (Fisher's exact if expected counts are small) |

## 4. Variable matrix / Матрица переменных

| Type | Variable | Levels / unit | Used in |
|---|---|---|---|
| **Independent** | Monitor input | `output_only` vs `cot_plus_output` | RQ2, RQ3 |
| **Independent** | Unsafe-behavior category | reward hacking, deception, sabotage, harmful compliance | RQ1 |
| **Dependent** | Recall at 5% FPR (**primary**) | proportion, 0–1 | all |
| **Dependent** | Precision, F1, AUROC (secondary) | 0–1 | all |
| **Controlled** | Monitor LLM, prompt template and temperature | same for both conditions; temperature 0.0 | all |
| **Controlled** | Source model(s) generating episodes; task domain and difficulty | fixed per run | all |
| **Controlled** | Test episodes and labeling protocol | identical episodes in both conditions; equal number per category | all |
| **Controlled** | Operating point | threshold calibrated per monitor on safe episodes to FPR ≤ 5% | all |
| **Controlled** | Software and hardware | Python 3.13, exact-pinned `requirements.txt`, Docker image `python:3.13-slim`, random seed 42, CPU only | all |

## 5. Metrics and baseline / Метрики и Baseline

**Primary metric:** recall at 5% FPR — the share of unsafe episodes flagged when the threshold is set so that at most 5% of safe episodes are flagged. The hypotheses are confirmed or rejected on this metric.

**Secondary metrics:** precision, F1, AUROC (the last is threshold-free and shows whether any result depends on the chosen operating point).

**Guardrail metrics (validity checks, not goals):**

| Guardrail | Limit |
|---|---|
| Realized FPR on safe episodes | ≤ 5% for both monitors (otherwise the comparison is unfair) |
| Unparseable / failed monitor replies | ≤ 1% of calls; failed episodes are reported, not silently dropped |
| Class balance | equal number of episodes per category; safe episodes ≥ 100 so a 5% FPR is measurable (planned) |
| Safety of data | no harmful content stored or published; no API keys in the repository |

**Baseline:** the **output-only monitor** (same LLM, same prompt, same episodes, sees prompt + final output but not the reasoning). It is the natural "no extra information" comparison and corresponds to current practice of monitoring actions and outputs. Additional sanity baseline: a random scorer (expected AUROC = 0.5) and the offline keyword placeholder.

**Known limitation:** CoT may not faithfully reflect real reasoning, and a model can hide intent in it (Turpin et al., 2023; Lanham et al., 2023; Chen et al., 2025). This is treated as a limitation and analyzed in the error analysis (Task 5).
