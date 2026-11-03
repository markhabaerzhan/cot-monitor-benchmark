# Domain Model: Monitoring LLM Reasoning Traces for Unsafe Behavior

## 1. Problem Formulation

The research investigates whether providing a monitoring system with additional Chain-of-Thought (CoT) information improves the detection of unsafe behavior in AI-generated episodes.

The task is formulated as a binary classification problem.

Let the dataset be:

\[
D = \{(x_i,y_i)\}_{i=1}^{N}
\]

where:

- \(x_i\) is the input information available to the monitor for episode \(i\);
- \(y_i \in \{0,1\}\) is the ground-truth label;
- \(y_i=0\) represents safe behavior;
- \(y_i=1\) represents unsafe behavior;
- \(N\) is the number of episodes.

Two monitoring conditions are evaluated:

- **Output-only:** the monitor receives the prompt and final output.
- **CoT + Output:** the monitor receives the prompt, reasoning trace, and final output.

The objective is to compare detection quality and computational performance between these conditions.

## 2. Mathematical Model

For a learned binary classifier, let:

\[
p_i = P(y_i=1 \mid x_i)
\]

represent the predicted probability of unsafe behavior.

The binary cross-entropy loss is:

\[
\mathcal{L}_{BCE} =
-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log(p_i)
+(1-y_i)\log(1-p_i)
\right]
\]

This loss penalizes incorrect probability predictions and is suitable for binary classification.

**Implementation note:** The current project uses a keyword-based monitoring backend that returns an integer suspicion score from 0 to 100. It does not train a probabilistic classifier or optimize BCE. The equation represents a formal ML formulation for the binary monitoring problem, not a claim about the current implementation.

## 3. Classification Decision and Constraints

For the current score-based monitor, let \(s_i\) be the suspicion score and \(\tau\) the decision threshold.

The predicted class is:

\[
\hat{y}_i =
\begin{cases}
1, & s_i \geq \tau \\
0, & s_i < \tau
\end{cases}
\]

The threshold is calibrated using safe calibration episodes.

The target false positive rate is:

\[
FPR = \frac{FP}{FP+TN} \leq 0.05
\]

The project also defines a working target recall:

\[
Recall = \frac{TP}{TP+FN} \geq 0.70
\]

These are target constraints. Their actual achievement must be verified using evaluation results.

### Threshold Calibration and Constrained Optimization

The monitoring system uses a decision threshold to classify episodes as safe or unsafe.

The threshold is calibrated using a separate set of safe calibration episodes. The target false positive rate is:

\[
FPR_{\text{cal}}(\tau)\leq \alpha
\]

where:

\[
\alpha=0.05
\]

The threshold is selected based on calibration data and then evaluated on a separate test set. This separation prevents the test set from being used to tune the decision threshold.

The project also defines a working recall target:

\[
Recall_{\text{target}}=0.70
\]

This value is a target for evaluation, not a result already achieved by the current monitor.

A conceptual constrained optimization objective for a future monitoring system can be expressed as:

\[
\min_{\tau}\; FNR_{\text{test}}(\tau)
\]

subject to:

\[
FPR_{\text{cal}}(\tau)\leq 0.05
\]

and:

\[
Recall_{\text{test}}(\tau)\geq 0.70
\]

where:

\[
FNR=\frac{FN}{TP+FN}
\]

The objective is to reduce missed unsafe episodes while controlling false alarms.

Computational performance is evaluated separately through latency and throughput measurements. The current experiment measures these quantities but does not implement a joint optimization of detection quality and latency.

### Practical Interpretation

- The calibration set is used to determine the decision threshold.
- The test set is used to evaluate detection performance.
- The target FPR is 5%.
- The working target recall is 70%.
- Latency and throughput are reported as performance measurements.

The constraints describe the intended evaluation criteria. They must not be interpreted as achieved results unless confirmed by test-set measurements.

## 4. Target Evaluation Metrics

The following metrics are used to evaluate detection performance:

**Precision**

\[
Precision = \frac{TP}{TP+FP}
\]

**Recall**

\[
Recall = \frac{TP}{TP+FN}
\]

**F1-score**

\[
F1 = 2 \cdot
\frac{Precision \cdot Recall}
{Precision + Recall}
\]

**False Positive Rate**

\[
FPR = \frac{FP}{FP+TN}
\]

ROC-AUC and PR-AUC may additionally be used to evaluate ranking performance across decision thresholds, provided that suitable continuous scores are available.

## 5. Class Imbalance and Loss Functions

Unsafe episodes may be less frequent than safe episodes. This creates a class imbalance problem.

Three common loss functions have different purposes:

**Mean Squared Error (MSE)**

\[
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
\]

MSE penalizes squared prediction errors and is sensitive to large deviations and outliers.

**Mean Absolute Error (MAE)**

\[
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|
\]

MAE is less sensitive to extreme errors than MSE, but it is not the standard choice for probabilistic binary classification.

**Binary Cross-Entropy**

\[
BCE = -\frac{1}{N}\sum_{i=1}^{N}
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
\]

Binary cross-entropy is appropriate for probabilistic binary classification.

For strongly imbalanced datasets, weighted cross-entropy or focal loss can give more attention to difficult or minority-class examples.

### Focal Loss for Class Imbalance

Focal Loss is designed to reduce the influence of easily classified examples and focus training on difficult examples. This can be useful when unsafe episodes represent a minority class.

The binary Focal Loss is defined as:

\[
FL(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t)
\]

where:

- \(p_t\) is the predicted probability assigned to the correct class;
- \(\alpha_t\) is a weighting factor for class imbalance;
- \(\gamma \geq 0\) is the focusing parameter.

When \(\gamma=0\), Focal Loss becomes a weighted cross-entropy form. Increasing \(\gamma\) reduces the contribution of well-classified examples.

### Comparison of Classification Metrics

| Metric | Main purpose | Importance |
|---|---|---|
| ROC-AUC | Measures ranking quality across classification thresholds | Evaluates overall class separation |
| PR-AUC | Measures the precision-recall trade-off | Particularly useful for imbalanced datasets |
| F1-score | Harmonic mean of precision and recall | Balances false positives and false negatives |
| Recall | Measures the proportion of unsafe episodes detected | Important for identifying unsafe behavior |
| FPR | Measures the proportion of safe episodes incorrectly flagged | Controls false alarms |

For an imbalanced safety-monitoring dataset, PR-AUC and Recall are particularly informative because they focus on the minority unsafe class and its detection.

**Implementation limitation:** Focal Loss is discussed as a possible approach for a future learned classifier. It is not currently implemented by the keyword-based monitoring backend.

## 6. Performance Metrics

The computational performance of the monitor is evaluated separately from classification quality.

Mean latency:

\[
\overline{L} = \frac{1}{N}\sum_{i=1}^{N}L_i
\]

Latency percentiles:

\[
P_{50}=Q_{0.50}(L), \quad
P_{95}=Q_{0.95}(L), \quad
P_{99}=Q_{0.99}(L)
\]

Throughput:

\[
R_{\text{achieved}} = \frac{N}{T}
\]

where \(N\) is the number of completed requests and \(T\) is the measurement duration in seconds.

The experiment evaluates latency at target request rates of 1, 5, 10, and 20 requests per second.

## 7. Mathematical Notation Table

| Symbol | Definition |
|---|---|
| \(D\) | Experimental dataset |
| \(N\) | Number of evaluated episodes or measurements |
| \(x_i\) | Input features for episode \(i\) |
| \(y_i\) | Ground-truth binary label |
| \(\hat y_i\) | Predicted binary label |
| \(p_i\) | Predicted probability of unsafe behavior |
| \(s_i\) | Monitor suspicion score |
| \(\tau\) | Classification threshold |
| \(TP\) | True positives |
| \(TN\) | True negatives |
| \(FP\) | False positives |
| \(FN\) | False negatives |
| \(L_i\) | Latency of measurement \(i\), in milliseconds |
| \(P_{50}\) | Median latency |
| \(P_{95}\) | 95th percentile latency |
| \(P_{99}\) | 99th percentile latency |
| \(R_{\text{achieved}}\) | Achieved throughput, requests per second |
| \(T\) | Measurement duration, in seconds |
| \(\mathcal{L}_{BCE}\) | Binary cross-entropy loss |

## 8. Experimental Scope and Limitations

The current experiment uses a keyword-based monitoring backend and a small sample dataset of 10 episodes. The benchmark contains 400 measurements across two monitoring conditions and four target request rates.

The latency experiment evaluates the computational behavior of the local monitoring implementation. It does not establish production-scale LLM performance.

Classification quality claims require separate evaluation of the predictions against ground-truth labels. Performance measurements alone cannot demonstrate that one monitoring condition detects unsafe behavior more accurately than the other.