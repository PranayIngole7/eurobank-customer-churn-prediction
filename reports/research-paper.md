# Validation-Selected Machine Learning for Bank Customer Churn

## Abstract

This project develops a binary classifier for the supplied bank-customer
churn dataset and evaluates a workflow that separates model selection from a
held-out final test. Six candidate estimators were trained on 6,000 internal
training rows and compared using 2,000 validation rows. Gradient Boosting
was selected by validation F1 subject to a 0.50 precision floor, at threshold
0.30. It was refit on all 8,000 original training rows and evaluated once on
2,000 held-out rows. Test F1 was 0.6205, ROC-AUC 0.8615, and Average Precision
0.7032. A post-selection audit on the held-out test sample showed material
group differences, particularly across age groups. These results demonstrate a
reproducible educational workflow, not evidence that the model is fit for
operational decisions.

## 1. Introduction

Customer churn classification can help organize retention review, but its
value depends on reliable data, appropriate operating objectives, and
responsible treatment of customers. This project predicts `Exited` (1 for
exited, 0 for retained) from customer/account attributes. The primary
objectives are to compare candidate algorithms without using the held-out
test data for selection, quantify test performance once, and document
explanation and subgroup limitations.

## 2. Dataset and feature construction

The local CSV contains 10,000 rows and 14 original columns. The target counts
are 2,037 exited and 7,963 retained. The data file is supplied separately;
its original publisher, collection process, sampling frame, and licensing
were not independently verified. Results should therefore not be treated as
representative of a particular bank, jurisdiction, or current customer
population.

Deterministic features are created by the shared training/inference
`engineer_features` function. In addition to preserving `HasBalance` and
`AgeGroup`, it adds:

- `BalanceToSalaryRatio = Balance / EstimatedSalary`, with zero salary
  mapped to 0.0 to guard division by zero.
- `ProductDensityIndicator = NumOfProducts / Tenure`, with zero tenure
  mapped to 0.0.
- `EngagementProductInteraction = IsActiveMember * NumOfProducts`.
- `AgeTenureInteraction = Age * Tenure`.

These features do not use the target. Their definitions and limitations are
specified in [feature-engineering.md](./feature-engineering.md). Categorical
encoding and numerical scaling are fit inside model pipelines, on each
training partition only.

## 3. Experimental design

Stratified sampling created an 80/20 split: 8,000 training rows and 2,000
held-out test rows. The training portion was split into 6,000 internal
training rows and 2,000 validation rows.

The six candidates were Majority-Class Dummy, Logistic Regression, Balanced
Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
Each candidate was fitted on internal training data and compared on
validation. For the threshold-0.50 comparison, metrics included accuracy,
precision, recall, F1, ROC-AUC, Average Precision, and confusion counts.

Thresholds 0.10 through 0.90 in 0.05 increments were evaluated using
validation probabilities. For each model, the highest-F1 threshold with
precision of at least 0.50 was selected; if none satisfied the floor, its
highest-F1 threshold was used. The model with the highest resulting F1 was
selected, with tie-breaks by Average Precision, precision, fewer false
positives, then model name. The chosen Gradient Boosting model was refit on
all original training rows. The test data did not participate in candidate fitting, threshold
selection, feature selection, or explanation analysis. After selection, it
was used for the final model evaluation and a separate descriptive subgroup
audit; neither audit result was used to revise model choices.

## 4. Validation comparison

| Candidate | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Majority-Class Dummy | 0.7965 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2035 |
| Logistic Regression | 0.8215 | 0.6420 | 0.2776 | 0.3877 | 0.7775 | 0.5265 |
| Balanced Logistic Regression | 0.7290 | 0.4020 | 0.6806 | 0.5055 | 0.7792 | 0.5260 |
| Decision Tree | 0.7550 | 0.4426 | 0.7862 | 0.5664 | 0.8286 | 0.6039 |
| Random Forest | 0.8245 | 0.5579 | 0.6634 | 0.6061 | 0.8498 | 0.6498 |
| Gradient Boosting | 0.8560 | 0.7767 | 0.4103 | 0.5370 | 0.8515 | 0.6699 |

These rows use the common 0.50 threshold and are not the selection metric
table. Gradient Boosting at the selected threshold 0.30 achieved validation
precision 0.6357, recall 0.6044, F1 0.6196, ROC-AUC 0.8515, and Average
Precision 0.6699.

## 5. Held-out test results

| Metric | Test result |
|---|---:|
| Accuracy | 0.8465 |
| Precision | 0.6244 |
| Recall | 0.6167 |
| F1-score | 0.6205 |
| ROC-AUC | 0.8615 |
| Average Precision | 0.7032 |

The confusion matrix was TN 1,442, FP 151, FN 156, TP 251. Accordingly, the
model identified 251 of 407 churned test records and missed 156. The test
results are a single split and do not establish expected future or external
performance.

## 6. Explainability and subgroup analysis

Permutation importance uses the 2,000 validation rows, ROC-AUC, 10 repeats,
and random state 42. The largest mean decrease followed shuffling Age
(0.11211), NumOfProducts (0.10760), and Geography (0.02310). Correlated
features can divide importance; the ranking is not a causal ranking.

SHAP uses `shap==0.49.1`, a permutation explainer, positive-class probability
output, 100 sampled validation rows, 50 background rows, random state 42, and
33 evaluations per explained row for 16 inputs. Partial dependence uses
scikit-learn's probability response for class 1 and 20 grid points for each
of the four new engineered features. Explanation results describe a separate
Gradient Boosting instance fitted on internal training rows, not the
full-training persisted final model. Because validation data informed model
selection, these explanations are not independent generalization evidence.

The post-selection test fairness audit uses a 0.30 threshold and 95% Wilson
intervals. Demographic-parity, equal-opportunity, FPR-gap, and
disparate-impact summaries, respectively, were: Gender (0.0805, 0.0131,
0.0406, 0.6701), Geography (0.1756, 0.1856, 0.0858, 0.4700), and AgeGroup
(0.7498, 0.8490, 0.4968, 0.0119). These disparities require investigation;
they are not legal conclusions or proof of discriminatory intent. The audit
uses the same held-out sample as final test evaluation and is not an
independent second test.

## 7. Discussion, limitations, and governance

The selected threshold reflects a validation F1 objective plus an explicit
precision floor. It is not optimized for campaign costs, customer value, or
review capacity. The reported counts support workload planning only; they do
not predict retention intervention success or financial return.

The dataset's provenance and representativeness are unknown, no prospective
or external validation was performed, probability calibration was not
assessed, and subgroup estimates may be unstable. Threshold and model
selection reuse the same validation partition, making validation summaries
selection-dependent. Explanations identify model behavior rather than
causation. Before any consequential use, the system would need data
provenance checks, external and temporal evaluation, operational and legal
review, calibration assessment if probabilities inform decisions, subgroup
monitoring, and meaningful human oversight.

## 8. Conclusion

The implementation provides a leakage-aware candidate comparison and
validation-only threshold-selection workflow, plus a final held-out
evaluation and post-selection fairness audit. Gradient Boosting at threshold
0.30 achieved test F1 0.6205 and ROC-AUC 0.8615 on the supplied split.
Material test subgroup differences and the lack of verified data provenance
constrain any operational interpretation. This is an educational prototype,
not a production-ready decision system.

## Reproducibility artifacts

See `model-selection-validation.json`, `model-comparison-validation.csv`,
`model-threshold-comparison-validation.csv`, `final-model-test-evaluation.json`,
`model-explainability.md`, and `fairness-analysis.md`.
