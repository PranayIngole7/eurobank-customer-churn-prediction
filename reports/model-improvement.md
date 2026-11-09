# Model Development and Improvement

## Dataset and split

The supplied dataset contains 10,000 rows, with 7,963 retained and 2,037
exited customers. Stratified splitting created 8,000 original training rows
and 2,000 held-out test rows. The training partition was further divided into
6,000 internal training and 2,000 validation rows.

Preprocessing is fitted within each candidate's training pipeline. The
held-out test partition is excluded from all model and threshold selection.

## Candidate comparison

Six probability-capable candidates are fitted and evaluated on validation:
Majority-Class Dummy, Logistic Regression, Balanced Logistic Regression,
Decision Tree, Random Forest, and Gradient Boosting. At threshold 0.50,
Random Forest achieved the largest F1 (0.6061), while Gradient Boosting had
the largest Average Precision (0.6699) and highest accuracy (0.8560). These
fixed-threshold metrics are descriptive and not the final selection rule.

## Model and threshold selection

Thresholds 0.10–0.90 in 0.05 increments were evaluated on validation. The
selection procedure chooses, per model, its highest-F1 threshold with
precision at least 0.50, falling back to maximum F1 if no threshold qualifies.
The model with the highest resulting F1 wins; ties are broken by Average
Precision, precision, fewer false positives, then model name.

This selected Gradient Boosting at threshold 0.30. Validation precision was
0.6357, recall 0.6044, F1 0.6196, ROC-AUC 0.8515, and Average Precision
0.6699. The precision floor and F1 objective are explicit design choices,
not business-validated costs.

## Final evaluation

The selected algorithm was refit on all 8,000 original training records.
On the untouched 2,000-record test partition it achieved:

| Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---:|---:|---:|---:|---:|---:|
| 0.8465 | 0.6244 | 0.6167 | 0.6205 | 0.8615 | 0.7032 |

Test confusion counts were TN 1,442, FP 151, FN 156, and TP 251. Test
results are reported after selection and were not used to revise it.

## Limitations and next work

The dataset source and representativeness are not independently verified.
There is no prospective or external validation, and probability calibration
was not separately measured. Validation summaries are selection-dependent.
The threshold has not been optimized against verified business costs or
service capacity. Fairness diagnostics show notable subgroup differences
and need specialist review. Before any operational use, conduct independent
validation, review data governance and applicable requirements, evaluate
calibration if needed, define human review, and monitor subgroup outcomes.

Detailed comparison and final evaluation results are in
`evaluation-report.md`, `model-comparison-validation.csv`, and
`final-model-test-evaluation.json`.
