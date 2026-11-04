# Customer Churn Model Evaluation Report

## 1. Objective

Evaluate the selected customer churn classification model using
classification metrics, visualizations, threshold analysis, and
business-oriented interpretation.

The selected model is Balanced Logistic Regression.

The positive class (`Exited = 1`) represents customers who churned.

## 2. Evaluation Methodology

The evaluation follows a leakage-safe workflow:

| Dataset | Records | Purpose |
|---|---:|---|
| Internal training | 6,000 | Model fitting during experimentation |
| Validation | 2,000 | Model and threshold selection |
| Original training | 8,000 | Final model refitting |
| Original test | 2,000 | Final unseen-data evaluation |

The selected model and classification threshold were determined
using validation data.

The original test dataset was reserved for final evaluation and
was not used for model selection or threshold optimization.

## 3. Selected Model Configuration

| Parameter | Configuration |
|---|---|
| Algorithm | Logistic Regression |
| Class weighting | balanced |
| Maximum iterations | 1,000 |
| Random state | 42 |
| Selected threshold | 0.65 |
| Selection metric | Validation F1-score |

The selected pipeline was refitted using all 8,000 original
training records.

## 4. Classification Evaluation

The evaluation implementation supports:

- Classification report.
- Confusion matrix.
- ROC curve and ROC-AUC.
- Precision-Recall curve and Average Precision.

The positive class is Churned (1).

The actual validation classification report is saved in
`classification-report-validation.json`. It contains class-wise
precision, recall, F1-score, support, and aggregate metrics.

The original final test metrics documented in Phase 7 are:

| Metric | Final Test Result |
|---|---:|
| Accuracy | 0.8000 |
| Precision | 0.5081 |
| Recall | 0.5405 |
| F1-score | 0.5238 |
| ROC-AUC | 0.7948 |

These are the final test results, not validation metrics.

## 5. Confusion Matrix Analysis

Final test confusion matrix:

| | Predicted Retained | Predicted Churned |
|---|---:|---:|
| Actual Retained | 1,380 | 213 |
| Actual Churned | 187 | 220 |

Interpretation:

- True Negatives: 1,380 retained customers correctly identified.
- False Positives: 213 retained customers incorrectly flagged.
- False Negatives: 187 churned customers missed.
- True Positives: 220 churned customers correctly identified.

The model identified 220 of 407 actual churners.

## 6. ROC and Precision-Recall Analysis

The validation ROC-AUC is 0.7795, while the previously
documented final test ROC-AUC is 0.7948.

The validation Average Precision is 0.5241.

ROC-AUC measures the model's ability to rank positive examples
above negative examples across classification thresholds.

Average Precision summarizes the precision-recall curve and
focuses on performance for the positive churn class. It is
different from ROC-AUC.

The following three visualizations were generated using
validation data only:

### Validation confusion matrix

![Validation confusion matrix](figures/confusion-matrix-validation.png)

### Validation ROC curve

![Validation ROC curve](figures/roc-curve-validation.png)

### Validation Precision-Recall curve

![Validation Precision-Recall curve](figures/precision-recall-curve-validation.png)

The final test ROC-AUC of 0.7948 is reused from the Phase 7
evaluation. A final test Average Precision has not been reported.
No new test evaluation was performed to generate these plots.

## 7. Classification Threshold Comparison

Thresholds from 0.10 to 0.90 were evaluated using validation
probabilities in increments of 0.05.

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.30 | 0.2803 | 0.8796 | 0.4252 |
| 0.50 | 0.4043 | 0.6904 | 0.5100 |
| 0.65 | 0.5243 | 0.5307 | 0.5275 |
| 0.80 | 0.6647 | 0.2727 | 0.3868 |

The threshold of 0.65 achieved the highest validation F1-score
among the evaluated thresholds.

### Threshold visualization

![Validation threshold comparison](figures/threshold-comparison.png)

Detailed results are available in:

- `threshold-comparison.csv`
- `figures/threshold-comparison.png`

## 8. Business Impact Analysis

The business impact analysis uses the 2,000 validation records.

| Threshold | Customers Flagged | Genuine Churners Detected | False Alerts | Missed Churners |
|---:|---:|---:|---:|---:|
| 0.30 | 1,277 | 358 | 919 | 49 |
| 0.50 | 695 | 281 | 414 | 126 |
| 0.65 | 412 | 216 | 196 | 191 |
| 0.80 | 167 | 111 | 56 | 296 |

At threshold 0.65:

- 412 customers are flagged for potential intervention.
- 216 are genuine churners.
- 196 are false alerts.
- 191 actual churners are missed.

A lower threshold increases churn detection but also increases
the intervention workload.

A higher threshold reduces intervention workload but misses
more genuine churners.

The selected threshold is based on F1-score, not a validated
financial cost function.

Actual retention costs, customer lifetime value, and campaign
success rates were not provided. Therefore, no financial savings
or return on investment are claimed.

See `business-impact-analysis.md` for the detailed analysis.

## 9. Validation and Test Performance

| Metric | Validation | Final Test |
|---|---:|---:|
| Accuracy | 0.8065 | 0.8000 |
| Precision | 0.5243 | 0.5081 |
| Recall | 0.5307 | 0.5405 |
| F1-score | 0.5275 | 0.5238 |
| ROC-AUC | 0.7795 | 0.7948 |\n| Average Precision | 0.5241 | Not reported |

The validation and test F1-scores are close.

This is encouraging evidence of consistent performance across
these two splits, but it does not establish performance on
future customer populations.

## 10. Limitations

1. Logistic Regression may not capture complex nonlinear patterns.
2. Class weighting changes the probability distribution.
3. Predicted probabilities have not been calibrated.
4. Threshold selection uses F1-score rather than a verified
   business cost function.
5. No external validation or prospective evaluation has been
   performed.
6. Actual retention campaign costs and customer lifetime values
   are unavailable.
7. The original test dataset must not be repeatedly used for
   further model tuning.

## 11. Conclusion

The evaluation process selected Balanced Logistic Regression
with a classification threshold of 0.65.

The model achieved:

- Validation F1-score: 0.5275.
- Final test F1-score: 0.5238.
- Final test ROC-AUC: 0.7948.
- Final test churn recall: 54.05%.

Threshold analysis demonstrated the trade-off between churn
detection and operational intervention workload.

The business impact analysis translated these classification
results into measurable customer counts without inventing
financial outcomes.

The model provides a useful foundation for customer churn
risk analysis, but it requires further validation, explainability,
fairness assessment, and application integration before any
production deployment.

## 12. Phase 8 Deliverables

- Classification report implementation and generated validation report.
- Generated validation confusion matrix.
- Generated validation ROC curve.
- Generated validation Precision-Recall curve.
- Validation ROC-AUC and Average Precision metrics.
- Threshold comparison implementation and visualization.
- Reproducible validation threshold experiment and CSV results.
- Business impact analysis report.
- Previously completed final unseen-test evaluation results.
- Consolidated evaluation report.
