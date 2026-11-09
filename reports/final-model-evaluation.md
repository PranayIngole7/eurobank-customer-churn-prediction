# Final Model Evaluation

## Model Information

- Model: gradient_boosting
- Algorithm: Gradient Boosting
- Classification threshold: 0.3
- Validation selection rationale: For each model, select the threshold with highest validation F1-score among thresholds with precision >= 0.50; if none meet the floor, use maximum F1-score across its thresholds. Select the model with highest resulting F1-score; break ties by Average Precision, precision, fewer false positives, then model name.
- Training records: 8000
- Test records: 2000

## Evaluation Metrics

| Metric            |  Value |
| ----------------- | -----: |
| Accuracy          | 0.8465 |
| Precision         | 0.6244 |
| Recall            | 0.6167 |
| F1 Score          | 0.6205 |
| Roc Auc           | 0.8615 |
| Average Precision | 0.7032 |

## Confusion Matrix

| Actual / Predicted | Retained (0) | Churned (1) |
| ------------------ | -----------: | ----------: |
| Retained (0)       |         1442 |         151 |
| Churned (1)        |          156 |         251 |

## Classification Report

| Class        | Precision | Recall | F1-score | Support |
| ------------ | --------: | -----: | -------: | ------: |
| Retained (0) |    0.9024 | 0.9052 |   0.9038 |    1593 |
| Churned (1)  |    0.6244 | 0.6167 |   0.6205 |     407 |

## Evaluation Methodology

- The original dataset was split using stratified 80/20 sampling.
- The model was fitted exclusively on the original training partition.
- The original test partition was used exclusively for final evaluation.
- Candidate models and the decision threshold were selected using only the internal validation split.
- The test results were not used to tune the model, threshold, or features.
- Feature preprocessing was fitted as part of the training pipeline.

## Post-selection fairness audit

A subgroup audit was conducted after model selection using the same
2,000-record held-out test partition. It did not inform model selection,
threshold tuning, or feature changes. The audit used a classification
threshold of 0.30 and 95% Wilson confidence intervals for selected group
rates.

| Attribute | Demographic parity difference | Equal opportunity difference | FPR difference | Disparate impact ratio |
| --------- | ----------------------------: | ---------------------------: | -------------: | ---------------------: |
| Gender    |                        0.0805 |                       0.0131 |         0.0406 |                 0.6701 |
| Geography |                        0.1756 |                       0.1856 |         0.0858 |                 0.4700 |
| AgeGroup  |                        0.7498 |                       0.8490 |         0.4968 |                 0.0119 |

The largest differences appear across age groups. The true-positive rate
ranges from 0.10 for customers under 30 to 0.949 for customers aged 50–59.
Geography also shows a notable true-positive-rate difference, with Germany
at 0.728 compared with 0.543 for France and 0.568 for Spain.

These are descriptive screening results, not proof of unlawful
discrimination or a determination of fairness. Some subgroup samples,
particularly the under-30 and 60+ groups, contain relatively few actual
churn cases. The results warrant further investigation and monitoring
before any operational use. The audit does not establish the causes of
the observed differences or guarantee future subgroup performance.

## Limitations

These metrics represent performance on one held-out test partition. They do not guarantee equivalent performance on future customer populations or production data.
