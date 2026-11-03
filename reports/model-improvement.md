# Model Improvement and Validation Report

## 1. Objective

The objective was to improve the baseline customer churn
prediction model by investigating:

- Classification threshold selection.
- Class imbalance handling.
- Standard versus class-weighted Logistic Regression.
- Validation-based model selection.
- Final evaluation on an untouched test dataset.

The positive class (`Exited = 1`) represents customers who churned.

## 2. Dataset and Class Distribution

The processed dataset contains 10,000 customer records.

| Class | Meaning  | Records | Percentage |
| ----- | -------- | ------: | ---------: |
| 0     | Retained |   7,963 |     79.63% |
| 1     | Churned  |   2,037 |     20.37% |
| Total |          |  10,000 |       100% |

The target is imbalanced, with approximately four retained customers
for every churned customer.

Accuracy alone is therefore insufficient for evaluating model
performance.

## 3. Data Splitting Strategy

The original dataset was divided using stratified sampling.

| Dataset           | Records | Purpose                       |
| ----------------- | ------: | ----------------------------- |
| Original training |   8,000 | Model development             |
| Original test     |   2,000 | Final evaluation              |
| Internal training |   6,000 | Model training during tuning  |
| Validation        |   2,000 | Model and threshold selection |

The original 8,000-record training dataset was further divided into
6,000 internal training records and 2,000 validation records.

Stratification preserved the target class proportions.

### Leakage prevention

During model experimentation:

1. Preprocessing was fitted only on the 6,000 internal training records.
2. Validation data was used for model comparison and threshold selection.
3. The original test dataset remained untouched during experimentation.
4. After selecting the model, the pipeline was refitted on all 8,000
   original training records.
5. The original test dataset was used for final evaluation.

This prevents information from the validation or test datasets from
leaking into model training.

## 4. Candidate Models

Two Logistic Regression configurations were compared.

### Standard Logistic Regression

Configuration:

- `class_weight=None`
- `max_iter=1000`
- `random_state=42`

This model treats both classes according to their observed frequencies.

### Balanced Logistic Regression

Configuration:

- `class_weight="balanced"`
- `max_iter=1000`
- `random_state=42`

Class weighting increases the penalty associated with misclassifying
the minority class.

For the 6,000-record internal training dataset:

| Class | Records | Approximate Weight |
| ----- | ------: | -----------------: |
| 0     |   4,777 |              0.628 |
| 1     |   1,223 |              2.453 |

Class weighting does not guarantee better performance. Its effectiveness
must be evaluated using validation data.

## 5. Classification Threshold Analysis

Logistic Regression produces probabilities rather than directly
determining business decisions.

A classification threshold converts predicted probabilities into
binary predictions:

- Probability greater than or equal to the threshold: predict churn.
- Probability below the threshold: predict retention.

Thresholds from 0.10 to 0.90 were evaluated in increments of 0.05.

### Standard Logistic Regression

Best validation F1-score:

| Metric    | Result |
| --------- | -----: |
| Threshold |   0.30 |
| Accuracy  | 0.7915 |
| Precision | 0.4893 |
| Recall    | 0.5627 |
| F1-score  | 0.5234 |
| ROC-AUC   | 0.7785 |

### Balanced Logistic Regression

Best validation F1-score:

| Metric    | Result |
| --------- | -----: |
| Threshold |   0.65 |
| Accuracy  | 0.8065 |
| Precision | 0.5243 |
| Recall    | 0.5307 |
| F1-score  | 0.5275 |
| ROC-AUC   | 0.7795 |

## 6. Model Selection

The primary selection metric was F1-score, with precision, recall,
and ROC-AUC considered as supporting metrics.

| Metric             | Standard LR | Balanced LR |
| ------------------ | ----------: | ----------: |
| Selected threshold |        0.30 |        0.65 |
| Accuracy           |      0.7915 |      0.8065 |
| Precision          |      0.4893 |      0.5243 |
| Recall             |      0.5627 |      0.5307 |
| F1-score           |      0.5234 |      0.5275 |
| ROC-AUC            |      0.7785 |      0.7795 |

Balanced Logistic Regression achieved the highest validation F1-score.

The improvement was marginal:

0.5275 - 0.5234 = 0.0041

The balanced model also achieved higher precision, while the standard
model achieved higher recall.

Therefore, the balanced model was selected based on the predefined
F1-score criterion, not because it was substantially better in every
metric.

A different business objective could justify choosing another
configuration or threshold.

## 7. Final Model Training

The selected Balanced Logistic Regression model was refitted using
all 8,000 original training records.

The selected classification threshold remained 0.65.

The original 2,000-record test dataset was reserved for final
evaluation.

The preprocessing pipeline was also refitted on the full training
dataset.

## 8. Final Test Evaluation

The selected model was evaluated on the untouched test dataset.

| Metric    | Final Test Result |
| --------- | ----------------: |
| Accuracy  |            0.8000 |
| Precision |            0.5081 |
| Recall    |            0.5405 |
| F1-score  |            0.5238 |
| ROC-AUC   |            0.7948 |

### Confusion Matrix

```text
                 Predicted
                 Retained  Churned

Actual Retained     1380      213
Actual Churned       187      220
```

| Outcome        | Records | Meaning                                            |
| -------------- | ------: | -------------------------------------------------- |
| True Negative  |   1,380 | Correctly identified retained customers            |
| False Positive |     213 | Retained customers incorrectly flagged as churners |
| False Negative |     187 | Churned customers missed by the model              |
| True Positive  |     220 | Correctly identified churned customers             |

### Interpretation

The model identified 220 of the 407 actual churned customers.

It missed 187 churned customers.

Among the 433 customers predicted to churn, 220 actually churned.

The final recall of 0.5405 means the model identified approximately
54.05% of actual churners.

The final precision of 0.5081 means approximately 50.81% of customers
flagged as churn risks actually churned.

## 9. Validation and Test Comparison

| Metric    | Validation | Final Test |
| --------- | ---------: | ---------: |
| Accuracy  |     0.8065 |     0.8000 |
| Precision |     0.5243 |     0.5081 |
| Recall    |     0.5307 |     0.5405 |
| F1-score  |     0.5275 |     0.5238 |
| ROC-AUC   |     0.7795 |     0.7948 |

The validation and test F1-scores are close.

The difference in F1-score is 0.0037, indicating reasonably consistent
performance on these two datasets.

The test ROC-AUC is higher than the validation ROC-AUC. This is an
observed difference and does not establish that the model will
generalize equally well to future customer populations.

## 10. Important Observations and Limitations

1. The model performs better than the majority-class Dummy Classifier.
2. Class weighting produced only a small validation F1 improvement.
3. Classification threshold selection materially changes precision
   and recall.
4. The selected threshold is based on F1-score, not a validated
   business cost function.
5. Class weighting changes the probability distribution. Predicted
   probabilities should not automatically be interpreted as calibrated
   real-world churn probabilities.
6. Logistic Regression is a linear model and may not capture complex
   nonlinear relationships.
7. No external validation dataset or prospective customer evaluation
   has been performed.
8. No probability calibration or cost-sensitive threshold optimization
   has been performed.

This is a useful first predictive model, not a production-ready
customer churn decision system.

## 11. Test Set Integrity

The original test dataset was not used for:

- Model selection.
- Threshold selection.
- Hyperparameter tuning.
- Feature selection.

It was reserved for the final evaluation after the validation-based
selection process.

The final test results should be treated as the current performance
estimate. Further model improvements should use training and validation
data or a new evaluation dataset, not repeated tuning against this
test set.

## 12. Conclusion

Successfully implemented and evaluated:

- Classification threshold analysis.
- Validation-based model selection.
- Class-weighted Logistic Regression.
- Final model refitting on the complete training dataset.
- Final evaluation on the untouched test dataset.
- Configurable threshold-based model evaluation.

Selected configuration:

- Model: Balanced Logistic Regression
- Class weight: balanced
- Classification threshold: 0.65
- Validation F1-score: 0.5275
- Final test F1-score: 0.5238
- Final test ROC-AUC: 0.7948

