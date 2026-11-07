# Model Improvement and Validation Report

## 1. Objective

This analysis evaluates improvements to the baseline customer churn prediction model through:

- Classification threshold selection.
- Class imbalance handling.
- Comparison of standard and class-weighted Logistic Regression.
- Validation-based model selection.
- Final evaluation on an untouched test dataset.

The positive class (`Exited = 1`) represents customers who churned.

## 2. Dataset and Class Distribution

The processed dataset contains 10,000 customer records.

| Class | Meaning  | Records | Percentage |
| ----- | -------- | ------: | ---------: |
| 0     | Retained |   7,963 |     79.63% |
| 1     | Churned  |   2,037 |     20.37% |
| Total |          |  10,000 |    100.00% |

Approximately four retained customers exist for every churned customer. Consequently, accuracy alone is insufficient to evaluate model performance. Precision, recall, F1-score, and ranking metrics provide additional information about minority-class detection.

## 3. Data Splitting Strategy

The dataset was divided using stratified sampling.

| Dataset           | Records | Purpose                                  |
| ----------------- | ------: | ---------------------------------------- |
| Original training |   8,000 | Model development and selection          |
| Original test     |   2,000 | Final evaluation                         |
| Internal training |   6,000 | Model fitting during experimentation     |
| Validation        |   2,000 | Model comparison and threshold selection |

The original training dataset was further divided into 6,000 internal training records and 2,000 validation records.

Stratification preserved the target class proportions across the splits.

### Leakage prevention

The model development process followed these rules:

1. Preprocessing was fitted using only the internal training data during experimentation.
2. Validation data was used for comparing candidate models and selecting classification thresholds.
3. The original test dataset was excluded from model development and selection.
4. After selecting the configuration, the complete pipeline was refitted using all 8,000 original training records.
5. The original test dataset was used for final evaluation.

This separation helps prevent validation and test information from influencing model fitting.

## 4. Candidate Models

Two Logistic Regression configurations were evaluated.

### Standard Logistic Regression

Configuration:

- `class_weight=None`
- `max_iter=1000`
- `random_state=42`

Both classes receive equal class weights, so the observed class distribution is reflected in the training objective.

### Balanced Logistic Regression

Configuration:

- `class_weight="balanced"`
- `max_iter=1000`
- `random_state=42`

Balanced class weighting adjusts the contribution of each class to the training loss according to its frequency. The minority churn class therefore receives a higher weight.

For the 6,000-record internal training dataset:

| Class | Records | Approximate Weight |
| ----- | ------: | -----------------: |
| 0     |   4,777 |              0.628 |
| 1     |   1,223 |              2.453 |

Class weighting does not guarantee better predictive performance. Its effectiveness must be assessed using validation data.

## 5. Classification Threshold Analysis

Logistic Regression produces predicted probabilities. A classification threshold converts these probabilities into binary predictions:

- Probability greater than or equal to the threshold: predict churn.
- Probability below the threshold: predict retention.

Thresholds from 0.10 to 0.90 were evaluated in increments of 0.05.

The threshold-selection criterion was the highest validation F1-score.

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

The two configurations selected different thresholds. This demonstrates that class weighting and threshold selection influence the resulting precision-recall trade-off.

## 6. Model Selection

F1-score was the primary selection metric because it balances precision and recall. Accuracy, precision, recall, and ROC-AUC were considered as supporting metrics.

| Metric             | Standard LR | Balanced LR |
| ------------------ | ----------: | ----------: |
| Selected threshold |        0.30 |        0.65 |
| Accuracy           |      0.7915 |      0.8065 |
| Precision          |      0.4893 |      0.5243 |
| Recall             |      0.5627 |      0.5307 |
| F1-score           |      0.5234 |      0.5275 |
| ROC-AUC            |      0.7785 |      0.7795 |

Balanced Logistic Regression achieved the higher validation F1-score.

The difference was small:

`0.5275 - 0.5234 = 0.0041`

The balanced configuration also achieved higher precision and slightly higher ROC-AUC, while standard Logistic Regression achieved higher recall.

The balanced model was selected because it achieved the highest validation F1-score under the predefined selection criterion. This does not mean it was substantially better across all metrics.

A business objective that places greater importance on identifying as many churners as possible could justify a different model or threshold.

## 7. Final Model Training

After model selection, Balanced Logistic Regression was refitted using all 8,000 original training records.

Selected configuration:

- Model: Balanced Logistic Regression.
- Class weight: `balanced`.
- Classification threshold: `0.65`.
- Maximum iterations: `1000`.
- Random state: `42`.

The preprocessing pipeline was also refitted on the complete training dataset.

The selected threshold remained 0.65. The original 2,000-record test dataset was reserved for final evaluation.

## 8. Final Test Evaluation

The selected model was evaluated on the untouched test dataset.

| Metric            | Final Test Result |
| ----------------- | ----------------: |
| Accuracy          |            0.8000 |
| Precision         |            0.5081 |
| Recall            |            0.5405 |
| F1-score          |            0.5238 |
| ROC-AUC           |            0.7948 |
| Average Precision |            0.5460 |

### Confusion Matrix

```text
                 Predicted
                 Retained  Churned

Actual Retained     1380      213
Actual Churned       187      220
```

| Outcome        | Records | Meaning                                            |
| -------------- | ------: | -------------------------------------------------- |
| True Negative  |   1,380 | Retained customers correctly identified            |
| False Positive |     213 | Retained customers incorrectly flagged as churners |
| False Negative |     187 | Churned customers missed by the model              |
| True Positive  |     220 | Churned customers correctly identified             |

### Interpretation

The test dataset contains 407 actual churned customers.

- The model correctly identified 220 churned customers.
- It missed 187 churned customers.
- It flagged 433 customers as potential churners, of whom 220 actually churned.

The recall of 0.5405 means the model identified approximately 54.05% of actual churners.

The precision of 0.5081 means approximately 50.81% of customers flagged as churn risks actually churned.

These results show that the model identifies a meaningful portion of churners, but it also produces false alerts and misses a substantial number of actual churners.

## 9. Validation and Test Comparison

| Metric            | Validation | Final Test |
| ----------------- | ---------: | ---------: |
| Accuracy          |     0.8065 |     0.8000 |
| Precision         |     0.5243 |     0.5081 |
| Recall            |     0.5307 |     0.5405 |
| F1-score          |     0.5275 |     0.5238 |
| ROC-AUC           |     0.7795 |     0.7948 |
| Average Precision |     0.5241 |     0.5460 |

The validation and test F1-scores differ by 0.0037, indicating similar observed performance on these two datasets.

Test ROC-AUC and Average Precision are higher than their validation counterparts. These differences are observations from the available samples; they do not establish that future customer populations will produce similar results.

## 10. Important Observations and Limitations

1. The selected model performs better than the majority-class Dummy Classifier on churn detection metrics.
2. Class weighting produced only a marginal validation F1 improvement over standard Logistic Regression.
3. Classification threshold selection changes the balance between precision and recall.
4. The selected threshold of 0.65 optimizes validation F1 among the evaluated thresholds; it is not based on a validated business cost function.
5. Class weighting influences the fitted coefficients and probability distribution. Predicted probabilities should not automatically be interpreted as calibrated real-world churn probabilities.
6. Logistic Regression is a linear model and may not capture complex nonlinear relationships.
7. No external validation dataset or prospective customer evaluation has been performed.
8. No independent probability calibration or cost-sensitive threshold optimization has been performed.
9. The comparison uses a single validation split and does not quantify variability across repeated splits or cross-validation.

This model is a useful predictive baseline, not a production-ready customer churn decision system.

## 11. Test Set Integrity

The original test dataset was not used for:

- Candidate model selection.
- Classification threshold selection.
- Hyperparameter tuning.
- Feature selection.

It was reserved for final evaluation after the validation-based selection process.

The test results should be treated as the current held-out performance estimate. Further model improvements should use training and validation data or a new evaluation dataset rather than repeatedly tuning against the existing test set.

## 12. Conclusion

The model improvement process evaluated class weighting, classification thresholds, and validation-based model selection before performing final evaluation on an untouched test dataset.

The selected configuration is:

- **Model:** Balanced Logistic Regression
- **Class weight:** `balanced`
- **Classification threshold:** `0.65`
- **Validation F1-score:** `0.5275`
- **Final test F1-score:** `0.5238`
- **Final test ROC-AUC:** `0.7948`
- **Final test Average Precision:** `0.5460`

Balanced Logistic Regression achieved the highest validation F1-score among the evaluated configurations. Its improvement over standard Logistic Regression was marginal, so the selection should be understood as the result of the chosen evaluation criterion rather than evidence of a universally superior model.

The final evaluation provides a held-out estimate of model performance while preserving the separation between model development, validation, and testing.
