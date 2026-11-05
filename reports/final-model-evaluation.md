# Final Model Evaluation

## Model Information

- Model: logistic_regression_balanced
- Algorithm: Balanced Logistic Regression
- Classification threshold: 0.65
- Training records: 8000
- Test records: 2000

## Evaluation Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.8000 |
| Precision | 0.5081 |
| Recall | 0.5405 |
| F1 Score | 0.5238 |
| Roc Auc | 0.7948 |
| Average Precision | 0.5460 |

## Confusion Matrix

| Actual / Predicted | Retained (0) | Churned (1) |
|---|---:|---:|
| Retained (0) | 1380 | 213 |
| Churned (1) | 187 | 220 |

## Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Retained (0) | 0.8807 | 0.8663 | 0.8734 | 1593 |
| Churned (1) | 0.5081 | 0.5405 | 0.5238 | 407 |

## Evaluation Methodology

- The original dataset was split using stratified 80/20 sampling.
- The model was fitted exclusively on the original training partition.
- The original test partition was used exclusively for final evaluation.
- The classification threshold of 0.65 was selected using validation data.
- The test results were not used to tune the model, threshold, or features.
- Feature preprocessing was fitted as part of the training pipeline.

## Limitations

These metrics represent performance on one held-out test partition. They do not guarantee equivalent performance on future customer populations or production data.
