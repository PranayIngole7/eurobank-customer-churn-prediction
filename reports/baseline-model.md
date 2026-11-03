# Baseline Model Evaluation

## 1. Experiment Overview

Project: Eurobank Customer Churn Prediction

Objective: Establish baseline performance for binary customer churn
classification.

Dataset: 10,000 customer records.

Training/testing split:
- Training: 8,000 records (80%)
- Testing: 2,000 records (20%)
- Random state: 42
- Stratification: Enabled

Class distribution:

| Dataset | Retained (0) | Churned (1) | Churn rate |
|---|---:|---:|---:|
| Full dataset | 7,963 | 2,037 | 20.37% |
| Training | 6,370 | 1,630 | 20.38% |
| Testing | 1,593 | 407 | 20.35% |

Target:
- 0: Customer retained
- 1: Customer churned

## 2. Models

### Dummy Classifier

Strategy: most_frequent

The model predicts the majority class for every customer.
It serves as a reference baseline.

### Logistic Regression

Configuration:
- max_iter: 1000
- random_state: 42
- class_weight: None
- Classification threshold: 0.50

Preprocessing:
- StandardScaler for numerical features.
- OneHotEncoder for categorical features.
- Binary features passed through unchanged.

Preprocessing was fitted through the pipeline using training
data only.

## 3. Evaluation Results

| Metric | Dummy | Logistic Regression |
|---|---:|---:|
| Accuracy | 0.7965 | 0.8260 |
| Precision | 0.0000 | 0.6766 |
| Recall | 0.0000 | 0.2776 |
| F1-score | 0.0000 | 0.3937 |
| ROC-AUC | 0.5000 | 0.7955 |

## 4. Confusion Matrices

### Dummy Classifier

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 1593 | 0 |
| Actual 1 | 407 | 0 |

### Logistic Regression

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 1539 | 54 |
| Actual 1 | 294 | 113 |

## 5. Findings

1. Logistic Regression outperforms the Dummy Classifier.
2. Logistic Regression achieves 82.60% accuracy.
3. Its ROC-AUC of 0.7955 indicates useful class-ranking
   ability.
4. Churn recall is only 27.76% at the default threshold.
5. The model misses 294 of the 407 actual churners.
6. Accuracy alone is insufficient for assessing churn
   prediction performance.

## 6. Limitations and Next Improvements

- Investigate classification threshold selection using
  validation data.
- Evaluate class imbalance strategies.
- Examine precision-recall trade-offs.
- Consider additional models after establishing a reliable
  Logistic Regression baseline.

The original test set must remain untouched during model
selection and threshold optimization.

## 7. Conclusion

The first Logistic Regression baseline demonstrates that
customer features contain useful predictive information.

However, churn detection recall requires improvement.

These results will serve as the reference point for subsequent
model development.
