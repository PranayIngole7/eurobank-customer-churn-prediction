# Baseline Model Evaluation

## Experiment Setup

The objective was to establish a reference performance for binary customer churn classification.

| Parameter      | Value                              |
| -------------- | ---------------------------------- |
| Dataset size   | 10,000 records                     |
| Training set   | 8,000 (80%)                        |
| Test set       | 2,000 (20%)                        |
| Random state   | 42                                 |
| Stratification | Enabled                            |
| Target         | `Exited` (0: retained, 1: churned) |

### Class Distribution

| Dataset      | Retained (0) | Churned (1) | Churn rate |
| ------------ | -----------: | ----------: | ---------: |
| Full dataset |        7,963 |       2,037 |     20.37% |
| Training     |        6,370 |       1,630 |     20.38% |
| Test         |        1,593 |         407 |     20.35% |

## Models and Configuration

**Dummy Classifier**
- Strategy: `most_frequent`
- Purpose: Majority-class reference baseline.

**Logistic Regression**
- `max_iter=1000`
- `random_state=42`
- `class_weight=None`
- Classification threshold: 0.50

Preprocessing used `StandardScaler` for numerical features, `OneHotEncoder` for categorical features, and passthrough for binary features. All preprocessing was fitted on training data only.

## Evaluation Results

| Metric    |  Dummy | Logistic Regression |
| --------- | -----: | ------------------: |
| Accuracy  | 0.7965 |              0.8260 |
| Precision | 0.0000 |              0.6766 |
| Recall    | 0.0000 |              0.2776 |
| F1-score  | 0.0000 |              0.3937 |
| ROC-AUC   | 0.5000 |              0.7955 |

### Confusion Matrices

**Dummy Classifier**

|          | Predicted 0 | Predicted 1 |
| -------- | ----------: | ----------: |
| Actual 0 |        1593 |           0 |
| Actual 1 |         407 |           0 |

**Logistic Regression**

|          | Predicted 0 | Predicted 1 |
| -------- | ----------: | ----------: |
| Actual 0 |        1539 |          54 |
| Actual 1 |         294 |         113 |

## Key Findings

- Logistic Regression improves accuracy over the majority-class baseline.
- ROC-AUC of 0.7955 indicates useful class discrimination.
- At threshold 0.50, churn recall is 27.76%.
- The model misses 294 of 407 actual churners.
- Accuracy alone is insufficient for evaluating performance on this imbalanced dataset.

## Limitations

The baseline uses the default classification threshold and does not address class imbalance. Further evaluation should consider recall, precision, threshold selection, and alternative classification strategies.

The test set must remain untouched during model selection and threshold optimization.
