# Fairness Analysis Report

## 1. Overview

This report evaluates differences in customer churn predictions across Gender, Geography, and AgeGroup using the validation dataset.

The analysis uses the Balanced Logistic Regression model with a classification threshold of 0.65. It compares actual churn prevalence, predicted churn rates, classification performance, and group-level fairness metrics.

This is a validation-set audit, not a comprehensive fairness certification or assessment of legal compliance.

## 2. Evaluation Method

- Evaluation dataset: Internal validation set
- Sample size: 2,000 customers
- Model: Balanced Logistic Regression
- Classification threshold: 0.65
- Audit attributes: Gender, Geography, AgeGroup
- Confidence intervals: 95% Wilson score intervals

The original held-out test set is excluded from this analysis.

The audit evaluates the following metrics:

| Metric               | Definition                                                  |
| -------------------- | ----------------------------------------------------------- |
| Actual churn rate    | Actual churners divided by group size                       |
| Predicted churn rate | Customers predicted to churn divided by group size          |
| Accuracy             | Correct predictions divided by group size                   |
| Precision            | Correct positive predictions divided by predicted positives |
| TPR                  | Correct positive predictions divided by actual positives    |
| FPR                  | Incorrect positive predictions divided by actual negatives  |
| FNR                  | Missed churners divided by actual positives                 |

## 3. Group-Level Results

All percentages below are calculated from the validation dataset.

### Gender

| Group  | Customers | Actual Churn | Predicted Churn | Accuracy |    TPR |    FPR |
| ------ | --------: | -----------: | --------------: | -------: | -----: | -----: |
| Male   |     1,111 |       17.46% |          13.86% |   82.54% | 39.69% |  8.40% |
| Female |       889 |       23.96% |          29.02% |   78.29% | 65.26% | 17.60% |

Female customers have higher observed churn prevalence and a higher predicted churn rate. The model also has higher TPR and FPR for this group.

### Geography

| Group   | Customers | Actual Churn | Predicted Churn | Accuracy |    TPR |    FPR |
| ------- | --------: | -----------: | --------------: | -------: | -----: | -----: |
| France  |     1,013 |       14.61% |          12.93% |   84.90% | 42.57% |  7.86% |
| Germany |       501 |       34.73% |          42.12% |   70.26% | 67.82% | 28.44% |
| Spain   |       486 |       17.49% |          14.40% |   82.51% | 41.18% |  8.73% |

Germany has the highest observed churn prevalence and predicted churn rate among the geographic groups. Its FPR is also substantially higher.

### Age Group

| Group | Customers | Actual Churn | Predicted Churn | Accuracy |    TPR |    FPR |
| ----- | --------: | -----------: | --------------: | -------: | -----: | -----: |
| <30   |       339 |        7.67% |           0.00% |   92.33% |  0.00% |  0.00% |
| 30–39 |       845 |       11.24% |           2.37% |   88.52% |  9.47% |  1.47% |
| 40–49 |       535 |       29.16% |          38.13% |   67.48% | 59.62% | 29.29% |
| 50–59 |       171 |       57.89% |          90.06% |   61.99% | 94.95% | 83.33% |
| 60+   |       110 |       28.18% |          30.91% |   77.27% | 64.52% | 17.72% |

The age groups show substantial variation:

- For customers below 30, the model predicts no positive churn cases. All 26 actual churners in this group are missed.
- The 40–49 group has an FPR of approximately 29.29%.
- The 50–59 group has a predicted churn rate of approximately 90.06% and an FPR of approximately 83.33%.
- The 60+ group contains only 110 customers, so its estimates should be interpreted cautiously.

These results highlight differences in model behavior that warrant further investigation.

## 4. Fairness Metrics

The following group-comparison metrics are calculated from the observed validation-set rates.

| Metric                        | Gender | Geography | AgeGroup |
| ----------------------------- | -----: | --------: | -------: |
| Demographic parity difference | 0.1516 |    0.2918 |   0.9006 |
| Equal opportunity difference  | 0.2557 |    0.2664 |   0.9495 |
| FPR difference                | 0.0921 |    0.2058 |   0.8333 |
| Equalized odds difference     | 0.2557 |    0.2664 |   0.9495 |
| Disparate impact ratio        | 0.4776 |    0.3071 |   0.0000 |

Definitions:

- **Demographic parity difference:** Maximum minus minimum predicted churn rate across groups.
- **Equal opportunity difference:** Maximum minus minimum true positive rate.
- **FPR difference:** Maximum minus minimum false positive rate.
- **Equalized odds difference:** Maximum of the TPR difference and FPR difference.
- **Disparate impact ratio:** Minimum predicted churn rate divided by maximum predicted churn rate.

Larger differences indicate greater variation in the corresponding group-level rates.

The AgeGroup results show particularly large differences. The disparate impact ratio is zero because the `<30` group has no positive predictions, while the 50–59 group has the highest predicted churn rate.

The four-fifths rule is sometimes used as a screening heuristic for disparate impact. It is not a universal definition of fairness or, by itself, a legal determination. These ratios should be interpreted alongside group sizes, underlying churn prevalence, model performance, and the intended use of predictions.

## 5. Confidence Intervals

The implementation calculates 95% Wilson score confidence intervals for:

- Actual churn rate
- Predicted churn rate
- True positive rate
- False positive rate

These intervals provide information about uncertainty in individual group-level rates, particularly where sample sizes or positive and negative case counts are limited.

Precision is undefined when a group has no predicted positive cases. For the `<30` group, precision is therefore represented as `null` in JSON and as a missing value in CSV rather than being treated as zero.

The intervals are pointwise estimates. They do not represent confidence intervals for differences between groups and do not account for model-training variability or multiple comparisons.

## 6. Limitations

This analysis has several limitations:

- It uses one validation sample and does not establish performance on future or external populations.
- Group sizes and numbers of actual positive and negative cases vary.
- Some groups have relatively few observations, which can make their estimates less stable.
- Confidence intervals do not account for model-training variability or multiple comparisons.
- Observed group differences do not establish causation or explain why those differences occur.
- The 0.65 threshold was selected using validation F1, not optimized for fairness or a financial objective.
- The analysis does not establish legal compliance or prove the presence or absence of discrimination.

## 7. Further Investigation

Potential follow-up analyses include:

1. Compare group-level metrics across alternative classification thresholds.
2. Evaluate probability calibration across groups.
3. Examine feature distributions and underlying churn prevalence.
4. Compare alternative models and their group-level performance.
5. Investigate the relationship between Age and AgeGroup.
6. Repeat the evaluation on independent or prospective data.
7. Evaluate the effects of potential fairness mitigation approaches without compromising predictive usefulness.

## 8. Generated Artifacts

The analysis produces the following artifacts:

| Artifact                                  | Description                                                                |
| ----------------------------------------- | -------------------------------------------------------------------------- |
| `reports/fairness-group-metrics.csv`      | Group-level counts, classification metrics, and confidence-interval bounds |
| `reports/fairness-summary.json`           | Group metrics and aggregate fairness comparisons                           |
| `reports/figures/fairness-comparison.png` | Group comparison visualization with 95% Wilson confidence intervals        |

Implementation and tests:

- `src/fairness.py`
- `scripts/analyze_model_fairness.py`
- `tests/test_fairness.py`

## 9. Conclusion

The validation-set audit identifies differences in churn prevalence, prediction rates, and classification performance across Gender, Geography, and AgeGroup.

The largest observed differences occur across age groups, particularly for customers below 30 and those aged 50–59. Geographic differences are also notable, especially for Germany.

These findings identify areas requiring further investigation. They should not be interpreted as definitive evidence of discrimination or as a comprehensive fairness assessment. Further validation on independent data, threshold sensitivity analysis, calibration assessment, and evaluation of mitigation strategies would be necessary before using these findings to support real-world customer decisions.