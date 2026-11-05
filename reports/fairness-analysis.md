# Fairness and Comparative Analysis Report

## 1. Objective

The objective of this phase is to investigate whether the selected Balanced Logistic Regression model produces different prediction outcomes and error rates across customer groups.

The analysis evaluates:

- Group-wise classification performance.
- Demographic parity.
- Equal opportunity.
- Equalized odds.
- Disparate impact.
- Statistical uncertainty using Wilson confidence intervals.

The purpose is to identify potential differences requiring further investigation. This analysis does not establish legal compliance or prove that the model is fair or unfair.

## 2. Dataset and Evaluation Methodology

The analysis uses the processed customer dataset containing 10,000 records.

The existing stratified training and validation splitting strategy was reused.

| Dataset           | Records | Purpose                     |
| ----------------- | ------: | --------------------------- |
| Internal training |   6,000 | Model fitting               |
| Validation        |   2,000 | Fairness evaluation         |
| Original test     |   2,000 | Excluded from this analysis |

The original training dataset was divided into 6,000 internal training records and 2,000 validation records.

The original test dataset was not used for this fairness analysis.

The feature engineering pipeline was applied consistently with previous modeling phases.

## 3. Model Configuration

The selected model was Balanced Logistic Regression.

| Parameter                | Configuration       |
| ------------------------ | ------------------- |
| Model                    | Logistic Regression |
| Class weighting          | `balanced`          |
| Maximum iterations       | 1,000               |
| Random state             | 42                  |
| Classification threshold | 0.65                |
| Positive class           | `Exited = 1`        |

Predicted probabilities were converted into binary churn predictions using the selected threshold:

- Probability >= 0.65: Predict churn.
- Probability < 0.65: Predict retention.

The threshold was retained from the earlier model-selection phase. It was not changed to improve fairness metrics.

## 4. Audit Attributes

Three attributes were selected for group-wise analysis.

| Attribute | Groups                             |
| --------- | ---------------------------------- |
| Gender    | Male, Female                       |
| Geography | France, Germany, Spain             |
| AgeGroup  | Below 30, 30-39, 40-49, 50-59, 60+ |

These attributes are present in the processed dataset and are available for evaluating differences in model outcomes.

They are treated as audit dimensions in this experiment. Their inclusion does not imply a legal determination of protected-class status in every jurisdiction.

## 5. Group-wise Model Performance

The following metrics were calculated independently for each group:

- Sample count.
- Actual churn rate.
- Predicted churn rate.
- Accuracy.
- Precision.
- True positive rate (TPR).
- False positive rate (FPR).
- False negative rate (FNR).

### 5.1 Gender

| Group  | Samples | Actual churn rate | Predicted churn rate | Accuracy | Precision |    TPR |    FPR |
| ------ | ------: | ----------------: | -------------------: | -------: | --------: | -----: | -----: |
| Male   |   1,111 |            17.46% |               13.86% |   82.54% |    50.00% | 39.69% |  8.40% |
| Female |     889 |            23.96% |               29.02% |   78.29% |    53.88% | 65.26% | 17.60% |

Female customers have a higher predicted churn rate and higher true positive rate than male customers.

However, the female group also has a higher false positive rate.

This indicates differences in both prediction rates and classification errors across the two groups.

### 5.2 Geography

| Group   | Samples | Actual churn rate | Predicted churn rate | Accuracy | Precision |    TPR |    FPR |
| ------- | ------: | ----------------: | -------------------: | -------: | --------: | -----: | -----: |
| France  |   1,013 |            14.61% |               12.93% |   84.90% |    48.09% | 42.57% |  7.86% |
| Germany |     501 |            34.73% |               42.12% |   70.26% |    55.92% | 67.82% | 28.44% |
| Spain   |     486 |            17.49% |               14.40% |   82.51% |    50.00% | 41.18% |  8.73% |

Germany has the highest actual and predicted churn rates among the three geographical groups.

It also has the highest false positive rate and lowest accuracy.

The difference in actual churn prevalence must be distinguished from differences in model error rates. Higher actual churn prevalence alone does not demonstrate model unfairness.

### 5.3 AgeGroup

| Group    | Samples | Actual churn rate | Predicted churn rate | Accuracy | Precision |    TPR |    FPR |
| -------- | ------: | ----------------: | -------------------: | -------: | --------: | -----: | -----: |
| Below 30 |     339 |             7.67% |                0.00% |   92.33% | Undefined |  0.00% |  0.00% |
| 30-39    |     845 |            11.24% |                2.37% |   88.52% |    45.00% |  9.47% |  1.47% |
| 40-49    |     535 |            29.16% |               38.13% |   67.48% |    45.59% | 59.62% | 29.29% |
| 50-59    |     171 |            57.89% |               90.06% |   61.99% |    61.04% | 94.95% | 83.33% |
| 60+      |     110 |            28.18% |               30.91% |   77.27% |    58.82% | 64.52% | 17.72% |

The age-group analysis reveals substantial differences.

Important observations:

- The below-30 group has 26 actual churn cases, but the model predicts no churn cases for this group.
- Its TPR is therefore zero and its FNR is 100%.
- The 30-39 group has a TPR of only 9.47%, meaning most actual churn cases in this group are missed.
- The 50-59 group has a predicted churn rate of 90.06% and an FPR of 83.33%.
- The 60+ group contains only 110 validation records, so its estimates require additional caution.

The below-30 group's accuracy of 92.33% is misleading when considered alone because the model fails to identify any actual churn cases in that group.

Similarly, the 50-59 group's high recall is accompanied by a very high false positive rate.

## 6. Demographic Parity

Demographic parity compares the proportion of positive predictions across groups.

The demographic parity difference is calculated as:

`DP difference = maximum predicted positive rate - minimum predicted positive rate`

| Audit attribute | Demographic parity difference |
| --------------- | ----------------------------: |
| Gender          |                        0.1516 |
| Geography       |                        0.2918 |
| AgeGroup        |                        0.9006 |

The age-group analysis has the largest demographic parity difference.

The below-30 group receives no positive churn predictions, while the 50-59 group receives positive predictions for approximately 90.06% of its members.

These differences warrant further investigation.

Demographic parity does not account for differences in actual churn prevalence, so it should not be interpreted independently of other metrics.

## 7. Equal Opportunity and Equalized Odds

### Equal Opportunity

Equal opportunity compares true positive rates across groups.

The equal opportunity difference is:

`EO difference = maximum TPR - minimum TPR`

| Audit attribute | Equal opportunity difference |
| --------------- | ---------------------------: |
| Gender          |                       0.2557 |
| Geography       |                       0.2664 |
| AgeGroup        |                       0.9495 |

The largest difference occurs across age groups.

The below-30 group has zero recall, while the 50-59 group has approximately 94.95% recall.

This indicates that the model identifies actual churners at substantially different rates across these groups.

### Equalized Odds

Equalized odds considers both true positive rate and false positive rate.

For this project, the summary uses:

`Equalized odds difference = maximum(TPR difference, FPR difference)`

| Audit attribute | TPR difference | FPR difference | Equalized odds difference |
| --------------- | -------------: | -------------: | ------------------------: |
| Gender          |         0.2557 |         0.0921 |                    0.2557 |
| Geography       |         0.2664 |         0.2058 |                    0.2664 |
| AgeGroup        |         0.9495 |         0.8333 |                    0.9495 |

AgeGroup again has the largest difference.

The results indicate substantial variation in both missed churn cases and false churn predictions across age groups.

## 8. Disparate Impact

Disparate impact is evaluated using the ratio of the minimum group positive prediction rate to the maximum group positive prediction rate.

`Disparate impact ratio = minimum predicted positive rate / maximum predicted positive rate`

| Audit attribute | Disparate impact ratio |
| --------------- | ---------------------: |
| Gender          |                 0.4776 |
| Geography       |                 0.3071 |
| AgeGroup        |                 0.0000 |

All three ratios are below 0.80.

The 0.80 threshold, sometimes called the four-fifths rule, is a screening heuristic used in certain contexts. It is not a universal definition of fairness or a substitute for legal assessment.

For multiple groups, this project uses the minimum-to-maximum prediction-rate ratio without selecting a single reference group.

The age-group ratio is zero because the below-30 group has no positive churn predictions.

These results should trigger investigation rather than an automatic conclusion of discrimination.

## 9. Statistical Reliability: Wilson Confidence Intervals

Point estimates can be unstable, especially when groups or metric denominators are small.

Wilson score confidence intervals were implemented to estimate uncertainty around the following group-level rates:

- Actual churn rate.
- Predicted churn rate.
- True positive rate.
- False positive rate.

The confidence level is 95%.

The intervals use the Wilson score method rather than the basic normal approximation.

Different metrics have different denominators:

| Metric               | Denominator                        |
| -------------------- | ---------------------------------- |
| Actual churn rate    | All records in the group           |
| Predicted churn rate | All records in the group           |
| TPR                  | Actual positive cases in the group |
| FPR                  | Actual negative cases in the group |

For example:

- The below-30 TPR is based on 26 actual churn cases.
- The 50-59 FPR is based on 72 actual non-churn cases.
- The 60+ group contains 110 total observations.

These denominators are important when interpreting uncertainty.

The intervals are pointwise intervals. They do not account for multiple comparisons, model-training variability, or uncertainty introduced by selecting a particular dataset and threshold.

Overlapping confidence intervals do not prove that two group rates are equal, and non-overlapping intervals should not be treated as a complete fairness or legal assessment.

## 10. Key Findings and Potential Risks

The analysis identifies several areas requiring additional investigation.

### Gender

- Female customers have a higher predicted churn rate.
- Female customers have higher TPR and FPR than male customers.
- The difference should be examined alongside actual churn prevalence and feature distributions.

### Geography

- Germany has higher actual churn prevalence.
- Germany also has substantially higher FPR and lower accuracy.
- Additional analysis should determine whether these differences are associated with data characteristics, model behavior, or other factors.

### AgeGroup

- Below-30 customers have zero predicted churn cases despite 26 actual churn cases.
- The 30-39 group has low churn recall.
- The 50-59 group has a very high false positive rate.
- Smaller groups, especially 50-59 and 60+, require careful interpretation.

### Accuracy limitation

High accuracy does not necessarily indicate satisfactory group-level performance.

For example, the below-30 group has 92.33% accuracy but fails to identify any actual churn cases.

Therefore, group-level recall, false positive rate, and confusion-matrix counts are essential for interpreting the model.

## 11. Potential Improvement Experiments

The following experiments are proposed for future investigation:

1. Compare group-wise performance across several classification thresholds.
2. Investigate calibration and probability reliability across groups.
3. Examine feature distributions and churn prevalence by audit attribute.
4. Evaluate alternative model configurations.
5. Investigate whether overlapping features such as Age and AgeGroup influence model behavior.
6. Compare group-level performance on an additional independent evaluation dataset.
7. Assess the effects of any proposed mitigation method on overall predictive utility.

Any mitigation strategy should be evaluated against both predictive performance and group-level outcomes.

Sensitive attributes should not be removed or modified automatically as a fairness solution because proxy relationships may remain.

No fairness-specific threshold adjustment or model modification was performed during this phase.

## 12. Limitations

This analysis has several limitations:

- It uses a single validation sample from the available dataset.
- The original test dataset was not used for this fairness investigation.
- Group sizes and positive/negative class counts vary.
- Wilson intervals quantify uncertainty in individual rates but not model-training variability.
- Multiple comparisons are not adjusted.
- The analysis does not establish causal relationships.
- The selected threshold was optimized in an earlier model-selection phase, not for fairness.
- The disparate impact ratio is a descriptive screening metric and does not establish legal compliance.
- The selected audit attributes and group definitions may not cover every relevant fairness concern.
- Fairness is context-dependent and cannot be established through a single metric.

The findings should therefore be treated as evidence for further investigation, not as a final certification of model fairness.

## 13. Generated Artifacts

The following artifacts were generated:

- `reports/fairness-group-metrics.csv`
- `reports/fairness-summary.json`
- `reports/figures/fairness-comparison.png`

Implementation and tests:

- `src/fairness.py`
- `scripts/analyze_model_fairness.py`
- `tests/test_fairness.py`

The CSV contains group-level confusion-matrix counts, performance metrics, and confidence interval bounds.

The JSON file contains the fairness summary for each audit attribute and confidence interval metadata.

The PNG visualizes predicted churn rate, TPR, and FPR across groups, including confidence interval error bars.

## 14. Conclusion

The fairness analysis identifies substantial differences in prediction rates and classification error rates across Gender, Geography, and AgeGroup.

The largest differences occur across age groups, particularly for the below-30 and 50-59 groups.

The analysis also demonstrates why accuracy alone is insufficient for understanding group-level model behavior.

Wilson confidence intervals provide additional information about the uncertainty of estimated rates, although they do not resolve all statistical limitations.

The model has not been modified to improve fairness metrics during this phase. The results will serve as a baseline for future comparative experiments.

Phase 11 establishes a reproducible fairness evaluation process and documents potential risks that should be investigated before any real-world deployment.