# Business Impact Analysis

## Objective

Evaluate how classification thresholds affect the workload of a hypothetical customer retention team.

The analysis uses 2,000 validation records. The original test dataset is reserved for final evaluation.

| Dataset attribute         | Value |
| ------------------------- | ----: |
| Validation customers      | 2,000 |
| Actual churners           |   407 |
| Actual retained customers | 1,593 |

## Threshold Comparison

| Threshold | Customers Flagged | Churners Detected | False Alerts | Missed Churners |
| --------: | ----------------: | ----------------: | -----------: | --------------: |
|      0.30 |             1,277 |               358 |          919 |              49 |
|      0.50 |               695 |               281 |          414 |             126 |
|      0.65 |               412 |               216 |          196 |             191 |
|      0.80 |               167 |               111 |           56 |             296 |

### Operational Metrics

| Threshold | Intervention Rate | Detection Rate | False Discovery Rate | Precision | Recall | F1-score |
| --------: | ----------------: | -------------: | -------------------: | --------: | -----: | -------: |
|      0.30 |            63.85% |         87.96% |               71.97% |    0.2803 | 0.8796 |   0.4252 |
|      0.50 |            34.75% |         69.04% |               59.57% |    0.4043 | 0.6904 |   0.5100 |
|      0.65 |            20.60% |         53.07% |               47.57% |    0.5243 | 0.5307 |   0.5275 |
|      0.80 |             8.35% |         27.27% |               33.53% |    0.6647 | 0.2727 |   0.3868 |

## Selected Threshold

A threshold of **0.65** was selected based on validation F1-score.

It produces the following results:

- Customers flagged: 412
- Genuine churners detected: 216
- False alerts: 196
- Missed churners: 191
- Intervention rate: 20.60%
- Churn detection rate: 53.07%
- False discovery rate: 47.57%

This threshold provides a balance between precision and recall under the selected evaluation metric. It is not necessarily the financially optimal operating threshold.

## Business Interpretation

- Lower thresholds detect more potential churners but increase intervention workload and false alerts.
- Higher thresholds reduce intervention workload but miss more potential churners.
- Operational threshold selection should consider team capacity and the relative costs of false alerts and missed churners.

## Financial Limitations

Actual campaign costs, customer lifetime value, churn-related losses, and retention success rates were not available.

Consequently, this analysis does not estimate actual financial savings, revenue protection, or return on investment. Financial projections require verified business data or explicitly documented assumptions.

## Data Source

- Source: `reports/threshold-comparison.csv`
- Records: 2,000 validation customers
- Selection criterion: Validation F1-score
- Original test set: Not used
