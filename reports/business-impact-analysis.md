# Business Impact Analysis

## 1. Objective

Evaluate how classification thresholds affect the operational workload of a hypothetical customer retention team.

This analysis uses validation data only. The original test dataset remains reserved for final evaluation.

## 2. Dataset Overview

- Validation customers: 2,000
- Actual churners: 407
- Actual retained customers: 1,593

## 3. Threshold Scenario Comparison

| Threshold | Customers Flagged | Genuine Churners Detected | False Alerts | Missed Churners |
|---:|---:|---:|---:|---:|
| 0.30 | 1,277 | 358 | 919 | 49 |
| 0.50 | 695 | 281 | 414 | 126 |
| 0.65 | 412 | 216 | 196 | 191 |
| 0.80 | 167 | 111 | 56 | 296 |

## 4. Operational Metrics

| Threshold | Intervention Rate | Churn Detection Rate | False Discovery Rate | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 63.85% | 87.96% | 71.97% | 0.2803 | 0.8796 | 0.4252 |
| 0.50 | 34.75% | 69.04% | 59.57% | 0.4043 | 0.6904 | 0.5100 |
| 0.65 | 20.60% | 53.07% | 47.57% | 0.5243 | 0.5307 | 0.5275 |
| 0.80 | 8.35% | 27.27% | 33.53% | 0.6647 | 0.2727 | 0.3868 |

## 5. Selected Threshold: 0.65

At threshold 0.65, the model flags **412 customers** for potential retention intervention.

- Genuine churners detected: 216
- False alerts: 196
- Missed churners: 191
- Intervention rate: 20.60%
- Churn detection rate: 53.07%
- False discovery rate: 47.57%

The threshold was selected using validation F1-score. It is not claimed to be the financially optimal threshold.

## 6. Business Interpretation

- A lower threshold identifies more churners but increases the number of customers requiring intervention.
- A higher threshold reduces false alerts and intervention workload but misses more genuine churners.
- The selected threshold represents a balance between precision and recall according to validation F1-score.
- The appropriate operational threshold ultimately depends on retention-team capacity and the relative costs of false alerts and missed churners.

## 7. Financial Impact Limitations

Actual retention campaign costs, customer lifetime value, churn-related financial losses, and campaign success rates were not provided.

Therefore, this report does not claim actual financial savings, revenue protection, or return on investment.
Any future financial analysis must use documented business assumptions or verified company data.

## 8. Data Integrity

- Source: `reports/threshold-comparison.csv`.
- Dataset: 2,000 validation records.
- Threshold selection: validation F1-score.
- Original 2,000-record test set: not used.
