# Validation Workload Analysis

## Purpose and scope

This report translates classification results into hypothetical review
workload counts. It is not a financial impact estimate. All figures below
come from the 2,000-row validation dataset, which was also used for model and
threshold selection. The original held-out test set was not used here.

The selected algorithm is Gradient Boosting, with threshold 0.30. At this
threshold, the validation confusion matrix is TN 1,452, FP 141, FN 161, and
TP 246.

| Validation measure       |       Result |
| ------------------------ | -----------: |
| Customers flagged        | 387 (19.35%) |
| Genuine churners flagged |          246 |
| False alerts             |          141 |
| Churners missed          |   161 of 407 |
| Precision                |       63.57% |
| Recall                   |       60.44% |
| F1-score                 |       61.96% |

If every flagged customer received human review, 387 records would require
review in this sample. The counts do not estimate the number of people who
would accept an intervention or remain customers after one.

## Threshold trade-off

| Threshold | Flagged | Churners detected | False alerts | Missed churners |  Precision |     Recall |         F1 |
| --------: | ------: | ----------------: | -----------: | --------------: | ---------: | ---------: | ---------: |
|      0.20 |     616 |               304 |          312 |             103 |     49.35% |     74.69% |     59.43% |
|      0.25 |     471 |               271 |          200 |             136 |     57.54% |     66.58% |     61.73% |
|  **0.30** | **387** |           **246** |      **141** |         **161** | **63.57%** | **60.44%** | **61.96%** |
|      0.35 |     325 |               220 |          105 |             187 |     67.69% |     54.05% |     60.11% |
|      0.50 |     215 |               167 |           48 |             240 |     77.67% |     41.03% |     53.70% |

The configured threshold is the highest-F1 validation threshold satisfying
the 0.50 minimum precision requirement among the tested model/threshold
choices. It is not a threshold optimized against verified financial costs
or review-team capacity.

## Financial and policy limitations

No campaign costs, customer lifetime values, intervention effectiveness,
false-alert costs, or missed-churn costs were supplied. Accordingly, no
revenue protection, savings, return-on-investment, or policy effectiveness
claims are made. Any operational use should first establish decision rights,
human review, customer safeguards, capacity, and applicable legal and
regulatory requirements.

Detailed threshold metrics are in `threshold-comparison.csv`. These are
selection-validation results and are not an independent test estimate.

## Post-selection fairness considerations

A separate subgroup audit was performed on the 2,000-record held-out test
partition after model selection. It found substantial differences in
predicted churn rates and true-positive rates, particularly across
AgeGroup and Geography. The AgeGroup equal-opportunity difference was
0.8490, with a disparate impact ratio of 0.0119. The corresponding
Geography values were 0.1856 and 0.4700.

These results are descriptive screening indicators, not legal conclusions.
Some age groups have small numbers of actual churn cases, so their estimates
require cautious interpretation. The audit uses the same held-out test
partition as the final performance evaluation; it is not an independent
second sample.

Any operational pilot should investigate these disparities, assess
subgroup performance with additional representative data, and establish
appropriate human review and customer safeguards before decisions are made.
