# Held-Out Test Fairness and Group Performance

## Scope and method

This post-selection descriptive audit uses the persisted Gradient Boosting
model and 2,000 customers from the original held-out test partition at the
validation-selected threshold of 0.30. The test data is not used to fit,
tune, or select the model or threshold. It is also used for the final model
evaluation, so this audit is not an additional independent test sample.

The audit compares observed churn and classification rates for Gender,
Geography, and AgeGroup. Group-rate confidence intervals use the 95% Wilson
score method. Confusion-matrix counts, precision, F1-score, and other group
metrics are included in the CSV and dashboard. Comparisons are unadjusted,
have no causal interpretation, and do not establish legal compliance or
whether a model is fair.

## Summary indicators

| Attribute | Demographic parity difference | Equal opportunity difference | FPR difference | Disparate impact ratio |
|---|---:|---:|---:|---:|
| Gender | 0.0805 | 0.0131 | 0.0406 | 0.6701 |
| Geography | 0.1756 | 0.1856 | 0.0858 | 0.4700 |
| AgeGroup | 0.7498 | 0.8490 | 0.4968 | 0.0119 |

These are ranges across groups for the selected threshold. The large observed
AgeGroup and Geography gaps warrant investigation before any operational
use. In particular, AgeGroup results vary alongside materially different
observed churn rates and group sizes; the smallest group in this test
sample has 100 records. Confidence intervals and sample sizes should be
reviewed with every point estimate.

| Audit attribute | Group counts | Group F1-score range |
|---|---|---:|
| Gender | Male 1,070; Female 930 | 0.6074–0.6304 |
| Geography | France 1,021; Germany 504; Spain 475 | 0.5689–0.6918 |
| AgeGroup | <30 331; 30-39 860; 40-49 539; 50-59 170; 60+ 100 | 0.1739–0.8194 |

## Interpretation and limitations

- **Demographic parity difference** is the range of predicted-positive rates.
- **Equal opportunity difference** is the range of true-positive rates.
- **FPR difference** is the range of false-positive rates.
- **Disparate impact ratio** is the lowest divided by the highest predicted
  positive rate. A ratio is a screening statistic, not a universal legal
  standard.
- Differences may reflect data composition, measurement, model behavior,
  sampling variability, or other factors; this analysis cannot identify
  causes.
- No intersectional, uncertainty-adjusted, causal, or legal assessment is
  included. The source dataset's provenance and representativeness have not
  been independently verified.
- The test-set audit is a one-time post-selection analysis; it must not be
  used to revise model or threshold choices.

No automatic customer decision should be based solely on these predictions.
Any future use requires stakeholder review, appropriate domain and legal
assessment, further validation, ongoing monitoring, and meaningful human
oversight.

## Reproducible outputs

- `fairness-group-metrics.csv`
- `fairness-summary.json`
- `figures/fairness-comparison.png`
