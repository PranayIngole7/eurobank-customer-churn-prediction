# Customer Risk Scoring

## 1. Overview

The customer risk-scoring component converts predicted churn probabilities into numerical risk scores and four business-facing risk categories. It also produces a binary churn prediction using the selected classification threshold.

The selected Balanced Logistic Regression model is trained on the original 8,000 training records. It is then used to score the 2,000 held-out customer records.

## 2. Scoring Methodology

- Churn probability: model-predicted probability of class 1.
- Risk score: churn probability multiplied by 100.
- Classification threshold: 0.65.
- Risk ranking: descending risk score, with CustomerId used as a deterministic tie-breaker.

### Risk Categories

| Category | Probability range |
|---|---|
| Low | 0.00–<0.30 |
| Moderate | 0.30–<0.50 |
| High | 0.50–<0.70 |
| Critical | 0.70–1.00 |

Risk categories and binary churn predictions serve different purposes. For example, a customer with a churn probability of 0.62 is classified as High risk but is not predicted to churn at the selected 0.65 threshold.

These category boundaries are configurable business-policy choices, not statistically optimized cutoffs.

## 3. Scoring Results

A total of 2,000 held-out customers were scored.

| Risk category | Customers | Percentage |
|---|---:|---:|
| Low | 724 | 36.20% |
| Moderate | 565 | 28.25% |
| High | 378 | 18.90% |
| Critical | 333 | 16.65% |
| **Total** | **2,000** | **100%** |

### Binary Churn Predictions

| Prediction | Customers |
|---|---:|
| Not predicted to churn | 1,567 |
| Predicted to churn | 433 |
| **Total** | **2,000** |

These are model predictions, not confirmed future customer outcomes.

## 4. Output

Generated file: `reports/customer-risk-scores.csv`

The CSV contains:

- CustomerId
- risk_rank
- churn_probability
- risk_score
- risk_category
- churn_prediction

Customer identifiers are retained for business reference and are not used as model features.

## 5. Validation

The generated output passed the following checks:

- Exactly 2,000 customer records.
- No missing values.
- No duplicate customer IDs or ranks.
- Risk scores correctly calculated from probabilities.
- Risk categories match the configured boundaries.
- Binary predictions match the 0.65 threshold.
- Sequential ranking from 1 to 2,000.
- All probabilities within the valid range of 0 to 1.
- Records sorted by descending risk score.

## 6. Limitations

- Risk categories are configurable business rules and require business validation.
- Predicted probabilities have not undergone a separate probability-calibration procedure.
- Actual future churn outcomes cannot be established from these predictions alone.
- No retention costs, customer lifetime values, or campaign success rates were provided; therefore, no financial savings or ROI are claimed.
- The held-out test set was previously used for final model evaluation in Phase 7. Its use here is for customer-level scoring demonstration, not further model selection.

## 7. Conclusion

The customer risk-scoring component successfully converts churn probabilities into ranked customer-level outputs. The generated scores, categories, and binary predictions passed all implemented consistency checks.
