# Customer Risk Scoring

## Scoring definitions

The risk-scoring utility maps the persisted model's positive-class probability
to a 0–100 score, a configured risk category, and a binary prediction.

| Output            | Definition                                                                                                                 |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Churn probability | Persisted model's estimated probability for `Exited = 1`                                                                   |
| Risk score        | `100 * churn_probability`                                                                                                  |
| Churn prediction  | Positive when the model probability is at least the threshold stored in model metadata                                     |
| Risk category     | Separate configurable probability bands: Low [0.00, 0.30), Moderate [0.30, 0.50), High [0.50, 0.70), Critical [0.70, 1.00] |

Risk bands are descriptive conventions, not calibrated risk groups or
business-validated decision cutoffs. The current model metadata records
Gradient Boosting and a validation-selected classification threshold of 0.30.
The binary threshold and risk-category boundaries serve different purposes.

## Generated output

The generated file is `reports/customer-risk-scores.csv`. It contains
2,000 ranked records scored by the persisted Gradient Boosting model
using the classification threshold stored in model metadata (0.30).

The output includes:
- `risk_rank`
- `churn_probability`
- `risk_score`
- `risk_category`
- `churn_prediction`

Customer identifiers are not exported. The file still contains
record-level predictions and should be handled appropriately.

Run `scripts/generate_risk_scores.py` to regenerate the output. The script
loads the persisted model and its metadata rather than training a separate
model. Regeneration overwrites the existing output file.

## Limitations

- Probability values have not undergone separate calibration.
- Risk categories require business and domain validation.
- The dataset provenance and representativeness are not independently
  verified.
- Scores do not predict the effect of a retention intervention or establish
  that any customer will churn.
- Do not use the output as the sole basis for consequential customer
  decisions.
