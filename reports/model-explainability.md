# Model Explainability

## 1. Objective

This phase examines the selected Balanced Logistic Regression model to understand its global feature relationships and individual customer predictions.

Three methods were implemented:

- Model coefficients and odds ratios.
- Permutation feature importance.
- Individual prediction explanations using additive log-odds contributions.

## 2. Methodology

The explainability analysis used the same feature engineering and model configuration selected during model improvement.

| Dataset | Records | Purpose |
|---|---:|---|
| Internal training | 6,000 | Fit explainability model |
| Validation | 2,000 | Global importance and individual explanations |
| Original test | 2,000 | Excluded from this analysis |

The model was Balanced Logistic Regression with `class_weight="balanced"`.

The original test set was not used for explainability analysis. The separate explainability model was trained on the internal training split so that validation observations remained out of sample.

## 3. Coefficient Analysis

Logistic Regression coefficients describe changes in predicted churn log-odds, holding other transformed features constant.

Odds ratios are calculated as:

`odds_ratio = exp(coefficient)`

For numerical features, coefficients represent a one-standard-deviation change because numerical inputs are standardized. Categorical coefficients describe the encoded category relative to its reference category.

Notable results:

| Feature | Coefficient | Odds ratio | Direction |
|---|---:|---:|---|
| AgeGroup 50-59 | 1.0690 | 2.9126 | Higher churn odds |
| IsActiveMember | -0.9101 | 0.4025 | Lower churn odds |
| AgeGroup 30-39 | -0.5008 | 0.6061 | Lower churn odds |
| Geography Germany | 0.4941 | 1.6390 | Higher churn odds |
| AgeGroup below 30 | -0.4729 | 0.6232 | Lower churn odds |

The 50-59 age group has the largest positive coefficient. Active membership has a strong negative coefficient, indicating lower churn odds in this model.

These are model associations, not causal effects.

## 4. Permutation Feature Importance

Permutation importance was calculated on the validation set using ROC-AUC scoring, with 10 repeats and random state 42.

| Feature | Mean importance | Standard deviation |
|---|---:|---:|
| AgeGroup | 0.07283 | 0.00714 |
| Age | 0.04507 | 0.00671 |
| IsActiveMember | 0.03425 | 0.00699 |
| Geography | 0.02400 | 0.00439 |
| Gender | 0.01211 | 0.00326 |

AgeGroup and Age have the highest importance, followed by IsActiveMember and Geography.

Age and AgeGroup represent overlapping information, so their individual importance values should not be interpreted as independent effects.

Some features have near-zero or slightly negative permutation importance. This means shuffling them did not reduce validation ROC-AUC in these experiments. It does not independently establish that they should be removed.

## 5. Individual Customer Explanations

Five validation customers with the highest predicted churn probabilities were selected for local explanation.

Selection was based only on model probabilities; validation target labels were not used to select or explain these customers.

| Rank | Validation record index | Predicted churn probability |
|---:|---:|---:|
| 1 | 9602 | 0.9613 |
| 2 | 6197 | 0.9604 |
| 3 | 2495 | 0.9601 |
| 4 | 1158 | 0.9577 |
| 5 | 8923 | 0.9506 |

The local explanation decomposes each prediction into:

`predicted log-odds = base log-odds + sum(feature contributions)`

Each contribution is the transformed feature value multiplied by its Logistic Regression coefficient.

For these five customers, the strongest positive contributions included:

- AgeGroup 50-59.
- Numerical Age.
- Geography Germany.
- Gender Female.
- HasBalance.

The Age contribution varies between customers. Contributions from shared categorical values are identical because the fitted model uses the same coefficient for each transformed feature.

The contributions explain the model's log-odds calculation, not a direct additive change in probability.

## 6. Limitations

- Coefficients describe conditional model associations and do not establish causation.
- Standardized numerical coefficients represent a one-standard-deviation change.
- Age and AgeGroup contain overlapping information.
- Permutation importance can be affected by correlated features.
- Near-zero or negative permutation importance does not automatically justify feature removal.
- Class weighting affects the fitted coefficients, intercept, and probability distribution.
- Predicted probabilities have not been independently calibrated.
- Local explanations describe the model's calculation, not the real-world cause of customer churn.
- The analysis uses a validation sample from the available dataset and does not establish performance on future customer populations.

## 7. Generated Artifacts

- `reports/feature-coefficients.csv`
- `reports/permutation-importance.csv`
- `reports/local-customer-summary.csv`
- `reports/local-customer-contributions.csv`
- `reports/figures/feature-coefficients.png`
- `reports/figures/permutation-importance.png`

## 8. Conclusion

The explainability analysis identifies AgeGroup, Age, IsActiveMember, and Geography as important contributors to the model's churn predictions.

Coefficient analysis provides direction and magnitude of model relationships, permutation importance measures validation performance sensitivity, and local explanations show how individual predictions are constructed.

Phase 10 adds global and individual interpretability while preserving the separation between training, validation, and the original test set.
