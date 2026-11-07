# Model Explainability

## 1. Objective

This analysis examines how the selected Balanced Logistic Regression model produces customer churn predictions.

Three complementary techniques are used:

- **Coefficient analysis:** examines the direction and magnitude of learned model relationships.
- **Permutation feature importance:** measures how shuffling an input feature affects validation ROC-AUC.
- **Local prediction explanations:** decomposes individual predictions into additive contributions on the log-odds scale.

The objective is to understand model behavior, not to establish causal relationships between customer characteristics and churn.

## 2. Methodology

The analysis uses the same feature engineering and model configuration selected during model development.

| Dataset           | Records | Purpose                                                       |
| ----------------- | ------: | ------------------------------------------------------------- |
| Internal training |   6,000 | Fit the explainability model                                  |
| Validation        |   2,000 | Evaluate global importance and explain individual predictions |
| Original test     |   2,000 | Excluded from this analysis                                   |

The model is Balanced Logistic Regression with `class_weight="balanced"`.

A separate model is fitted using only the internal training split. Validation observations remain out of sample.

The original test split is excluded from coefficient extraction, permutation importance, and local explanation analysis.

**Validation-label usage differs by method:**

- Coefficient analysis uses the fitted model parameters.
- Permutation importance uses validation features and labels to calculate ROC-AUC changes.
- Local customer selection and explanations use predicted probabilities and customer features, without using validation target labels.

## 3. Coefficient Analysis

Logistic Regression calculates a linear decision score using transformed input features:

`log-odds = intercept + sum(coefficient × transformed feature value)`

The odds ratio associated with a coefficient is:

`odds_ratio = exp(coefficient)`

An odds ratio greater than 1 indicates higher modeled churn odds, while a value below 1 indicates lower modeled churn odds, holding other transformed features constant.

Numerical features are standardized by the preprocessing pipeline. Their coefficients therefore represent the change in log-odds associated with a one-standard-deviation increase in the corresponding input feature.

Categorical features are one-hot encoded. Each displayed categorical coefficient describes its encoded category relative to the corresponding omitted reference category.

### Top five coefficients by absolute magnitude

| Feature           | Coefficient | Odds ratio | Direction         |
| ----------------- | ----------: | ---------: | ----------------- |
| AgeGroup 50-59    |      1.0690 |     2.9126 | Higher churn odds |
| IsActiveMember    |     -0.9101 |     0.4025 | Lower churn odds  |
| AgeGroup 30-39    |     -0.5008 |     0.6061 | Lower churn odds  |
| Geography Germany |      0.4941 |     1.6390 | Higher churn odds |
| AgeGroup below 30 |     -0.4729 |     0.6232 | Lower churn odds  |

The AgeGroup 50-59 encoded feature has the largest positive coefficient among the transformed features. IsActiveMember has a strong negative coefficient.

These values describe relationships learned by this particular fitted model. They do not establish that changing a customer's age group, membership status, or geography would cause a corresponding change in churn.

Age and AgeGroup also contain overlapping information, which should be considered when interpreting their coefficients.

## 4. Permutation Feature Importance

Permutation importance measures the reduction in model performance after independently shuffling an input feature.

The analysis uses:

- Dataset: internal validation set.
- Scoring metric: ROC-AUC.
- Number of repeats: 10.
- Random state: 42.

For each feature, its values are shuffled repeatedly while the other features remain unchanged.

Importance is calculated as:

`importance = baseline ROC-AUC - shuffled-feature ROC-AUC`

A larger positive value indicates greater reliance on that feature for validation ROC-AUC.

### Complete feature-importance ranking

| Feature         | Mean importance | Standard deviation |
| --------------- | --------------: | -----------------: |
| AgeGroup        |         0.07283 |            0.00714 |
| Age             |         0.04507 |            0.00671 |
| IsActiveMember  |         0.03425 |            0.00699 |
| Geography       |         0.02400 |            0.00439 |
| Gender          |         0.01211 |            0.00326 |
| Balance         |         0.00174 |            0.00118 |
| HasBalance      |         0.00143 |            0.00100 |
| CreditScore     |         0.00140 |            0.00052 |
| NumOfProducts   |         0.00067 |            0.00021 |
| Tenure          |        -0.00017 |            0.00025 |
| HasCrCard       |        -0.00023 |            0.00028 |
| EstimatedSalary |        -0.00052 |            0.00066 |

AgeGroup and Age have the highest permutation importance, followed by IsActiveMember, Geography, and Gender.

Age and AgeGroup represent overlapping information. Their individual importance values should therefore not be interpreted as independent contributions.

Some features have near-zero or slightly negative mean importance. This indicates that shuffling those features did not reduce validation ROC-AUC on average across the experiments. A negative value does not establish that a feature is harmful or should automatically be removed.

Permutation importance can be affected by correlated features because other features may retain overlapping predictive information.

## 5. Individual Customer Explanations

Five validation customers with the highest predicted churn probabilities were selected for local explanations.

Selection was based exclusively on predicted probabilities. Validation target labels were not used to select customers or calculate their individual explanations.

### Selected customers

| Rank | Validation record index | Predicted churn probability |
| ---: | ----------------------: | --------------------------: |
|    1 |                    9602 |                      0.9613 |
|    2 |                    6197 |                      0.9604 |
|    3 |                    2495 |                      0.9601 |
|    4 |                    1158 |                      0.9577 |
|    5 |                    8923 |                      0.9506 |

### Explanation methodology

The Logistic Regression decision score is decomposed into:

`predicted log-odds = intercept + sum(feature contributions)`

Each feature contribution is calculated as:

`contribution = coefficient × transformed feature value`

The baseline log-odds is the fitted model's intercept:

`base log-odds = 0.1218896`

The contributions are additive on the log-odds scale. Their sum plus the intercept reconstructs the model's decision score.

The predicted churn probability is obtained by applying the logistic function:

`probability = 1 / (1 + exp(-predicted log-odds))`

The implementation checks that the calculated contributions reconstruct the model's actual decision score within a numerical tolerance.

### Example: highest-risk validation customer

For validation record index 9602, the five largest positive contributions were:

| Feature           | Contribution | Direction                |
| ----------------- | -----------: | ------------------------ |
| AgeGroup 50-59    |       1.0690 | Increases churn log-odds |
| Age               |       0.8466 | Increases churn log-odds |
| Geography Germany |       0.4941 | Increases churn log-odds |
| Gender Female     |       0.3441 | Increases churn log-odds |
| HasBalance        |       0.2431 | Increases churn log-odds |

Other features also contribute to the final prediction. For example, the customer's EstimatedSalary contributes -0.0016, slightly decreasing churn log-odds.

The contribution direction depends on both the coefficient and the customer's transformed feature value. A negative coefficient can produce a positive contribution when the transformed value is negative.

Categorical features that are not active for a particular customer have zero contribution in the one-hot encoded representation.

These explanations describe how the model calculates its prediction. They do not identify the actual reasons why a customer might leave a bank.

## 6. Limitations

- Coefficients describe conditional model associations, not causal effects.
- Numerical coefficients represent changes per one-standard-deviation increase in the standardized feature.
- Categorical coefficients depend on the encoding and omitted reference categories.
- Age and AgeGroup contain overlapping information.
- Permutation importance can be affected by correlated features and the particular validation sample.
- Near-zero or negative permutation importance does not independently justify removing a feature.
- Class weighting influences the fitted coefficients, intercept, and predicted probability distribution.
- Predicted probabilities have not been independently calibrated.
- Local explanations describe the model's log-odds calculation, not real-world causes of customer churn.
- The five locally explained customers were selected from the highest predicted validation probabilities and are not representative of all customers.
- The analysis uses a validation sample from the available dataset and does not establish performance or explanation stability across future customer populations.

## 7. Generated Artifacts

### Numerical outputs

- `reports/feature-coefficients.csv`
- `reports/permutation-importance.csv`
- `reports/local-customer-summary.csv`
- `reports/local-customer-contributions.csv`

### Visualizations

- `reports/figures/feature-coefficients.png`
- `reports/figures/permutation-importance.png`

### Implementation

- `src/explainability.py`
- `scripts/analyze_model_explainability.py`
- `scripts/explain_customer_predictions.py`

## 8. Conclusion

The explainability analysis provides complementary views of the Balanced Logistic Regression model.

Coefficient analysis identifies the direction and magnitude of learned relationships. Permutation importance measures the effect of individual input features on validation ROC-AUC. Local explanations decompose individual predictions into additive log-odds contributions.

AgeGroup, Age, IsActiveMember, Geography, and Gender are the leading features in the global permutation-importance analysis.

These findings improve transparency into model behavior while retaining important limitations: feature associations are not causal explanations, correlated features can complicate importance estimates, and local contributions describe model calculations rather than actual customer motivations.
