# Model Explainability

## Scope and model

The selected algorithm is Gradient Boosting. Explanation analyses use a
separate instance of that algorithm fitted on the 6,000 internal training
rows, with the 2,000 validation rows used as the explanation/evaluation
population. The final persisted model was separately refit on all 8,000
original training rows. Therefore these artifacts explain the selected
algorithm trained on the internal split; they are not a record-level audit of
the final persisted model. The original held-out test data was not used.

The same feature engineering and preprocessing pipeline is used. Validation
labels are used only by permutation importance; SHAP and partial dependence
use model outputs and features, not validation labels. Model and threshold
selection also used this validation partition, so these explanations are
descriptive rather than independent evidence of generalization.

Gradient Boosting has no logistic-regression coefficients or odds ratios, so
the dashboard omits coefficient analysis for this model.

## Permutation importance

`calculate_permutation_importance` measures the reduction in validation
ROC-AUC after independently shuffling each original input feature. It uses
10 repeats and random state 42:

`importance = baseline validation ROC-AUC - shuffled-feature validation ROC-AUC`

| Highest mean ROC-AUC decrease |    Mean | Standard deviation |
| ----------------------------- | ------: | -----------------: |
| Age                           | 0.11211 |            0.00786 |
| NumOfProducts                 | 0.10760 |            0.00740 |
| Geography                     | 0.02310 |            0.00282 |
| EngagementProductInteraction  | 0.01136 |            0.00378 |
| Balance                       | 0.00604 |            0.00072 |

Age and AgeGroup, as well as product count and product-related engineered
features, can contain overlapping information. Shuffling one correlated
feature at a time may understate or redistribute importance. Near-zero or
negative importance is not proof that a feature is harmful or should be
removed.

## SHAP

The implementation uses the pinned `shap==0.49.1` dependency and
`shap.Explainer(..., algorithm="permutation")`. It explains the model's
positive-class (`Exited = 1`) predicted probability through a
`predict_proba` callback. For the 16 model inputs, the permutation explainer
uses `max_evals = 33` for 16 inputs (`2 * 16 + 1`), set explicitly by the
analysis scripts. The utility accepts a configurable evaluation budget no
smaller than this minimum. A reproducible sample of 100 validation rows and a
50-row validation background are selected with random state 42.
Global importance is mean absolute SHAP value across the sampled rows.
For every run, the implementation verifies that the base value plus summed
feature attributions reconstructs each explained positive-class probability
within `rtol=1e-5` and `atol=1e-6`.

A same-sample budget check compared 33 and 330 evaluations for five
validation rows and the same 50-row background. Flattened attribution
Spearman correlation was 0.9991, feature-importance rank correlation was
0.9970, and mean absolute difference in attribution magnitudes was 0.000468
(maximum absolute raw-attribution difference 0.002761). Both runs passed the
additivity check. The first call included SHAP initialization, so its
observed runtime is not used as a fair budget timing comparison. The
10-fold higher budget gave no material ranking change in this small
stability check; it is not evidence of population-wide stability.

Categorical columns are temporarily represented by integer category codes
for the masker, whose tabular invariance check expects numeric values. The
callback decodes those values to the original categories before calling the
fitted pipeline; reported feature names remain the original model-input
names. This is a model-output decomposition, not causal attribution.

The saved local example represents validation record index 2992, with a
predicted churn probability of 0.169953 (16.9953%). The largest absolute
attributions were Age (+0.121741), NumOfProducts (-0.107115), and
EngagementProductInteraction (-0.045765). The CSV records the validation
row index and predicted probability alongside each feature value and
attribution. The index is an internal dataset row reference, not a customer
identifier. These attributions describe the model output relative to the
sampled background; they are not percentage-point effects of interventions.

The separate local-customer explanation script ranks validation rows by
predicted probability (without using validation labels) and computes SHAP
values for the five highest-scoring rows. Their validation row indexes and
predicted probabilities are in `local-customer-summary.csv`; feature-level
values are in `local-customer-contributions.csv`. These are internal row
indexes, not customer identifiers.

The installed dependency is declared in `requirements.txt`; the
implementation raises an explicit `ImportError` if SHAP is unavailable and
does not substitute fabricated values.

## Partial dependence

The implementation uses scikit-learn
`PartialDependenceDisplay.from_estimator` with `response_method="predict_proba"`,
positive target class 1, and a 20-point grid. It generated plots for
`BalanceToSalaryRatio`, `ProductDensityIndicator`,
`EngagementProductInteraction`, and `AgeTenureInteraction`.

Each plot varies one feature while averaging model responses over the
validation records. These engineered inputs are correlated with their
underlying customer fields; varying one independently can create unrealistic
combinations. A flat response indicates little modeled sensitivity over that
grid for this fitted model, not proof that the feature has no relationship in
the population.

## Artifacts and limitations

- `permutation-importance.csv`
- `shap-global-importance.csv`
- `shap-local-example.csv`
- `local-customer-summary.csv`
- `local-customer-contributions.csv`
- `figures/shap-global-importance.png`
- `figures/pdp-*.png`

Explanations describe a selected fitted algorithm on a validation population
that participated in model selection. They do not prove causation, guarantee
stability, or establish fairness. Age and related variables warrant
particular care, and any consequential use requires additional validation,
human review, and applicable legal and policy assessment.
