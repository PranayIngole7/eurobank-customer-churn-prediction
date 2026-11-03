# Data Preprocessing and Encoding

## Objective

Prepare engineered customer features for machine learning by applying
appropriate transformations to numerical, categorical, and binary features.

## Feature Selection

### Numerical features
- CreditScore
- Age
- Tenure
- Balance
- NumOfProducts
- EstimatedSalary

Transformation: StandardScaler.

### Categorical features
- Geography
- Gender
- AgeGroup

Transformation: OneHotEncoder with handle_unknown="ignore".

### Binary features
- HasCrCard
- IsActiveMember
- HasBalance

Transformation: Passed through unchanged.

### Excluded columns
- Year: constant feature.
- CustomerId: identifier.
- Surname: identifier-like, high-cardinality feature.
- Exited: prediction target.

## Implementation

Created `src/preprocessing.py`.

The module provides:
- Feature column definitions.
- Input feature validation.
- A reusable `build_preprocessor()` factory.
- A ColumnTransformer combining numerical scaling,
  categorical encoding, and binary passthrough.

The preprocessor is returned unfitted. In Phase 6, it will be fitted
using training data only to prevent data leakage.

## Transformation Verification

A diagnostic sample containing 8,000 rows was used to inspect
the transformation.

Results:
- Input features: 12
- Numerical output features: 6
- One-hot encoded output features: 10
- Binary output features: 3
- Total transformed features: 19
- Numerical means: approximately 0
- Numerical standard deviations: 1
- Binary output values: 0 and 1

This diagnostic fit is not the final model-training procedure.
The formal train/test split and leakage-safe fitting will be handled
in Phase 6.

## Testing

Created `tests/test_preprocessing.py`.

Coverage includes:
- Valid and invalid input validation.
- Missing columns and missing values.
- Infinite and non-numeric values.
- Invalid binary values.
- Unfitted and fresh preprocessor instances.
- Feature transformation.
- Original input preservation.
- Unknown categorical values.
- Binary feature passthrough.

Final project test result: 45 passed, 0 failures, 0 warnings.

Four deliberate invalid-data scenarios were also tested and rejected
by the validation function.

## Outcome

Data preprocessing and encoding implementation completed.

Files added:
- `src/preprocessing.py`
- `tests/test_preprocessing.py`
- `reports/phase-5-preprocessing.md`
