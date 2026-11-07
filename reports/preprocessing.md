# Data Preprocessing and Encoding

## 1. Objective

Prepare customer features for machine learning by applying appropriate transformations to numerical, categorical, and binary inputs.

The preprocessing pipeline is designed to provide consistent transformations while preventing data leakage during model training and evaluation.

## 2. Feature Selection

The model uses 12 input features, organized into three groups.

### Numerical features

- `CreditScore`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `EstimatedSalary`

**Transformation:** `StandardScaler`

Standardization places numerical features on a comparable scale by removing the training-data mean and scaling according to the training-data standard deviation.

### Categorical features

- `Geography`
- `Gender`
- `AgeGroup`

**Transformation:** `OneHotEncoder(handle_unknown="ignore")`

One-hot encoding converts categorical values into numerical indicator features. The `handle_unknown="ignore"` setting allows the transformer to process previously unseen categories without raising an encoding error.

### Binary features

- `HasCrCard`
- `IsActiveMember`
- `HasBalance`

**Transformation:** Passed through unchanged.

These features are already represented as binary numerical values and do not require scaling or categorical encoding.

### Excluded columns

| Column       | Reason for exclusion                        |
| ------------ | ------------------------------------------- |
| `Year`       | Constant feature with no useful variation   |
| `CustomerId` | Customer identifier, not a predictive input |
| `Surname`    | Identifier-like, high-cardinality feature   |
| `Exited`     | Prediction target, not an input feature     |

The target column `Exited` is kept separate from the model input features.

## 3. Implementation

The preprocessing implementation is located in `src/preprocessing.py`.

The module provides:

- Feature column definitions.
- Input feature validation.
- A reusable `build_preprocessor()` factory.
- A `ColumnTransformer` combining numerical scaling, categorical encoding, and binary passthrough.

The preprocessor factory returns an unfitted transformer.

During model development, the preprocessing pipeline must be fitted using training data only. Validation and test data are transformed using the already-fitted training pipeline.

This prevents information from validation and test datasets from influencing preprocessing parameters such as numerical means, standard deviations, or categorical encoding.

## 4. Transformation Verification

An 8,000-row diagnostic dataset was used to inspect the expected transformation structure.

| Measurement                     | Result |
| ------------------------------- | -----: |
| Input features                  |     12 |
| Numerical output features       |      6 |
| One-hot encoded output features |     10 |
| Binary output features          |      3 |
| Total transformed features      |     19 |

Additional checks examined:

- Standardized numerical features having approximately zero means.
- Standardized numerical features having approximately unit standard deviations.
- Binary features retaining values of 0 and 1.

This diagnostic transformation was used to verify preprocessing behavior and output dimensions. It should not be confused with the final model-training procedure, where preprocessing is fitted separately within the training pipeline.

## 5. Testing

Preprocessing tests are located in `tests/test_preprocessing.py`.

Test coverage includes:

- Valid and invalid input validation.
- Missing columns and missing values.
- Infinite and non-numeric values.
- Invalid binary values.
- Unfitted and fresh preprocessor instances.
- Feature transformation.
- Preservation of original input data.
- Unknown categorical values.
- Binary feature passthrough.

These tests verify input validation, transformation behavior, and the reusable nature of the preprocessing implementation.

## 6. Outcome

The preprocessing module provides a reusable transformation pipeline for customer churn model development.

It separates numerical scaling, categorical encoding, and binary passthrough while keeping the target and identifier columns outside the model inputs.

The resulting preprocessing pipeline supports consistent feature transformations during training, validation, final evaluation, and application inference.
