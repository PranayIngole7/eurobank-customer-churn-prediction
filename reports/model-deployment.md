# Model Deployment and Prediction Service

## 1. Overview

This document explains how to train, persist, load, and use the final customer churn prediction model.

The project provides:
- A reusable Python prediction service for single-customer and batch inference.
- Persisted model and metadata artifacts.
- A four-page Streamlit dashboard for exploring predictions and model insights.

The dashboard is launched using `streamlit run app.py`. A REST API is not currently implemented.

## 2. Prediction Architecture

The application follows this prediction workflow:

```text
Raw Customer Input
       |
       v
PredictionService
       |
       v
Raw Input Validation
       |
       v
Feature Engineering
  - HasBalance
  - AgeGroup
  - BalanceToSalaryRatio
  - ProductDensityIndicator
  - EngagementProductInteraction
  - AgeTenureInteraction
       |
       v
Model Feature Selection
and Validation
       |
       v
Persisted Scikit-learn Pipeline
  - Numerical and categorical preprocessing
  - Gradient Boosting
       |
       v
Risk Scoring (threshold = 0.30)
       |
       v
Prediction Results
  - churn_probability
  - risk_score
  - risk_category
  - churn_prediction
```

The prediction service performs feature engineering and input validation before passing data to the persisted model.

The saved scikit-learn pipeline contains the fitted preprocessing components and classifier. It does not independently accept raw customer records because the project's feature engineering is performed separately by `PredictionService`.

## 3. Model Training and Persistence

### Prerequisites

- Python environment with project dependencies installed.
- Original dataset at `data/raw/european-bank.csv`.
- Required training and preprocessing modules.

Install the runtime dependencies:

```bash
pip install -r requirements.txt
```

### Train the final model

Run:

```bash
python -m scripts.train_final_model
```

The training script performs the following operations:

1. Loads the original customer dataset.
2. Applies the shared feature engineering process.
3. Splits off the held-out test partition and performs candidate comparison
   and model/threshold selection using training and validation rows only.
4. Refits the selected Gradient Boosting algorithm on all original training
   rows.
5. Evaluates once on the held-out test partition at the validation-selected
   threshold of 0.30.
6. Generates model metadata and persists the fitted pipeline.
7. Saves machine-readable selection and evaluation results and a Markdown
   final evaluation report.

The classification threshold was selected using validation data. The original
test partition is used for final evaluation and a post-selection fairness
audit, but not to tune the model, threshold, or features.

### Generated artifacts

| Artifact                                   | Purpose                                                                       |
| ------------------------------------------ | ----------------------------------------------------------------------------- |
| `models/churn_model.joblib`                | Persisted fitted scikit-learn pipeline                                        |
| `models/model_metadata.json`               | Model configuration, feature information, threshold, and environment metadata |
| `reports/final-model-test-evaluation.json` | Machine-readable final test evaluation                                        |
| `reports/final-model-evaluation.md`        | Human-readable final test evaluation                                          |

Model artifacts are generated locally and excluded from Git.

After cloning the repository, run the training script to generate the model and metadata artifacts before using the prediction service or dashboard.

## 4. Model Metadata

The metadata file describes the persisted model and its expected input configuration.

It contains:

- Metadata schema version.
- Model name and algorithm.
- Classification threshold.
- Expected feature names and order.
- Number of training records.
- Random state.
- Target column and positive class.
- Model class labels.
- Python, scikit-learn, and joblib versions.
- Artifact creation timestamp.

The prediction service uses metadata to retrieve the classification threshold and expected feature order.

The persistence module validates the metadata structure, expected feature
configuration, and model class labels. It also checks that the fitted
estimator's feature names and their order match the metadata before the
model is saved or loaded.

Recorded environment versions provide traceability. They do not constitute automatic compatibility checks for every dependency. Use the project's declared requirements when recreating the environment.

## 5. Single-Customer Prediction

The `PredictionService` class provides a reusable interface for generating predictions from raw customer attributes.

Example:

```python
from src.prediction_service import PredictionService

service = PredictionService()

customer = {
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 125000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 101348.88
}

result = service.predict_one(customer)

print(result)
```

The service returns a dictionary containing:

| Field               | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `churn_probability` | Estimated probability of customer churn, between 0 and 1 |
| `risk_score`        | Churn probability expressed on a 0–100 scale             |
| `risk_category`     | Assigned customer risk category                          |
| `churn_prediction`  | Binary prediction: 0 for retained or 1 for churned       |

The classification threshold is loaded from model metadata rather than hard-coded in the prediction method.

## 6. Batch Prediction

The prediction service also supports batch inference using a pandas DataFrame.

Example:

```python
import pandas as pd

from src.prediction_service import PredictionService

service = PredictionService()

customers = pd.DataFrame([
    {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 125000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 101348.88
    },
    {
        "CreditScore": 720,
        "Geography": "Spain",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 0.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 85000.0
    }
])

results = service.predict_many(customers)

print(results)
```

The method returns a DataFrame containing prediction results for each input customer. The original input index is preserved.

The service loads the persisted model during initialization and reuses it for subsequent predictions.

## 7. Input Features

The prediction service expects ten raw customer attributes.

| Feature           | Description                     |
| ----------------- | ------------------------------- |
| `CreditScore`     | Customer credit score           |
| `Age`             | Customer age                    |
| `Tenure`          | Number of years with the bank   |
| `Balance`         | Customer account balance        |
| `NumOfProducts`   | Number of bank products held    |
| `EstimatedSalary` | Estimated customer salary       |
| `Geography`       | Customer's geography            |
| `Gender`          | Customer gender category        |
| `HasCrCard`       | Credit card ownership indicator |
| `IsActiveMember`  | Active membership indicator     |

The application derives six features:

- `HasBalance`: Indicates whether the customer has a positive balance.
- `AgeGroup`: Groups customers into age categories.
- `BalanceToSalaryRatio`: Balance divided by estimated salary; zero salary
  returns `0.0` as a division-by-zero convention.
- `ProductDensityIndicator`: Product count divided by tenure; zero tenure
  returns `0.0` as a division-by-zero convention.
- `EngagementProductInteraction`: Active-member indicator times product count.
- `AgeTenureInteraction`: Age times tenure.

The resulting model input contains 16 features in the order declared by the
model metadata. The shared `engineer_features` implementation is used for
training and inference. See `reports/feature-engineering.md` for definitions
and limitations.

Additional columns, such as customer identifiers, may be supplied but are not included in model inference. The target column `Exited` must not be supplied.

## 8. Input Validation and Error Handling

The prediction service validates incoming data before inference.

Validation includes:

- Correct input types for single and batch prediction.
- Non-empty batch input.
- No duplicate column names.
- Presence of all required raw customer features.
- Rejection of the target column `Exited`.
- Valid age and balance values during feature engineering.
- Validation of the complete engineered feature set.
- Selection of model features in the expected order.

The feature engineering module rejects missing or non-numeric age and balance values, infinite values, non-integer ages, ages outside the supported 18–100 range, and negative balances.

The preprocessing pipeline also validates the model feature data.

The fitted OneHotEncoder uses `handle_unknown="ignore"`, allowing previously unseen categorical values to be processed without an unknown-category error. This does not guarantee that predictions for unfamiliar categories will be reliable.

Invalid input raises an appropriate exception instead of silently producing a prediction.

## 9. Final Model Test Performance

The selected Gradient Boosting model, using the validation-selected
threshold of 0.30, achieved the following results on the untouched 2,000-row
test partition.

| Metric            | Test Result |
| ----------------- | ----------: |
| Accuracy          |      0.8465 |
| Precision         |      0.6244 |
| Recall            |      0.6167 |
| F1-score          |      0.6205 |
| ROC-AUC           |      0.8615 |
| Average Precision |      0.7032 |

### Confusion Matrix

| Actual / Predicted | Retained (0) | Churned (1) |
| ------------------ | -----------: | ----------: |
| Retained (0)       |        1,442 |         151 |
| Churned (1)        |          156 |         251 |

At the selected threshold of 0.30, the model correctly identified 251 of
the 407 customers who churned in the test partition. It missed 156 actual
churners and generated 151 false-positive churn predictions.

These results illustrate the model's ability to identify potential churn while also showing that its predictions require careful interpretation.

For detailed evaluation methodology, class-level metrics, and limitations, refer to `reports/final-model-evaluation.md`.

## 10. Streamlit Dashboard

The project includes an interactive Streamlit dashboard launched from the repository root:

```bash
streamlit run app.py
```

The application provides four pages:

| Page                       | Purpose                                            |
| -------------------------- | -------------------------------------------------- |
| Overview                   | Presents the project and customer churn overview   |
| Single Customer Prediction | Generates predictions for an individual customer   |
| Batch Prediction           | Supports predictions for multiple customer records |
| Model Insights             | Presents model evaluation and related insights     |

The batch page includes a probability-distribution histogram. The single
customer page includes a what-if scenario simulator that calls the existing
`PredictionService` for both baseline and modified inputs. Scenario changes
are model-based comparisons, not causal estimates.

The dashboard uses the project's application and prediction modules rather than implementing a separate prediction pipeline.

The dashboard is an interactive local application. It is not a separately deployed production service.

## 11. Security and Operational Limitations

### Model artifact security

The persisted model uses joblib serialization, which relies on Python's pickle-based serialization mechanisms.

Only load model artifacts from trusted sources. Loading an untrusted or malicious serialized artifact can execute arbitrary code.

### Customer data protection

- Do not commit customer-level sensitive data or generated model artifacts.
- Avoid logging raw customer records or personally identifiable information.
- Protect access to the original dataset and generated prediction outputs.

### Environment compatibility

Use the declared project dependencies when training or loading the model. Differences in Python and library versions can affect serialized model compatibility.

### Scope limitations

This project is an educational customer churn prediction and model deployment demonstration.

It does not currently provide:

- A REST API.
- Production banking infrastructure.
- Automated production model monitoring.
- A production model retraining pipeline.
- A validated financial decision-making system.

Model predictions should be interpreted as estimates, not guarantees of future customer behavior. Any real-world retention or customer-management decisions would require additional business validation, governance, monitoring, and human oversight.

## 12. Current Capabilities

The project currently provides:

- Reproducible final model training and held-out test evaluation.
- Model persistence and metadata validation.
- Raw customer input validation and feature engineering.
- Single-customer prediction.
- Batch prediction.
- Probability-based customer risk scoring.
- A four-page Streamlit dashboard.
- Automated tests covering training, persistence, preprocessing, prediction, and dashboard-related functionality.

The prediction service and dashboard provide the current application interface. A REST API and production deployment infrastructure are outside the current implementation scope.
