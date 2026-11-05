# Model Deployment and Prediction Service

## 1. Overview

This document explains how to train, persist, load, and use the final customer churn prediction model.

Phase 12 introduces a reusable Python prediction service. A web dashboard or REST API is not part of this phase; the Streamlit application is planned for Phase 13.

## 2. Prediction Architecture

The inference workflow is:

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
       |
       v
Feature Column Selection and Validation
       |
       v
Persisted Scikit-learn Pipeline
  - StandardScaler
  - OneHotEncoder
  - Balanced Logistic Regression
       |
       v
Risk Scoring (threshold = 0.65)
       |
       v
Prediction Results
  - churn_probability
  - risk_score
  - risk_category
  - churn_prediction
```

**Important:** The persisted scikit-learn pipeline contains the fitted preprocessor and classifier. Feature engineering is performed separately by `PredictionService` through `engineer_features()`. Therefore, the saved pipeline alone does not accept raw customer records.

## 3. Final Model Training

### Prerequisites

- Python environment activated.
- Project dependencies installed from `requirements.txt`.
- Raw dataset available at `data/raw/european-bank.csv`.

### Run Training

From the project root:

```bash
python -m scripts.train_final_model
```

The training process:

1. Loads the raw customer dataset.
2. Performs feature engineering.
3. Creates the stratified 80/20 training and test split.
4. Trains the selected Balanced Logistic Regression pipeline on the original 8,000-record training partition.
5. Evaluates the model on the untouched 2,000-record test partition.
6. Uses the previously selected classification threshold of 0.65.
7. Saves the model, metadata, and evaluation reports.

The classification threshold was selected using validation data. The original test partition is reserved for final evaluation and is not used to tune the threshold or model.

### Generated Artifacts

| Artifact                                   | Purpose                                     |
| ------------------------------------------ | ------------------------------------------- |
| `models/churn_model.joblib`                | Persisted fitted scikit-learn pipeline      |
| `models/model_metadata.json`               | Model configuration and provenance metadata |
| `reports/final-model-test-evaluation.json` | Machine-readable test evaluation            |
| `reports/final-model-evaluation.md`        | Human-readable test evaluation              |

The generated model files are ignored by Git. After cloning the repository, run the training command to generate them locally before using the prediction service.

## 4. Model Metadata

The metadata file records:

- Metadata schema version
- Model name and algorithm
- Classification threshold
- Expected feature names and their order
- Number of training records
- Random seed
- Target column and positive class
- Model classes
- Python, scikit-learn, and joblib versions
- Model creation timestamp

The prediction service uses the metadata to obtain the configured classification threshold and expected feature order.

Version information is recorded for traceability; it does not constitute an automatic dependency-compatibility check. Use the project's `requirements.txt` when reproducing the environment.

## 5. Single-Customer Prediction

The prediction service loads the persisted model and metadata when the service instance is created.

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
    "EstimatedSalary": 101348.88,
}

result = service.predict_one(customer)

print(result)
```

The result contains:

| Field               | Description                                     |
| ------------------- | ----------------------------------------------- |
| `churn_probability` | Estimated probability of churn, between 0 and 1 |
| `risk_score`        | Churn probability expressed on a 0–100 scale    |
| `risk_category`     | Assigned risk category                          |
| `churn_prediction`  | Binary prediction: 1 for churn, 0 for retained  |

The classification decision uses the threshold stored in model metadata.

## 6. Batch Prediction

The service also accepts a pandas DataFrame containing multiple customers.

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
        "EstimatedSalary": 101348.88,
    },
    {
        "CreditScore": 750,
        "Geography": "Germany",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 0.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 90000.0,
    },
])

results = service.predict_many(customers)

print(results)
```

The returned DataFrame contains the prediction results and preserves the input DataFrame's index.

## 7. Required Raw Input Features

The prediction service expects these ten raw customer features:

| Feature           | Description                               |
| ----------------- | ----------------------------------------- |
| `CreditScore`     | Customer credit score                     |
| `Age`             | Customer age                              |
| `Tenure`          | Years associated with the bank            |
| `Balance`         | Customer account balance                  |
| `NumOfProducts`   | Number of bank products                   |
| `EstimatedSalary` | Estimated customer salary                 |
| `Geography`       | Customer geography                        |
| `Gender`          | Customer gender category from the dataset |
| `HasCrCard`       | Credit card ownership indicator           |
| `IsActiveMember`  | Active membership indicator               |

The service derives `AgeGroup` and `HasBalance` during feature engineering.

Additional columns, such as customer identifiers, may be supplied but are not used for model inference. The target column `Exited` is rejected as an input to prevent target leakage.

## 8. Input Validation

The prediction service validates incoming data before inference.

It checks:

- Input type and non-empty batch data.
- Required feature columns.
- Duplicate column names.
- Presence of the target column.
- Valid age and balance values through feature engineering.
- Missing or invalid feature values.
- Expected feature order.

The fitted OneHotEncoder uses `handle_unknown="ignore"`, allowing previously unseen categorical values to pass through preprocessing without failing solely because a category was absent during training.

Invalid input raises an exception rather than silently producing a prediction.

## 9. Final Test Evaluation

The final model was trained on 8,000 records and evaluated on 2,000 held-out records.

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          | 0.8000 |
| Precision         | 0.5081 |
| Recall            | 0.5405 |
| F1 Score          | 0.5238 |
| ROC-AUC           | 0.7948 |
| Average Precision | 0.5460 |

### Confusion Matrix

|                 | Predicted retained | Predicted churned |
| --------------- | -----------------: | ----------------: |
| Actual retained |              1,380 |               213 |
| Actual churned  |                187 |               220 |

The model detected 220 of the 407 customers who churned in the test partition and missed 187. It also classified 213 retained customers as churn risks.

Accuracy should not be considered in isolation because the dataset is imbalanced. Precision, recall, F1, ROC-AUC, Average Precision, and the confusion matrix provide additional information about model behavior.

See `reports/final-model-evaluation.md` for the complete evaluation.

## 10. Security and Operational Considerations

- Joblib model files use Python pickle-based serialization. Only load model artifacts from trusted sources.
- Do not commit generated model artifacts or customer-level sensitive information to the repository.
- Avoid logging raw customer records or personally identifiable information in a future application.
- Model metadata records dependency versions, but users should maintain a compatible environment using `requirements.txt`.
- This project is an educational and portfolio demonstration. It is not a production-ready banking decision system.
- Predictions should support human review and should not be treated as definitive decisions about customers.

## 11. Current Scope and Next Phase

Phase 12 provides:

- Final model training and held-out test evaluation.
- Model persistence and metadata validation.
- Single-customer prediction.
- Batch prediction.
- Automated tests for training, persistence, and inference.

A user-facing Streamlit dashboard is intentionally deferred to **Phase 13 — Streamlit Dashboard Development**.