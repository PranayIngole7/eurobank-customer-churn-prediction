# Predictive Modeling and Risk Scoring for Bank Customer Churn

A machine learning project focused on predicting customer churn in the banking sector using classification models, customer risk scoring, model explainability, and fairness analysis.

## 1. Project Overview

Customer churn is an important challenge for financial institutions because retaining existing customers is often more cost-effective than acquiring new ones.

This project develops a machine learning solution to identify customers who are likely to leave a bank. It uses historical customer information to train classification models, estimate churn probabilities, and assign customer risk categories.

The project follows a structured machine learning workflow, from data inspection and exploratory analysis to model development, evaluation, explainability, fairness analysis, model persistence, and application integration.

The final selected model is Balanced Logistic Regression, with a reusable Python prediction service supporting single-customer and batch predictions.

## 2. Project Objectives

- Understand customer characteristics and churn patterns through exploratory data analysis.
- Perform data cleaning, validation, and preprocessing.
- Engineer meaningful features for predictive modeling.
- Develop and compare multiple classification algorithms.
- Address class imbalance during model training.
- Evaluate models using appropriate classification metrics.
- Optimize the classification threshold using validation data.
- Generate customer-level churn probabilities and risk categories.
- Explain model predictions using feature importance and local explanations.
- Evaluate model fairness across demographic groups.
- Persist the final trained model and its metadata.
- Implement reusable single-customer and batch prediction capabilities.
- Develop an interactive Streamlit dashboard in a future phase.

## 3. Dataset Overview

The project uses a European bank customer churn dataset containing 10,000 customer records and 14 original columns.

### Dataset Information

| Attribute         | Description                    |
| ----------------- | ------------------------------ |
| Total Records     | 10,000                         |
| Original Features | 13                             |
| Target Variable   | `Exited`                       |
| Problem Type      | Binary Classification          |
| Domain            | Banking and Customer Analytics |

### Original Dataset Features

| Feature           | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
| `Year`            | Year associated with the record                               |
| `CustomerId`      | Unique customer identifier                                    |
| `Surname`         | Customer surname                                              |
| `CreditScore`     | Customer credit score                                         |
| `Geography`       | Customer geographical location                                |
| `Gender`          | Customer gender                                               |
| `Age`             | Customer age                                                  |
| `Tenure`          | Number of years associated with the bank                      |
| `Balance`         | Customer account balance                                      |
| `NumOfProducts`   | Number of bank products owned                                 |
| `HasCrCard`       | Credit card ownership indicator                               |
| `IsActiveMember`  | Active membership indicator                                   |
| `EstimatedSalary` | Estimated customer salary                                     |
| `Exited`          | Target variable indicating whether the customer left the bank |

The identifier and administrative columns `Year`, `CustomerId`, and `Surname` are excluded from model training.

### Initial Target Distribution

| Customer Status |  Count | Percentage |
| --------------- | -----: | ---------: |
| Retained (0)    |  7,963 |     79.63% |
| Churned (1)     |  2,037 |     20.37% |
| Total           | 10,000 |       100% |

The dataset has an imbalanced target distribution, making appropriate evaluation metrics and class imbalance handling important.

## 4. Technology Stack

| Category             | Technologies        |
| -------------------- | ------------------- |
| Programming Language | Python 3.10+        |
| Data Manipulation    | Pandas, NumPy       |
| Data Visualization   | Matplotlib, Seaborn |
| Machine Learning     | Scikit-learn        |
| Statistical Analysis | SciPy               |
| Model Persistence    | Joblib              |
| Notebook Environment | JupyterLab          |
| Testing              | Pytest              |
| Version Control      | Git and GitHub      |
| Future Dashboard     | Streamlit           |

## 5. Machine Learning Methodology

The project follows a structured machine learning lifecycle.

### Completed Workflow

1. Project setup and dataset inspection.
2. Data cleaning and quality checks.
3. Exploratory Data Analysis.
4. Feature engineering.
5. Data preprocessing.
6. Train-test split and baseline model preparation.
7. Model improvement and validation.
8. Model evaluation and threshold optimization.
9. Customer churn risk scoring.
10. Model explainability.
11. Fairness and comparative analysis.
12. Model persistence and application integration.

### Upcoming Workflow

13. Interactive Streamlit dashboard development.
14. Additional testing and documentation.
15. Final packaging and submission.

### Models Evaluated

The following classification algorithms were evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

The selected model is **Balanced Logistic Regression**, with class weighting enabled to address class imbalance.

### Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision (PR-AUC)
- Confusion Matrix
- Specificity
- Brier Score and probability calibration

Model selection considers predictive performance, class imbalance, and the practical consequences of false positives and false negatives.

The final classification threshold of 0.65 was selected using validation data and retained for final test evaluation.

## 6. Feature Engineering

Feature engineering transforms the original customer attributes into additional features that support exploratory analysis and predictive modeling.

The implemented feature engineering includes:

- `HasBalance`: Indicates whether a customer has a non-zero account balance.
- `AgeGroup`: Groups customer ages into meaningful age categories for analysis.

The engineered dataset is used consistently in preprocessing, model training, evaluation, explainability, and fairness analysis.

During application inference, the `PredictionService` applies the same feature engineering logic before passing customer records to the persisted scikit-learn pipeline.

## 7. Project Structure

```text
eurobank-customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── european-bank.csv
│   └── processed/
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── reports/
│   ├── baseline-model.md
│   ├── business-impact-analysis.md
│   ├── classification-report-validation.json
│   ├── customer-risk-scores.csv
│   ├── customer-risk-scoring.md
│   ├── evaluation-report.md
│   ├── fairness-analysis.md
│   ├── fairness-group-metrics.csv
│   ├── fairness-summary.json
│   ├── feature-coefficients.csv
│   ├── final-model-evaluation.md
│   ├── final-model-test-evaluation.json
│   ├── local-customer-contributions.csv
│   ├── local-customer-summary.csv
│   ├── model-deployment.md
│   ├── model-explainability.md
│   ├── model-improvement.md
│   ├── permutation-importance.csv
│   ├── preprocessing.md
│   ├── threshold-comparison.csv
│   └── figures/
│
├── scripts/
│   ├── analyze_business_impact.py
│   ├── analyze_model_explainability.py
│   ├── analyze_model_fairness.py
│   ├── compare_thresholds.py
│   ├── evaluate_validation.py
│   ├── explain_customer_predictions.py
│   ├── generate_risk_scores.py
│   └── train_final_model.py
│
├── src/
│   ├── data_quality.py
│   ├── data_split.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── fairness.py
│   ├── feature_engineering.py
│   ├── final_model_training.py
│   ├── modeling.py
│   ├── model_persistence.py
│   ├── prediction_service.py
│   ├── preprocessing.py
│   └── risk_scoring.py
│
├── tests/
│   ├── test_data_quality.py
│   ├── test_data_split.py
│   ├── test_evaluation.py
│   ├── test_explainability.py
│   ├── test_fairness.py
│   ├── test_feature_engineering.py
│   ├── test_final_model_training.py
│   ├── test_model_persistence.py
│   ├── test_modeling.py
│   ├── test_prediction_service.py
│   ├── test_preprocessing.py
│   ├── test_risk_scoring.py
│   └── test_validation_split.py
│
├── models/
│   └── .gitkeep
│
├── .gitignore
├── requirements.txt
└── README.md
```

The `.venv/` directory and generated model artifacts are local and excluded from version control.

The `models/` directory contains locally generated model artifacts after final model training:

- `churn_model.joblib`
- `model_metadata.json`

See `reports/model-deployment.md` for detailed model training, persistence, and prediction-service instructions.

## 8. Current Development Progress

The project is being developed incrementally through 15 phases.

| Phase | Description                                     | Status   |
| ----- | ----------------------------------------------- | -------- |
| 1     | Project Setup and Dataset Inspection            | Complete |
| 2     | Data Cleaning and Quality Checks                | Complete |
| 3     | Exploratory Data Analysis                       | Complete |
| 4     | Feature Engineering                             | Complete |
| 5     | Data Preprocessing                              | Complete |
| 6     | Train-Test Split and Baseline Model Preparation | Complete |
| 7     | Model Improvement and Validation                | Complete |
| 8     | Model Evaluation and Threshold Optimization     | Complete |
| 9     | Customer Churn Risk Scoring                     | Complete |
| 10    | Model Explainability                            | Complete |
| 11    | Fairness and Comparative Analysis               | Complete |
| 12    | Model Persistence and Application Integration   | Complete |
| 13    | Streamlit Dashboard Development                 | Planned  |
| 14    | Testing and Documentation                       | Planned  |
| 15    | Final Packaging and Submission                  | Planned  |

### Completed Work

- Established the Python development environment and project structure.
- Inspected and validated the original customer dataset.
- Implemented automated data quality checks and unit tests.
- Performed exploratory data analysis and documented customer churn patterns.
- Implemented feature engineering and preprocessing pipelines.
- Established stratified training, validation, and test datasets.
- Developed and compared classification models.
- Improved model performance using class balancing and validation experiments.
- Evaluated classification performance using multiple metrics and visualizations.
- Selected a classification threshold based on validation performance.
- Generated customer-level churn risk scores.
- Implemented model explainability using feature coefficients, permutation importance, and local explanations.
- Implemented group-wise fairness metrics across Gender, Geography, and AgeGroup.
- Evaluated demographic parity, equal opportunity, equalized odds, and disparate impact.
- Calculated Wilson score confidence intervals for selected group-level rates.
- Generated fairness analysis CSV, JSON, and visualization artifacts.
- Documented fairness findings, limitations, and potential improvement experiments.
- Trained the final Balanced Logistic Regression model using the original training partition.
- Evaluated the final model on the untouched original test partition.
- Persisted the fitted model using Joblib.
- Implemented model metadata generation and validation.
- Implemented a reusable prediction service for single-customer inference.
- Implemented batch prediction using pandas DataFrames.
- Added automated tests for final model training, model persistence, and prediction-service behavior.
- Documented model deployment, inference architecture, and operational considerations.

### Current Phase

**Phase 12 — Model Persistence and Application Integration: Complete.**

The final model has been trained and evaluated. Model persistence, metadata validation, and single-customer and batch prediction capabilities are implemented and tested.

The model deployment guide is available at:

`reports/model-deployment.md`

**Next Phase:** Phase 13 — Streamlit Dashboard Development.

## 9. Getting Started

### Prerequisites

- Python 3.10 or later.
- Git.
- pip.

### Clone the Repository

```bash
git clone https://github.com/PranayIngole7/eurobank-customer-churn-prediction.git

cd eurobank-customer-churn-prediction
```

### Create a Virtual Environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Dataset Setup

Place the authorized dataset file at:

```text
data/raw/european-bank.csv
```

The dataset is not included in this repository.

### Run Data Quality Tests

```bash
python -m pytest -v
```

### Train and Persist the Final Model

After placing the dataset and installing dependencies, run:

```bash
python -m scripts.train_final_model
```

This command:

1. Loads the raw customer dataset.
2. Performs feature engineering.
3. Trains the selected Balanced Logistic Regression model.
4. Evaluates it on the held-out test partition.
5. Saves the fitted model and metadata.
6. Generates the final model evaluation reports.

Generated model artifacts are excluded from Git. Run this command locally before using the prediction service.

### Run the Prediction Service

The prediction service supports single-customer and batch inference.

Single-customer example:

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

prediction = service.predict_one(customer)

print(prediction)
```

For batch inference, provide a pandas DataFrame to:

```python
service.predict_many(customers)
```

Refer to `reports/model-deployment.md` for complete examples and input requirements.

### Launch JupyterLab

```bash
jupyter lab
```

Open the notebook:

```text
notebooks/01_exploratory_data_analysis.ipynb
```

## 10. Expected Deliverables

The completed project is expected to include:

- A validated and documented data preparation workflow.
- Exploratory data analysis and feature engineering.
- Comparative machine learning models.
- A churn prediction pipeline producing customer-level probabilities.
- Model explainability and fairness analysis.
- A persisted final model with metadata.
- A reusable Python prediction service supporting single-customer and batch inference.
- An interactive Streamlit dashboard.
- A research paper documenting the methodology and findings.
- An executive summary presenting relevant insights and recommendations.

## 11. Important Considerations

- Exploratory associations do not establish causation.
- Model performance must be evaluated on unseen data.
- Classification thresholds should reflect business requirements.
- Probability estimates should be assessed for calibration.
- Fairness analysis is necessary when evaluating differences across customer groups.
- Dataset provenance, licensing, and generalizability must be documented before drawing conclusions about real-world banking populations.
- The final test partition must not be reused for model or threshold tuning.
- Joblib model artifacts use pickle-based serialization and must only be loaded from trusted sources.
- Generated model artifacts are excluded from version control.
- This project is an educational and portfolio demonstration, not a production-ready banking decision system.

## 12. Project Status

**Current Status:** Phases 1–12 are complete.

Phase 12 implemented final model training, held-out test evaluation, model persistence, metadata validation, and a reusable prediction service supporting single-customer and batch predictions.

### Final Model Test Results

The final Balanced Logistic Regression model was trained on 8,000 records and evaluated on 2,000 held-out records using the previously selected classification threshold of 0.65.

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          | 0.8000 |
| Precision         | 0.5081 |
| Recall            | 0.5405 |
| F1-score          | 0.5238 |
| ROC-AUC           | 0.7948 |
| Average Precision | 0.5460 |

Detailed results are available in `reports/final-model-evaluation.md`.

Model training, persistence, and inference instructions are documented in `reports/model-deployment.md`.

**Next Phase:** Phase 13 — Streamlit Dashboard Development.

This is an evolving machine learning project. Results, model comparisons, and conclusions will be updated as development progresses.

---

**Developed as part of the Unified Mentor Machine Learning Internship.**