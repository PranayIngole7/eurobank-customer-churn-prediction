# Predictive Modeling and Risk Scoring for Bank Customer Churn

A machine learning project focused on predicting bank customer churn, estimating customer-level churn probabilities, identifying potential churn drivers, and supporting data-driven customer retention strategies.

## 1. Project Overview

Customer churn is an important challenge for banking institutions. When customers discontinue their relationship with a bank, the institution may lose revenue, customer relationships, and opportunities for long-term growth.

This project aims to develop a supervised machine learning solution that identifies customers who are more likely to leave the bank.

The solution will explore customer demographics, account characteristics, product usage, and engagement patterns to build predictive models and generate interpretable churn risk scores.

The project is being developed incrementally, with emphasis on data quality, statistical analysis, model evaluation, interpretability, and practical usability.

## 2. Project Objectives

- Analyze customer characteristics and historical churn patterns.
- Perform data cleaning, validation, and exploratory data analysis.
- Engineer meaningful features to improve predictive performance.
- Develop and compare multiple binary classification models.
- Generate customer-level churn probabilities between 0 and 1.
- Optimize classification thresholds based on business requirements.
- Identify important churn-related features.
- Evaluate model performance using appropriate classification metrics.
- Investigate model fairness across relevant customer groups.
- Develop an interactive Streamlit dashboard for churn risk analysis.

## 3. Dataset Overview

The project uses a bank customer dataset containing 10,000 records and 14 columns.

The target variable is:

| Column   | Description                                         |
| -------- | --------------------------------------------------- |
| `Exited` | Customer churn indicator: 1 = churned, 0 = retained |

The dataset includes the following customer attributes:

| Feature           | Description                               |
| ----------------- | ----------------------------------------- |
| `Year`            | Dataset year                              |
| `CustomerId`      | Customer identifier                       |
| `Surname`         | Customer surname                          |
| `CreditScore`     | Customer credit score                     |
| `Geography`       | Customer geographical region              |
| `Gender`          | Customer gender                           |
| `Age`             | Customer age                              |
| `Tenure`          | Number of years associated with the bank  |
| `Balance`         | Customer account balance                  |
| `NumOfProducts`   | Number of bank products used              |
| `HasCrCard`       | Credit card ownership indicator           |
| `IsActiveMember`  | Customer activity indicator               |
| `EstimatedSalary` | Estimated customer salary                 |
| `Exited`          | Target variable indicating customer churn |

### Initial Target Distribution

| Customer Status | Records | Percentage |
| --------------- | ------: | ---------: |
| Retained        |   7,963 |     79.63% |
| Churned         |   2,037 |     20.37% |
| Total           |  10,000 |       100% |

The dataset has a moderate class imbalance, which will be considered during model development and evaluation.

**Dataset note:** The original CSV is maintained locally under `data/raw/` and is excluded from Git. Dataset provenance and redistribution permissions must be verified before making the dataset publicly available.

## 4. Technology Stack

| Category                | Technologies        |
| ----------------------- | ------------------- |
| Programming Language    | Python 3.10+        |
| Data Manipulation       | Pandas, NumPy       |
| Data Visualization      | Matplotlib, Seaborn |
| Machine Learning        | Scikit-learn        |
| Statistical Analysis    | SciPy               |
| Model Persistence       | Joblib              |
| Interactive Dashboard   | Streamlit (planned) |
| Development Environment | JupyterLab          |
| Testing                 | Pytest              |
| Version Control         | Git and GitHub      |

## 5. Machine Learning Methodology

The project follows a structured machine learning workflow covering data preparation, model development, evaluation, explainability, and fairness analysis.

### Completed Workflow

1. Dataset inspection and data quality validation.
2. Exploratory data analysis.
3. Feature engineering.
4. Data preprocessing and encoding.
5. Stratified training, validation, and test splitting.
6. Baseline model development and model comparison.
7. Model improvement and validation.
8. Classification evaluation and threshold optimization.
9. Customer-level churn risk scoring.
10. Model explainability and feature importance.
11. Fairness and comparative analysis across customer groups.

### Upcoming Workflow

12. Model persistence and application integration.
13. Interactive Streamlit dashboard development.
14. Additional testing and documentation.
15. Final packaging and submission.

### Models Evaluated

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

The selected model is Balanced Logistic Regression, with class weighting enabled to address class imbalance.

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

## 6. Feature Engineering

Feature engineering transforms the original customer attributes into additional features that support exploratory analysis and predictive modeling.

The implemented feature engineering includes:

- `HasBalance`: Indicates whether a customer has a non-zero account balance.
- `AgeGroup`: Groups customer ages into meaningful age categories for analysis.

The engineered dataset is used consistently in subsequent preprocessing, model training, evaluation, explainability, and fairness analysis.

## 7. Project Structure

```text
eurobank-customer-churn-prediction/
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
│   ├── local-customer-contributions.csv
│   ├── local-customer-summary.csv
│   ├── model-explainability.md
│   ├── model-improvement.md
│   ├── permutation-importance.csv
│   ├── preprocessing.md
│   ├── threshold-comparison.csv
│   └── figures/
│       ├── confusion-matrix-validation.png
│       ├── fairness-comparison.png
│       ├── feature-coefficients.png
│       ├── permutation-importance.png
│       ├── precision-recall-curve-validation.png
│       ├── roc-curve-validation.png
│       └── threshold-comparison.png
├── scripts/
│   ├── analyze_business_impact.py
│   ├── analyze_model_explainability.py
│   ├── analyze_model_fairness.py
│   ├── compare_thresholds.py
│   ├── evaluate_validation.py
│   ├── explain_customer_predictions.py
│   └── generate_risk_scores.py
├── src/
│   ├── data_quality.py
│   ├── data_split.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── fairness.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── preprocessing.py
│   └── risk_scoring.py
└── tests/
├── test_data_quality.py
├── test_data_split.py
├── test_evaluation.py
├── test_explainability.py
├── test_fairness.py
├── test_feature_engineering.py
├── test_modeling.py
├── test_preprocessing.py
├── test_risk_scoring.py
└── test_validation_split.py
```

The `.venv/` directory is local and excluded from version control.

## 8. Current Development Progress

The project is being developed incrementally through the following phases.

| Phase | Description                                     | Status           |
| ----- | ----------------------------------------------- | ---------------- |
| 1     | Project Setup and Dataset Inspection            | Complete         |
| 2     | Data Cleaning and Quality Checks                | Complete         |
| 3     | Exploratory Data Analysis                       | Complete         |
| 4     | Feature Engineering                             | Complete         |
| 5     | Data Preprocessing                              | Complete         |
| 6     | Train-Test Split and Baseline Model Preparation | Complete         |
| 7     | Model Improvement and Validation                | Complete         |
| 8     | Model Evaluation and Threshold Optimization     | Complete         |
| 9     | Customer Churn Risk Scoring                     | Complete         |
| 10    | Model Explainability                            | Complete         |
| 11    | Fairness and Comparative Analysis               | Final Validation |
| 12    | Model Saving and Application Integration        | Planned          |
| 13    | Streamlit Dashboard Development                 | Planned          |
| 14    | Testing and Documentation                       | Planned          |
| 15    | Final Packaging and Submission                  | Planned          |

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

### Current Phase

Phase 11 — Fairness and Comparative Analysis.

The fairness analysis implementation and report are complete. Final project validation and documentation review are pending.

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

## 12. Project Status

**Current Status:** Phases 1–10 are complete. Phase 11 — Fairness and Comparative Analysis — has completed implementation, testing, and report generation. Final visual and repository validation are pending.

The project has progressed from initial data inspection through model development, evaluation, risk scoring, explainability, and fairness analysis.

The next planned phase is Phase 12 — Model Saving and Application Integration.

This is an evolving machine learning project. Results, model comparisons, and conclusions will be updated as development progresses.

---

**Developed as part of the Unified Mentor Machine Learning Internship.**