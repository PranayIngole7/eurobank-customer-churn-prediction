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

The project follows a structured machine learning workflow.

1. Dataset inspection and data quality validation.
2. Exploratory data analysis.
3. Feature engineering.
4. Data preprocessing and encoding.
5. Stratified training and testing split.
6. Baseline model development.
7. Model comparison and evaluation.
8. Classification threshold optimization.
9. Customer-level churn risk scoring.
10. Model explainability and feature importance.
11. Fairness and comparative analysis.
12. Model persistence and application integration.
13. Interactive dashboard development.
14. Testing and documentation.

### Planned Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost (optional, subject to evaluation and project requirements)

### Evaluation Metrics

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Specificity
- Brier Score and probability calibration

Model selection will consider both predictive performance and the practical cost of false positives and false negatives.

## 6. Feature Engineering

The following candidate features will be investigated:

- Balance-to-salary ratio.
- Product density indicator.
- Engagement-product interaction.
- Age-tenure interaction.

Their usefulness will be evaluated experimentally rather than assumed.

## 7. Project Structure

```text
eurobank-customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│   └── data_quality.py
│
├── models/
│
├── reports/
│
├── tests/
│   └── test_data_quality.py
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/
```

The `.venv/` directory is local and excluded from version control.

## 8. Current Development Progress

The project is being developed through the following phases:

| Phase | Description                                 | Status   |
| ----- | ------------------------------------------- | -------- |
| 1     | Project Setup and Dataset Inspection        | Complete |
| 2     | Data Cleaning and Quality Checks            | Complete |
| 3     | Exploratory Data Analysis                   | Complete |
| 4     | Feature Engineering                         | Upcoming |
| 5     | Data Preprocessing                          | Planned  |
| 6     | Train-Test Split and Cross-Validation       | Planned  |
| 7     | Model Development                           | Planned  |
| 8     | Model Evaluation and Threshold Optimization | Planned  |
| 9     | Churn Risk Scoring                          | Planned  |
| 10    | Model Explainability                        | Planned  |
| 11    | Fairness and Comparative Analysis           | Planned  |
| 12    | Model Saving and Application Integration    | Planned  |
| 13    | Streamlit Dashboard Development             | Planned  |
| 14    | Testing and Documentation                   | Planned  |
| 15    | Final Packaging and Submission              | Planned  |

### Completed Work

- Established the Python development environment.
- Created the project directory structure.
- Inspected the original customer dataset.
- Implemented automated dataset quality validation.
- Added unit tests for data validation.
- Performed univariate, bivariate, and multivariate exploratory analysis.
- Investigated numerical feature correlations with customer churn.
- Documented initial findings and modeling considerations.

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

**Current Status:** Foundation, data quality validation, and exploratory data analysis completed. Feature engineering is the next development phase.

This is an evolving machine learning project. Results, model comparisons, and conclusions will be updated as development progresses.

---

**Developed as part of the Unified Mentor Machine Learning Internship.**