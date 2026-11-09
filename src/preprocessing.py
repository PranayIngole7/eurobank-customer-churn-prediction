"""
Data preprocessing module for Eurobank customer churn prediction.

Handles:
1. Numerical feature scaling.
2. Categorical feature encoding.
3. Binary feature passthrough.

The preprocessor is intentionally returned unfitted.
It must be fitted on training data only to prevent data leakage.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "BalanceToSalaryRatio",
    "ProductDensityIndicator",
    "EngagementProductInteraction",
    "AgeTenureInteraction",
]

CATEGORICAL_FEATURES = [
    "Geography",
    "Gender",
    "AgeGroup",
]

BINARY_FEATURES = [
    "HasCrCard",
    "IsActiveMember",
    "HasBalance",
]

FEATURE_COLUMNS = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
    + BINARY_FEATURES
)

TARGET_COLUMN = "Exited"


def validate_feature_data(df: pd.DataFrame) -> None:
    """
    Validate the input feature DataFrame.

    The target column and identifier columns should not be passed
    to this function.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_columns = set(FEATURE_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {sorted(missing_columns)}"
        )

    features = df[FEATURE_COLUMNS]

    if features.isnull().any().any():
        raise ValueError("Feature data contains missing values.")

    for column in NUMERICAL_FEATURES:
        values = pd.to_numeric(
            features[column],
            errors="coerce",
        )

        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(
                f"{column} must contain finite numerical values."
            )

    for column in BINARY_FEATURES:
        if not features[column].isin([0, 1]).all():
            raise ValueError(
                f"{column} must contain only 0 and 1."
            )


def build_preprocessor() -> ColumnTransformer:
    """
    Build a fresh, unfitted preprocessing transformer.

    Numerical features:
        StandardScaler

    Categorical features:
        OneHotEncoder with unknown-category handling

    Binary features:
        Passed through unchanged
    """

    numerical_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "binary",
                "passthrough",
                BINARY_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor
