"""
Feature engineering module for Eurobank customer churn prediction.

Creates additional customer features without modifying
the original dataset.

Engineered features:
1. HasBalance
2. AgeGroup
"""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Age", "Balance"]

AGE_BINS = [18, 30, 40, 50, 60, 101]

AGE_LABELS = [
    "<30",
    "30-39",
    "40-49",
    "50-59",
    "60+",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from customer attributes.

    Parameters
    ----------
    df : pandas.DataFrame
        Original customer dataset.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataset with engineered features.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.

    ValueError
        If required columns are missing, values are missing,
        or numerical values are invalid.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = df.copy(deep=True)

    for column in REQUIRED_COLUMNS:

        values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"{column} contains missing or non-numeric values."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains infinite values."
            )

        result[column] = values

    if not result["Age"].mod(1).eq(0).all():
        raise ValueError("Age must contain integer values.")

    if ((result["Age"] < 18) | (result["Age"] > 100)).any():
        raise ValueError("Age must be between 18 and 100.")

    if (result["Balance"] < 0).any():
        raise ValueError("Balance cannot contain negative values.")

    # Feature 1: Customer balance status
    result["HasBalance"] = (
        result["Balance"] > 0
    ).astype("int8")

    # Feature 2: Customer age segmentation
    result["AgeGroup"] = pd.cut(
        result["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=False,
        ordered=True,
    )

    if result["AgeGroup"].isna().any():
        raise ValueError(
            "Some age values fall outside the configured age groups."
        )

    return result
