"""
Train/test splitting module for Eurobank customer churn prediction.

Responsibilities:
1. Validate the input dataset.
2. Separate model features and target.
3. Perform reproducible stratified train/test splitting.
"""

import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_data,
)


TEST_SIZE = 0.20
RANDOM_STATE = 42


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Validate and split the customer dataset.

    Uses stratified sampling to preserve the target distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature-engineered customer dataset containing the target.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.

    ValueError
        If required columns or valid target values are missing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Validate model input features.
    validate_feature_data(X)

    # Validate target.
    if y.isnull().any():
        raise ValueError("Target column contains missing values.")

    if not y.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    if y.nunique() != 2:
        raise ValueError("Target must contain both classes: 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
