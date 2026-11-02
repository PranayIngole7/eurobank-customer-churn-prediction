"""
Data quality validation module.

This module checks the structure, completeness,
uniqueness, and validity of the bank customer
churn dataset.

It does not modify the original dataset.
"""

import pandas as pd


EXPECTED_COLUMNS = [
    "Year",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]

BINARY_COLUMNS = [
    "HasCrCard",
    "IsActiveMember",
    "Exited",
]

CATEGORICAL_RULES = {
    "Geography": {"France", "Germany", "Spain"},
    "Gender": {"Male", "Female"},
}

INTEGER_COLUMNS = [
    "Year",
    "CustomerId",
    "CreditScore",
    "Age",
    "Tenure",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "Exited",
]

def validate_dataset(df: pd.DataFrame) -> list:


    """
    Validate the bank customer churn dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to validate.

    Returns
    -------
    list
        A list of dictionaries containing validation results.
    """

    results = []

    def add_result(check, passed, details):
        results.append({
            "check": check,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        })

    # 1. Dataset structure
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    unexpected_columns = set(df.columns) - set(EXPECTED_COLUMNS)

    add_result(
        "Required columns",
        len(missing_columns) == 0,
        f"Missing: {sorted(missing_columns)}; "
        f"Unexpected: {sorted(unexpected_columns)}",
    )

    if missing_columns:
        return results

    # 2. Missing values
    missing_count = int(df.isnull().sum().sum())

    add_result(
        "Missing values",
        missing_count == 0,
        f"Total missing values: {missing_count}",
    )

    # 3. Duplicate records
    duplicate_count = int(df.duplicated().sum())

    add_result(
        "Duplicate rows",
        duplicate_count == 0,
        f"Duplicate rows: {duplicate_count}",
    )

    # 4. Customer ID uniqueness
    duplicate_ids = int(df["CustomerId"].duplicated().sum())

    add_result(
        "Customer ID uniqueness",
        duplicate_ids == 0,
        f"Duplicate customer IDs: {duplicate_ids}",
    )

    # 5. Target validation
    valid_target = df["Exited"].dropna().isin([0, 1]).all()

    add_result(
        "Target values",
        bool(valid_target),
        f"Observed values: {sorted(df['Exited'].dropna().unique())}",
    )

    # 6. Binary feature validation
    for column in BINARY_COLUMNS:
        valid_binary = df[column].dropna().isin([0, 1]).all()

        add_result(
            f"{column} binary validation",
            bool(valid_binary),
            f"Observed values: {sorted(df[column].dropna().unique())}",
        )

    # 7. Categorical value validation
    for column, allowed_values in CATEGORICAL_RULES.items():
        observed_values = set(df[column].dropna().unique())

        invalid_values = observed_values - allowed_values

        add_result(
            f"{column} categorical validation",
            len(invalid_values) == 0,
            f"Invalid values: {sorted(invalid_values)}",
        )

    # 8. Integer-only feature validation
    for column in INTEGER_COLUMNS:
        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid_integer = (
            numeric_values.notna().all()
            and numeric_values.mod(1).eq(0).all()
        )

        add_result(
            f"{column} integer validation",
            bool(valid_integer),
            f"Data type: {df[column].dtype}",
        )

    # 9. Numerical boundary validation
    numerical_rules = {
        "CreditScore": (300, 850),
        "Age": (18, 100),
        "Tenure": (0, 10),
        "Balance": (0, float("inf")),
        "NumOfProducts": (1, 4),
        "EstimatedSalary": (0, float("inf")),
    }

    for column, (minimum, maximum) in numerical_rules.items():

        valid_range = (
            df[column].dropna().between(minimum, maximum).all()
        )

        add_result(
            f"{column} range validation",
            bool(valid_range),
            f"Observed range: {df[column].min()} to {df[column].max()}",
        )

    # 10. Year consistency
    unique_years = df["Year"].dropna().unique()

    add_result(
        "Year consistency",
        len(unique_years) == 1,
        f"Observed years: {sorted(unique_years.tolist())}",
    )

    return results