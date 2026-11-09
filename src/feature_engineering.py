"""Feature engineering utilities for Eurobank customer churn prediction.

The project retains the original legacy features (HasBalance and AgeGroup)
while adding the assignment-aligned engineered features used for customer
profiling and explainability.

Canonical feature names:
- BalanceToSalaryRatio
- ProductDensityIndicator
- EngagementProductInteraction
- AgeTenureInteraction

Compatibility aliases retained for callers that already use the older names:
- BalanceToSalary
- ProductDensity
- EngagementProduct
- AgeTenure

Zero-handling policy:
- EstimatedSalary == 0 -> BalanceToSalaryRatio = 0.0
- Tenure == 0 -> ProductDensityIndicator = 0.0

These are explicit domain-safe guards, not silent replacements of invalid data.
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


def _coerce_numeric(series: pd.Series, column_name: str) -> pd.Series:
    """Convert a column to numeric values or raise a clear validation error."""
    values = pd.to_numeric(series, errors="coerce")

    if values.isna().any():
        raise ValueError(
            f"{column_name} contains missing or non-numeric values."
        )

    if not np.isfinite(values).all():
        raise ValueError(f"{column_name} contains infinite values.")

    return values


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from raw customer attributes.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw customer dataset containing at least Age and Balance.

    Returns
    -------
    pandas.DataFrame
        A copy of the input dataset with original and engineered features.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If required values are missing, invalid, non-finite, or outside the
        supported domain.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = df.copy(deep=True)

    result["Age"] = _coerce_numeric(result["Age"], "Age")
    result["Balance"] = _coerce_numeric(result["Balance"], "Balance")

    products = None
    if "NumOfProducts" in result.columns:
        products = _coerce_numeric(
            result["NumOfProducts"],
            "NumOfProducts",
        )
        if not products.mod(1).eq(0).all():
            raise ValueError("NumOfProducts must contain integer values.")
        if ((products < 1) | (products > 4)).any():
            raise ValueError("NumOfProducts must be between 1 and 4.")

    if not result["Age"].mod(1).eq(0).all():
        raise ValueError("Age must contain integer values.")

    if ((result["Age"] < 18) | (result["Age"] > 100)).any():
        raise ValueError("Age must be between 18 and 100.")

    if (result["Balance"] < 0).any():
        raise ValueError("Balance cannot contain negative values.")

    result["HasBalance"] = (result["Balance"] > 0).astype("int8")

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

    if "EstimatedSalary" in result.columns:
        salary = _coerce_numeric(result["EstimatedSalary"], "EstimatedSalary")

        if (salary < 0).any():
            raise ValueError(
                "EstimatedSalary cannot contain negative values."
            )

        ratio = pd.Series(0.0, index=result.index, dtype=float)
        non_zero_salary = salary > 0
        ratio.loc[non_zero_salary] = (
            result.loc[non_zero_salary, "Balance"]
            / salary.loc[non_zero_salary]
        )

        result["BalanceToSalaryRatio"] = ratio.astype(float)
        result["BalanceToSalary"] = result["BalanceToSalaryRatio"]
    else:
        result["BalanceToSalaryRatio"] = 0.0
        result["BalanceToSalary"] = 0.0

    if "Tenure" in result.columns:
        tenure = _coerce_numeric(result["Tenure"], "Tenure")

        if not tenure.mod(1).eq(0).all():
            raise ValueError("Tenure must contain integer values.")
        if ((tenure < 0) | (tenure > 10)).any():
            raise ValueError("Tenure must be between 0 and 10.")

        product_density = pd.Series(0.0, index=result.index, dtype=float)

        if products is not None:
            valid_tenure = tenure > 0
            product_density.loc[valid_tenure] = (
                products.loc[valid_tenure] / tenure.loc[valid_tenure]
            )

        result["ProductDensityIndicator"] = product_density.astype(float)
        result["ProductDensity"] = result["ProductDensityIndicator"]
    else:
        result["ProductDensityIndicator"] = 0.0
        result["ProductDensity"] = 0.0

    if "IsActiveMember" in result.columns:
        active_member = _coerce_numeric(
            result["IsActiveMember"],
            "IsActiveMember",
        )

        if not active_member.isin([0, 1]).all():
            raise ValueError("IsActiveMember must contain only 0 and 1.")

        if products is not None:
            engagement_interaction = active_member * products
        else:
            engagement_interaction = active_member * 0

        result["EngagementProductInteraction"] = (
            engagement_interaction.astype(float)
        )
        result["EngagementProduct"] = result["EngagementProductInteraction"]
    else:
        result["EngagementProductInteraction"] = 0.0
        result["EngagementProduct"] = 0.0

    if "Tenure" in result.columns:
        result["AgeTenureInteraction"] = result["Age"] * result["Tenure"]
        result["AgeTenure"] = result["AgeTenureInteraction"]
    else:
        result["AgeTenureInteraction"] = 0.0
        result["AgeTenure"] = 0.0

    return result
