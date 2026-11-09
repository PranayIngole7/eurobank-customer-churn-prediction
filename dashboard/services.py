"""Shared dashboard services."""

import numpy as np
import pandas as pd
import streamlit as st

from src.prediction_service import PredictionService, RAW_REQUIRED_COLUMNS


def simulate_customer_scenario(prediction_service, base_customer, adjustments):
    """Return production-service predictions for a customer scenario."""
    if not hasattr(prediction_service, "predict_one"):
        raise TypeError("prediction_service must support predict_one().")

    if not isinstance(base_customer, dict):
        raise TypeError("base_customer must be a dictionary of feature values.")

    if not isinstance(adjustments, dict):
        raise TypeError("adjustments must be a dictionary of feature updates.")

    allowed_adjustments = {"IsActiveMember", "NumOfProducts"}
    unexpected_adjustments = set(adjustments) - allowed_adjustments
    if unexpected_adjustments:
        raise ValueError(
            "Scenario adjustments may change only IsActiveMember and "
            "NumOfProducts."
        )

    updated_customer = dict(base_customer)
    updated_customer.update(adjustments)

    original_prediction = prediction_service.predict_one(base_customer)
    adjusted_prediction = prediction_service.predict_one(updated_customer)

    probability_change = float(
        adjusted_prediction["churn_probability"]
        - original_prediction["churn_probability"]
    )
    return {
        "original_prediction": original_prediction,
        "adjusted_prediction": adjusted_prediction,
        "probability_change": probability_change,
        "absolute_probability_change": abs(probability_change),
        "direction": (
            "increased"
            if probability_change > 0
            else "decreased"
            if probability_change < 0
            else "unchanged"
        ),
    }


def prepare_probability_distribution(probabilities, bins=10):
    """Build identifier-free histogram counts for predicted probabilities."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty 1D sequence.")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must contain only finite values.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be between 0 and 1.")

    counts, edges = np.histogram(values, bins=np.linspace(0.0, 1.0, bins + 1))
    labels = [
        f"{edges[index]:.1f}-{edges[index + 1]:.1f}"
        for index in range(bins)
    ]
    return pd.DataFrame({
        "Probability range": labels,
        "Customers": counts.astype(int),
    })


def build_prediction_output(customers, predictions):
    """Combine only required raw features and predictions for safe export."""
    if not isinstance(customers, pd.DataFrame):
        raise TypeError("customers must be a pandas DataFrame.")
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")
    if len(customers) != len(predictions):
        raise ValueError("Customer and prediction row counts must match.")

    missing_columns = set(RAW_REQUIRED_COLUMNS) - set(customers.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required customer columns: {sorted(missing_columns)}"
        )

    feature_columns = list(RAW_REQUIRED_COLUMNS)
    return pd.concat(
        [
            customers[feature_columns].reset_index(drop=True),
            predictions.reset_index(drop=True),
        ],
        axis=1,
    )


@st.cache_resource
def get_prediction_service():
    """Load and reuse the persisted prediction service."""
    return PredictionService()