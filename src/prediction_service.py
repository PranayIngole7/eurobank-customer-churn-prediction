"""
Application-level prediction service for customer churn.

Responsibilities:
1. Load the persisted model and metadata once.
2. Accept raw customer input.
3. Apply the project's feature engineering.
4. Validate model features.
5. Generate single-customer and batch predictions.
"""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.feature_engineering import engineer_features
from src.model_persistence import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    load_model,
)
from src.preprocessing import (
    FEATURE_COLUMNS,
    validate_feature_data,
)
from src.risk_scoring import score_customers


# Features required from the application before feature engineering.
RAW_REQUIRED_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in {"AgeGroup", "HasBalance"}
]


class PredictionService:
    """
    Reusable service for generating customer churn predictions.

    The persisted model is loaded once during initialization.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ):
        self._model, self._metadata = load_model(
            model_path=model_path,
            metadata_path=metadata_path,
        )

        # Verify that the fitted model expects the feature order
        # declared in its metadata.
        model_feature_columns = getattr(
            self._model,
            "feature_names_in_",
            None,
        )

        if (
            model_feature_columns is not None
            and list(model_feature_columns)
            != self._metadata["feature_columns"]
        ):
            raise ValueError(
                "Model feature order does not match metadata."
            )

        self._threshold = float(self._metadata["threshold"])
        self._feature_columns = tuple(
            self._metadata["feature_columns"]
        )

    @property
    def metadata(self) -> dict:
        """Return a deep copy of the loaded model metadata."""
        return deepcopy(self._metadata)

    def predict_one(self, customer: Mapping) -> dict:
        """
        Generate a prediction for one raw customer record.

        Parameters
        ----------
        customer:
            Mapping containing the customer's raw attributes.

        Returns
        -------
        dict
            Churn probability, risk score, risk category,
            and churn prediction.
        """

        if not isinstance(customer, Mapping):
            raise TypeError(
                "customer must be a mapping of feature names to values."
            )

        customer_data = pd.DataFrame([dict(customer)])

        predictions = self.predict_many(customer_data)

        return predictions.iloc[0].to_dict()

    def predict_many(
        self,
        customers: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions for multiple raw customer records.

        Parameters
        ----------
        customers:
            DataFrame containing raw customer attributes.

        Returns
        -------
        pandas.DataFrame
            Prediction results for every customer.
        """

        if not isinstance(customers, pd.DataFrame):
            raise TypeError(
                "customers must be a pandas DataFrame."
            )

        if customers.empty:
            raise ValueError(
                "customers must contain at least one customer."
            )

        if customers.columns.duplicated().any():
            raise ValueError(
                "Customer input must not contain duplicate column names."
            )

        if "Exited" in customers.columns:
            raise ValueError(
                "Target column 'Exited' must not be supplied for prediction."
            )

        missing_columns = (
            set(RAW_REQUIRED_COLUMNS) - set(customers.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required customer columns: "
                f"{sorted(missing_columns)}"
            )

        # Work on a copy so the caller's DataFrame remains unchanged.
        raw_customers = customers.copy(deep=True)

        # Generate HasBalance and AgeGroup.
        engineered_customers = engineer_features(raw_customers)

        # Select the exact model feature columns and their stored order.
        model_features = engineered_customers[
            list(self._feature_columns)
        ]

        # Validate the complete feature set before inference.
        validate_feature_data(model_features)

        return score_customers(
            self._model,
            model_features,
            threshold=self._threshold,
        )
