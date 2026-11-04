import numpy as np
import pandas as pd
import pytest

from src.risk_scoring import score_customers


class FixedProbabilityModel:
    """Simple fitted model for testing risk scores."""

    classes_ = np.array([0, 1])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities)

    def predict_proba(self, X):
        probabilities = self.probabilities

        return np.column_stack(
            [1 - probabilities, probabilities]
        )


def make_features(size):
    return pd.DataFrame(
        {"CreditScore": range(size)},
        index=range(100, 100 + size),
    )


def test_scores_and_categories_are_calculated_correctly():
    probabilities = [0.10, 0.40, 0.60, 0.85]
    X = make_features(4)

    result = score_customers(
        FixedProbabilityModel(probabilities), X
    )

    assert result["risk_score"].tolist() == [10, 40, 60, 85]
    assert result["risk_category"].tolist() == [
        "Low", "Moderate", "High", "Critical"
    ]


def test_risk_category_boundaries():
    probabilities = [
        0.0, 0.2999, 0.30, 0.4999,
        0.50, 0.6999, 0.70, 1.0,
    ]

    result = score_customers(
        FixedProbabilityModel(probabilities),
        make_features(len(probabilities)),
    )

    assert result["risk_category"].tolist() == [
        "Low", "Low",
        "Moderate", "Moderate",
        "High", "High",
        "Critical", "Critical",
    ]


def test_classification_threshold_is_independent_of_risk_category():
    result = score_customers(
        FixedProbabilityModel([0.62, 0.65, 0.80]),
        make_features(3),
        threshold=0.65,
    )

    assert result["risk_category"].tolist() == [
        "High", "High", "Critical"
    ]
    assert result["churn_prediction"].tolist() == [0, 1, 1]


def test_custom_classification_threshold_is_supported():
    result = score_customers(
        FixedProbabilityModel([0.40, 0.50, 0.60]),
        make_features(3),
        threshold=0.50,
    )

    assert result["churn_prediction"].tolist() == [0, 1, 1]


def test_result_preserves_customer_indexes():
    X = make_features(2)

    result = score_customers(
        FixedProbabilityModel([0.25, 0.75]), X
    )

    assert result.index.tolist() == [100, 101]


def test_result_has_expected_columns():
    result = score_customers(
        FixedProbabilityModel([0.50]),
        make_features(1),
    )

    assert result.columns.tolist() == [
        "churn_probability",
        "risk_score",
        "risk_category",
        "churn_prediction",
    ]


def test_non_dataframe_input_is_rejected():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        score_customers(
            FixedProbabilityModel([0.50]),
            [[100]],
        )


def test_empty_dataframe_is_rejected():
    with pytest.raises(ValueError, match="at least one customer"):
        score_customers(
            FixedProbabilityModel([]),
            pd.DataFrame({"CreditScore": []}),
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_threshold_values_are_rejected(threshold):
    with pytest.raises(ValueError):
        score_customers(
            FixedProbabilityModel([0.50]),
            make_features(1),
            threshold=threshold,
        )


@pytest.mark.parametrize("threshold", [None, "0.65", True])
def test_non_numeric_thresholds_are_rejected(threshold):
    with pytest.raises(TypeError):
        score_customers(
            FixedProbabilityModel([0.50]),
            make_features(1),
            threshold=threshold,
        )


def test_model_without_predict_proba_is_rejected():
    class InvalidModel:
        classes_ = np.array([0, 1])

    with pytest.raises(TypeError, match="predict_proba"):
        score_customers(InvalidModel(), make_features(1))


def test_unfitted_model_is_rejected():
    class UnfittedModel:
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]])

    with pytest.raises(ValueError, match="fitted"):
        score_customers(UnfittedModel(), make_features(1))


def test_model_without_positive_class_is_rejected():
    class InvalidClassModel:
        classes_ = np.array([0, 2])

        def predict_proba(self, X):
            return np.array([[0.5, 0.5]])

    with pytest.raises(ValueError, match="positive class"):
        score_customers(InvalidClassModel(), make_features(1))


@pytest.mark.parametrize(
    "probabilities",
    [
        [np.nan],
        [np.inf],
        [-0.1],
        [1.1],
    ],
)
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError):
        score_customers(
            FixedProbabilityModel(probabilities),
            make_features(1),
        )
