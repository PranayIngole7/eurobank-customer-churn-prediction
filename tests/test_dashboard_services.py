import numpy as np
import pandas as pd
import pytest

from dashboard.services import (
    build_prediction_output,
    prepare_probability_distribution,
    simulate_customer_scenario,
)


class FakePredictionService:
    def __init__(self):
        self.calls = []

    def predict_one(self, customer):
        self.calls.append(dict(customer))
        probability = (
            0.8
            - 0.2 * customer["IsActiveMember"]
            + 0.05 * (customer["NumOfProducts"] - 1)
        )
        return {
            "churn_probability": probability,
            "risk_category": "High" if probability >= 0.5 else "Moderate",
            "churn_prediction": int(probability >= 0.65),
        }


def test_scenario_calls_production_prediction_service_without_mutating_input():
    service = FakePredictionService()
    customer = {
        "CreditScore": 650,
        "Age": 40,
        "Tenure": 4,
        "Balance": 1000.0,
        "NumOfProducts": 1,
        "EstimatedSalary": 50000.0,
        "Geography": "France",
        "Gender": "Female",
        "HasCrCard": 1,
        "IsActiveMember": 0,
    }
    original = dict(customer)

    result = simulate_customer_scenario(
        service,
        customer,
        {"IsActiveMember": 1, "NumOfProducts": 2},
    )

    assert len(service.calls) == 2
    assert service.calls[0] == original
    assert service.calls[1]["IsActiveMember"] == 1
    assert service.calls[1]["NumOfProducts"] == 2
    assert customer == original
    assert result["probability_change"] == pytest.approx(-0.15)
    assert result["absolute_probability_change"] == pytest.approx(0.15)
    assert result["direction"] == "decreased"


def test_scenario_rejects_adjustments_outside_supported_controls():
    with pytest.raises(ValueError, match="only IsActiveMember and NumOfProducts"):
        simulate_customer_scenario(
            FakePredictionService(),
            {"Age": 40},
            {"Age": 41},
        )


def test_probability_distribution_counts_bins_without_identifiers():
    distribution = prepare_probability_distribution(
        [0.0, 0.1, 0.55, 0.99, 1.0],
        bins=5,
    )

    assert list(distribution.columns) == ["Probability range", "Customers"]
    assert distribution["Customers"].sum() == 5
    assert len(distribution) == 5


@pytest.mark.parametrize(
    "values",
    [[], [np.nan], [-0.01], [1.01], [[0.1, 0.9]]],
)
def test_probability_distribution_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        prepare_probability_distribution(values)


def test_prediction_output_excludes_identifiers_and_unknown_columns():
    customers = pd.DataFrame({
        "CreditScore": [650],
        "Geography": ["France"],
        "Gender": ["Female"],
        "Age": [40],
        "Tenure": [4],
        "Balance": [1000.0],
        "NumOfProducts": [1],
        "HasCrCard": [1],
        "IsActiveMember": [1],
        "EstimatedSalary": [50000.0],
        "CustomerId": [123456],
        "Surname": ["Example"],
        "case_note": ["sensitive"],
    })
    predictions = pd.DataFrame({
        "churn_probability": [0.2],
        "risk_score": [20.0],
        "risk_category": ["Low"],
        "churn_prediction": [0],
    })

    result = build_prediction_output(customers, predictions)

    assert len(result) == 1
    assert not {"CustomerId", "Surname", "case_note", "Exited"}.intersection(
        result.columns
    )
    assert result["churn_probability"].tolist() == [0.2]
