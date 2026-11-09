import numpy as np
import pandas as pd
import pytest

from src import prediction_service
from src.prediction_service import (
    PredictionService,
    RAW_REQUIRED_COLUMNS,
)
from src.feature_engineering import engineer_features
from src.preprocessing import FEATURE_COLUMNS


class FakeModel:
    """Small fitted-model substitute for prediction-service tests."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        self.last_input = X.copy(deep=True)
        # Customers with a balance receive probability 0.8.
        # Customers without a balance receive probability 0.2.
        positive_probabilities = np.where(
            X["HasBalance"].to_numpy() == 1,
            0.8,
            0.2,
        )

        return np.column_stack(
            (
                1 - positive_probabilities,
                positive_probabilities,
            )
        )


@pytest.fixture
def customer():
    return {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 125000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 101348.88,
    }


@pytest.fixture
def model_metadata():
    return {
        "model_name": "test_model",
        "algorithm": "Fake Logistic Regression",
        "threshold": 0.65,
        "feature_columns": FEATURE_COLUMNS.copy(),
        "training_records": 100,
    }


@pytest.fixture
def service(monkeypatch, model_metadata):
    fake_model = FakeModel()

    def fake_load_model(model_path, metadata_path):
        return fake_model, model_metadata

    monkeypatch.setattr(
        prediction_service,
        "load_model",
        fake_load_model,
    )

    return PredictionService()


def test_service_loads_model_during_initialization(
    monkeypatch,
    model_metadata,
):
    calls = []
    fake_model = FakeModel()

    def fake_load_model(model_path, metadata_path):
        calls.append((model_path, metadata_path))
        return fake_model, model_metadata

    monkeypatch.setattr(
        prediction_service,
        "load_model",
        fake_load_model,
    )

    service = PredictionService(
        model_path="custom-model.joblib",
        metadata_path="custom-metadata.json",
    )

    assert len(calls) == 1
    assert calls[0] == (
        "custom-model.joblib",
        "custom-metadata.json",
    )
    assert service.metadata["model_name"] == "test_model"


def test_predict_one_returns_prediction_dictionary(
    service,
    customer,
):
    result = service.predict_one(customer)

    assert isinstance(result, dict)

    assert result["churn_probability"] == pytest.approx(0.8)
    assert result["risk_score"] == pytest.approx(80.0)
    assert result["risk_category"] == "Critical"
    assert result["churn_prediction"] == 1


def test_predict_one_accepts_mapping(
    service,
    customer,
):
    result = service.predict_one(customer)

    assert "churn_probability" in result


@pytest.mark.parametrize(
    "invalid_customer",
    [
        None,
        [],
        "invalid customer",
        123,
    ],
)
def test_predict_one_rejects_non_mapping_input(
    service,
    invalid_customer,
):
    with pytest.raises(TypeError, match="mapping"):
        service.predict_one(invalid_customer)


def test_predict_many_returns_dataframe(
    service,
    customer,
):
    customers = pd.DataFrame(
        [
            customer,
            {
                **customer,
                "Balance": 0.0,
            },
        ]
    )

    result = service.predict_many(customers)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    assert list(result.columns) == [
        "churn_probability",
        "risk_score",
        "risk_category",
        "churn_prediction",
    ]

    assert result["churn_probability"].tolist() == pytest.approx(
        [0.8, 0.2]
    )

    assert result["churn_prediction"].tolist() == [1, 0]


def test_inference_uses_training_feature_engineering_and_order(
    service,
    customer,
):
    service.predict_one(customer)
    expected = engineer_features(pd.DataFrame([customer]))[
        list(FEATURE_COLUMNS)
    ]

    pd.testing.assert_frame_equal(
        service._model.last_input.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_predict_many_preserves_input_index(
    service,
    customer,
):
    customers = pd.DataFrame(
        [customer, {**customer, "Balance": 0.0}],
        index=[10, 20],
    )

    result = service.predict_many(customers)

    assert result.index.tolist() == [10, 20]


def test_predict_many_rejects_non_dataframe_input(
    service,
    customer,
):
    with pytest.raises(TypeError, match="DataFrame"):
        service.predict_many([customer])


def test_predict_many_rejects_empty_dataframe(service):
    with pytest.raises(ValueError, match="at least one customer"):
        service.predict_many(pd.DataFrame())


def test_predict_many_rejects_missing_columns(
    service,
    customer,
):
    customers = pd.DataFrame(
        [{key: value for key, value in customer.items()
          if key != "CreditScore"}]
    )

    with pytest.raises(ValueError, match="CreditScore"):
        service.predict_many(customers)


def test_predict_many_rejects_target_column(
    service,
    customer,
):
    customer_with_target = {
        **customer,
        "Exited": 1,
    }

    with pytest.raises(ValueError, match="Target column"):
        service.predict_many(
            pd.DataFrame([customer_with_target])
        )


def test_predict_many_rejects_duplicate_columns(
    service,
    customer,
):
    customers = pd.DataFrame([customer])
    customers["Age_duplicate"] = customers["Age"]
    customers.columns = [
        "Age" if column == "Age_duplicate" else column
        for column in customers.columns
    ]

    with pytest.raises(ValueError, match="duplicate"):
        service.predict_many(customers)


def test_predict_many_rejects_invalid_age(
    service,
    customer,
):
    invalid_customer = {
        **customer,
        "Age": 150,
    }

    with pytest.raises(ValueError, match="Age must be between"):
        service.predict_many(pd.DataFrame([invalid_customer]))


def test_predict_many_rejects_negative_balance(
    service,
    customer,
):
    invalid_customer = {
        **customer,
        "Balance": -100.0,
    }

    with pytest.raises(ValueError, match="negative"):
        service.predict_many(pd.DataFrame([invalid_customer]))


def test_predict_many_rejects_missing_values(
    service,
    customer,
):
    invalid_customer = {
        **customer,
        "CreditScore": np.nan,
    }

    with pytest.raises(ValueError, match="missing"):
        service.predict_many(pd.DataFrame([invalid_customer]))


def test_predict_many_accepts_unknown_categories(
    service,
    customer,
):
    unknown_customer = {
        **customer,
        "Geography": "Atlantis",
        "Gender": "Other",
    }

    result = service.predict_many(
        pd.DataFrame([unknown_customer])
    )

    assert len(result) == 1
    assert result.iloc[0]["churn_probability"] == pytest.approx(0.8)


def test_predict_many_does_not_modify_input(
    service,
    customer,
):
    customers = pd.DataFrame([customer])
    original = customers.copy(deep=True)

    service.predict_many(customers)

    pd.testing.assert_frame_equal(customers, original)


def test_metadata_returns_copy(service):
    metadata = service.metadata
    metadata["model_name"] = "modified"

    assert service.metadata["model_name"] == "test_model"


def test_model_loading_failure_propagates(monkeypatch):
    def failing_load_model(model_path, metadata_path):
        raise FileNotFoundError("Model artifact not found.")

    monkeypatch.setattr(
        prediction_service,
        "load_model",
        failing_load_model,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Model artifact not found",
    ):
        PredictionService()


def test_raw_required_columns_match_expected_features():
    assert set(RAW_REQUIRED_COLUMNS) == {
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
        "Geography",
        "Gender",
        "HasCrCard",
        "IsActiveMember",
    }

def test_metadata_returns_deep_copy(service):
    metadata = service.metadata
    metadata["feature_columns"].append("UnexpectedFeature")

    assert "UnexpectedFeature" not in service.metadata["feature_columns"]


def test_service_rejects_model_feature_order_mismatch(
    monkeypatch,
    model_metadata,
):
    fake_model = FakeModel()
    fake_model.feature_names_in_ = np.array(["WrongFeature"])

    def fake_load_model(model_path, metadata_path):
        return fake_model, model_metadata

    monkeypatch.setattr(
        prediction_service,
        "load_model",
        fake_load_model,
    )

    with pytest.raises(ValueError, match="feature order"):
        PredictionService()