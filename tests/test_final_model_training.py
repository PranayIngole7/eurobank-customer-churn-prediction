import numpy as np
import pandas as pd
import pytest

import src.final_model_training as final_training

from src.preprocessing import FEATURE_COLUMNS


@pytest.fixture
def processed_data():
    rows = []

    for index in range(100):
        rows.append({
            "CreditScore": 600 + index,
            "Age": 25 + (index % 40),
            "Tenure": index % 11,
            "Balance": 1000.0 * index,
            "NumOfProducts": 1 + (index % 2),
            "EstimatedSalary": 30000.0 + index,
            "Geography": ["France", "Germany", "Spain"][index % 3],
            "Gender": ["Male", "Female"][index % 2],
            "AgeGroup": [
                "<30",
                "30-39",
                "40-49",
                "50-59",
                "60+",
            ][index % 5],
            "HasCrCard": index % 2,
            "IsActiveMember": (index + 1) % 2,
            "HasBalance": int(index > 0),
            "Exited": index % 2,
        })

    return pd.DataFrame(rows)


class RecordingModel:
    """A lightweight model that records which rows it is trained on."""

    def fit(self, X, y):
        self.training_indexes = set(X.index)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        probabilities = np.full(len(X), 0.40)

        return np.column_stack((
            1 - probabilities,
            probabilities,
        ))


def test_final_model_uses_original_training_partition_only(
    processed_data,
    monkeypatch,
):
    recording_model = RecordingModel()

    monkeypatch.setattr(
        final_training,
        "build_baseline_models",
        lambda: {
            final_training.SELECTED_MODEL_NAME: recording_model
        },
    )

    X_train, X_test, _, _ = final_training.split_dataset(
        processed_data
    )

    model, results = (
        final_training.train_and_evaluate_final_model(processed_data)
    )

    assert model is recording_model
    assert model.training_indexes == set(X_train.index)
    assert model.training_indexes.isdisjoint(set(X_test.index))

    assert results["training_records"] == 80
    assert results["test_records"] == 20


def test_final_model_preserves_selected_model_and_threshold(
    processed_data,
    monkeypatch,
):
    recording_model = RecordingModel()

    monkeypatch.setattr(
        final_training,
        "build_baseline_models",
        lambda: {
            final_training.SELECTED_MODEL_NAME: recording_model
        },
    )

    _, results = final_training.train_and_evaluate_final_model(
        processed_data
    )

    assert results["model_name"] == (
        "logistic_regression_balanced"
    )
    assert results["algorithm"] == "Balanced Logistic Regression"
    assert results["threshold"] == 0.65


def test_final_model_returns_expected_evaluation_metrics(
    processed_data,
    monkeypatch,
):
    recording_model = RecordingModel()

    monkeypatch.setattr(
        final_training,
        "build_baseline_models",
        lambda: {
            final_training.SELECTED_MODEL_NAME: recording_model
        },
    )

    _, results = final_training.train_and_evaluate_final_model(
        processed_data
    )

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
        "confusion_matrix",
    }

    assert set(results["metrics"]) == expected_metrics

    for metric, value in results["metrics"].items():
        if metric == "confusion_matrix":
            assert len(value) == 2
            assert all(len(row) == 2 for row in value)
        else:
            assert 0.0 <= value <= 1.0

    assert "Retained (0)" in results["classification_report"]
    assert "Churned (1)" in results["classification_report"]


def test_final_model_supports_explicit_threshold(
    processed_data,
    monkeypatch,
):
    recording_model = RecordingModel()

    monkeypatch.setattr(
        final_training,
        "build_baseline_models",
        lambda: {
            final_training.SELECTED_MODEL_NAME: recording_model
        },
    )

    _, results = final_training.train_and_evaluate_final_model(
        processed_data,
        threshold=0.70,
    )

    assert results["threshold"] == 0.70


@pytest.mark.parametrize(
    "threshold",
    [-0.01, 1.01, np.nan, np.inf, -np.inf, "0.65", None, True],
)
def test_final_model_rejects_invalid_thresholds(
    processed_data,
    threshold,
    monkeypatch,
):
    recording_model = RecordingModel()

    monkeypatch.setattr(
        final_training,
        "build_baseline_models",
        lambda: {
            final_training.SELECTED_MODEL_NAME: recording_model
        },
    )

    with pytest.raises(ValueError):
        final_training.train_and_evaluate_final_model(
            processed_data,
            threshold=threshold,
        )
