import pandas as pd
import pytest

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.modeling import (
    build_baseline_models,
    evaluate_model,
)
from src.preprocessing import FEATURE_COLUMNS


@pytest.fixture
def sample_training_data():
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
            "AgeGroup": ["<30", "30-39", "40-49", "50-59", "60+"][index % 5],
            "HasCrCard": index % 2,
            "IsActiveMember": (index + 1) % 2,
            "HasBalance": int(index > 0),
            "Exited": index % 2,
        })

    df = pd.DataFrame(rows)

    X = df[FEATURE_COLUMNS]
    y = df["Exited"]

    return X, y


def test_build_baseline_models_returns_expected_models():
    models = build_baseline_models()

    assert set(models.keys()) == {
        "dummy",
        "logistic_regression",
    }

    assert isinstance(models["dummy"], DummyClassifier)
    assert isinstance(models["logistic_regression"], Pipeline)


def test_models_are_unfitted_initially():
    models = build_baseline_models()

    assert not hasattr(models["dummy"], "classes_")
    assert not hasattr(
        models["logistic_regression"].named_steps["classifier"],
        "classes_",
    )


def test_logistic_regression_uses_expected_configuration():
    models = build_baseline_models()

    classifier = models["logistic_regression"].named_steps["classifier"]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.max_iter == 1000
    assert classifier.random_state == 42


def test_models_can_be_fitted(sample_training_data):
    X, y = sample_training_data
    models = build_baseline_models()

    for model in models.values():
        model.fit(X, y)

        assert hasattr(model, "classes_")


def test_evaluation_returns_expected_metrics(sample_training_data):
    X, y = sample_training_data
    model = build_baseline_models()["logistic_regression"]

    model.fit(X, y)

    results = evaluate_model(model, X, y)

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
    }

    assert set(results.keys()) == expected_metrics

    for metric in expected_metrics - {"confusion_matrix"}:
        assert 0.0 <= results[metric] <= 1.0

    assert len(results["confusion_matrix"]) == 2
    assert all(len(row) == 2 for row in results["confusion_matrix"])


def test_dummy_classifier_has_zero_churn_recall(sample_training_data):
    X, y = sample_training_data

    model = build_baseline_models()["dummy"]
    model.fit(X, y)

    results = evaluate_model(model, X, y)

    assert results["recall"] == 0.0
    assert results["precision"] == 0.0


def test_evaluation_requires_fitted_model(sample_training_data):
    X, y = sample_training_data
    model = build_baseline_models()["dummy"]

    with pytest.raises((AttributeError, ValueError)):
        evaluate_model(model, X, y)


def test_fresh_model_instances_are_independent():
    first = build_baseline_models()
    second = build_baseline_models()

    assert first["dummy"] is not second["dummy"]
    assert first["logistic_regression"] is not second["logistic_regression"]
