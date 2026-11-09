import numpy as np
import pandas as pd
import pytest

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.modeling import (
    build_baseline_models,
    build_candidate_models,
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
            "BalanceToSalaryRatio": (1000.0 * index) / (30000.0 + index),
            "ProductDensityIndicator": (
                (1 + (index % 2)) / (index % 11)
                if index % 11 > 0 else 0.0
            ),
            "EngagementProductInteraction": (
                (index + 1) % 2
            ) * (1 + (index % 2)),
            "AgeTenureInteraction": (25 + (index % 40)) * (index % 11),
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
        "logistic_regression_balanced",
    }

    assert isinstance(models["dummy"], DummyClassifier)
    assert isinstance(models["logistic_regression"], Pipeline)
    assert isinstance(models["logistic_regression_balanced"], Pipeline)


def test_candidate_models_include_all_required_algorithms():
    models = build_candidate_models()

    assert set(models) == {
        "dummy",
        "logistic_regression",
        "logistic_regression_balanced",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    }

    assert isinstance(
        models["decision_tree"].named_steps["classifier"],
        DecisionTreeClassifier,
    )
    assert isinstance(
        models["random_forest"].named_steps["classifier"],
        RandomForestClassifier,
    )
    assert isinstance(
        models["gradient_boosting"].named_steps["classifier"],
        GradientBoostingClassifier,
    )
    for model_name in (
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ):
        assert isinstance(models[model_name], Pipeline)
        assert models[model_name].named_steps["preprocessor"] is not None
        assert models[model_name].named_steps["classifier"].random_state == 42


def test_models_are_unfitted_initially():
    models = build_baseline_models()

    for model in models.values():
        assert not hasattr(model, "classes_")


def test_logistic_regression_uses_expected_configuration():
    models = build_baseline_models()

    classifier = models["logistic_regression"].named_steps["classifier"]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.max_iter == 1000
    assert classifier.random_state == 42
    assert classifier.class_weight is None


def test_balanced_logistic_regression_uses_expected_configuration():
    models = build_baseline_models()

    classifier = (
        models["logistic_regression_balanced"]
        .named_steps["classifier"]
    )

    assert isinstance(classifier, LogisticRegression)
    assert classifier.max_iter == 1000
    assert classifier.random_state == 42
    assert classifier.class_weight == "balanced"


def test_models_can_be_fitted(sample_training_data):
    X, y = sample_training_data
    models = build_baseline_models()

    for model in models.values():
        model.fit(X, y)

        assert hasattr(model, "classes_")


def test_candidate_models_fit_and_predict_probabilities(sample_training_data):
    X, y = sample_training_data
    models = build_candidate_models()

    for model in models.values():
        model.fit(X, y)
        probabilities = model.predict_proba(X.iloc[:5])

        assert probabilities.shape == (5, 2)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


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
        "average_precision",
        "confusion_matrix",
    }

    assert set(results.keys()) == expected_metrics

    for metric in expected_metrics - {"confusion_matrix"}:
        assert 0.0 <= results[metric] <= 1.0

    assert len(results["confusion_matrix"]) == 2
    assert all(len(row) == 2 for row in results["confusion_matrix"])


def test_custom_threshold_changes_predictions():
    class FixedProbabilityModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            churn_probabilities = np.array([0.20, 0.40, 0.60, 0.80])

            return np.column_stack((
                1 - churn_probabilities,
                churn_probabilities,
            ))

    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])

    model = FixedProbabilityModel()

    default_results = evaluate_model(model, X, y)
    custom_results = evaluate_model(model, X, y, threshold=0.65)

    assert (
        default_results["confusion_matrix"]
        != custom_results["confusion_matrix"]
    )

    assert default_results["confusion_matrix"] == [
        [1, 1],
        [1, 1],
    ]

    assert custom_results["confusion_matrix"] == [
        [2, 0],
        [1, 1],
    ]


def test_evaluation_threshold_zero_predicts_all_churn(sample_training_data):
    X, y = sample_training_data
    model = build_baseline_models()["logistic_regression"]

    model.fit(X, y)

    results = evaluate_model(model, X, y, threshold=0.0)

    assert results["confusion_matrix"][0][0] == 0
    assert results["confusion_matrix"][1][0] == 0


def test_evaluation_threshold_one_predicts_no_churn(sample_training_data):
    X, y = sample_training_data
    model = build_baseline_models()["logistic_regression"]

    model.fit(X, y)

    results = evaluate_model(model, X, y, threshold=1.0)

    assert results["confusion_matrix"][0][1] == 0
    assert results["confusion_matrix"][1][1] == 0


@pytest.mark.parametrize(
    "threshold",
    [
        -0.01,
        1.01,
        np.nan,
        np.inf,
        -np.inf,
        "0.65",
        None,
        True,
    ],
)
def test_evaluation_rejects_invalid_thresholds(
    sample_training_data,
    threshold,
):
    X, y = sample_training_data
    model = build_baseline_models()["logistic_regression"]

    model.fit(X, y)

    with pytest.raises(ValueError):
        evaluate_model(model, X, y, threshold=threshold)


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

    for name in first:
        assert first[name] is not second[name]