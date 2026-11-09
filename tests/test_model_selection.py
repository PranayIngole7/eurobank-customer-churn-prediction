import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model_selection import (
    compare_candidate_models,
    select_model_and_threshold,
)


def _features(indexes):
    rows = []
    for index in indexes:
        age = 25 + index % 40
        tenure = index % 11
        products = 1 + index % 3
        balance = float(index * 750)
        salary = float(30000 + index * 100)
        active = index % 2
        rows.append({
            "CreditScore": 600 + index % 200,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "EstimatedSalary": salary,
            "BalanceToSalaryRatio": balance / salary,
            "ProductDensityIndicator": products / tenure if tenure else 0.0,
            "EngagementProductInteraction": active * products,
            "AgeTenureInteraction": age * tenure,
            "Geography": ("France", "Germany", "Spain")[index % 3],
            "Gender": ("Female", "Male")[index % 2],
            "AgeGroup": ("<30", "30-39", "40-49", "50-59", "60+")[index % 5],
            "HasCrCard": index % 2,
            "IsActiveMember": active,
            "HasBalance": int(balance > 0),
        })
    return pd.DataFrame(rows, index=indexes)


@pytest.fixture
def train_validation_data():
    train_indexes = list(range(100))
    validation_indexes = list(range(200, 250))
    X_train = _features(train_indexes)
    X_validation = _features(validation_indexes)
    y_train = pd.Series(
        [int(index % 5 == 0 or index % 7 == 0) for index in train_indexes],
        index=train_indexes,
        name="Exited",
    )
    y_validation = pd.Series(
        [int(index % 5 == 0 or index % 7 == 0) for index in validation_indexes],
        index=validation_indexes,
        name="Exited",
    )
    return X_train, y_train, X_validation, y_validation


def test_comparison_fits_and_evaluates_all_candidates(train_validation_data):
    X_train, y_train, X_validation, y_validation = train_validation_data

    models, comparison, probabilities = compare_candidate_models(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    assert len(models) == 6
    assert set(comparison["model_name"]) == set(models)
    assert set(probabilities) == set(models)
    assert {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
        "tn",
        "fp",
        "fn",
        "tp",
        "threshold",
    }.issubset(comparison.columns)
    assert all(len(scores) == len(y_validation) for scores in probabilities.values())


def test_comparison_metrics_match_validation_probabilities(
    train_validation_data,
):
    X_train, y_train, X_validation, y_validation = train_validation_data
    models, comparison, probabilities = compare_candidate_models(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    row = comparison.set_index("model_name").loc["logistic_regression"]

    assert row["average_precision"] == pytest.approx(
        average_precision_score(
            y_validation,
            probabilities["logistic_regression"],
        )
    )
    assert row["roc_auc"] == pytest.approx(
        roc_auc_score(
            y_validation,
            probabilities["logistic_regression"],
        )
    )


def test_comparison_rejects_overlapping_train_and_validation_indexes(
    train_validation_data,
):
    X_train, y_train, X_validation, y_validation = train_validation_data
    X_validation.index = X_validation.index - 200
    y_validation.index = X_validation.index

    with pytest.raises(ValueError, match="must not overlap"):
        compare_candidate_models(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )


def test_selection_uses_validation_probabilities_and_thresholds(
    train_validation_data,
):
    X_train, y_train, X_validation, y_validation = train_validation_data
    models, _, probabilities = compare_candidate_models(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    selection, threshold_results = select_model_and_threshold(
        models,
        probabilities,
        y_validation,
        thresholds=[0.25, 0.5, 0.75],
    )

    assert selection["model_name"] in models
    assert selection["threshold"] in {0.25, 0.5, 0.75}
    assert selection["validation_metrics"]["f1_score"] >= 0
    assert set(threshold_results["model_name"]) == set(models)


@pytest.mark.parametrize(
    "scores",
    [
        np.array([0.1, np.nan]),
        np.array([0.1, 1.1]),
        np.array([[0.1, 0.9]]),
    ],
)
def test_selection_rejects_invalid_probabilities(scores):
    model = object()
    y_validation = pd.Series([0, 1])

    with pytest.raises(ValueError, match="Invalid validation"):
        select_model_and_threshold(
            {"model": model},
            {"model": scores},
            y_validation,
            thresholds=[0.5],
        )
