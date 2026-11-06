"""Tests for dashboard model-insight artifact loading and validation."""

import json

import pandas as pd
import pytest

from dashboard import data_access as data

def valid_test_evaluation():
    """Return a valid held-out test evaluation artifact."""

    return {
        "model_name": "logistic_regression_balanced",
        "algorithm": "Balanced Logistic Regression",
        "threshold": 0.65,
        "training_records": 8000,
        "test_records": 100,
        "metrics": {
            "accuracy": 0.80,
            "precision": 0.60,
            "recall": 0.50,
            "f1_score": 0.5454545,
            "roc_auc": 0.85,
            "average_precision": 0.65,
            "confusion_matrix": [
                [70, 10],
                [10, 10],
            ],
        },
        "classification_report": {
            "Retained (0)": {
                "precision": 0.875,
                "recall": 0.875,
                "f1-score": 0.875,
                "support": 80,
            },
            "Churned (1)": {
                "precision": 0.50,
                "recall": 0.50,
                "f1-score": 0.50,
                "support": 20,
            },
            "accuracy": 0.80,
            "macro avg": {
                "precision": 0.6875,
                "recall": 0.6875,
                "f1-score": 0.6875,
                "support": 100,
            },
            "weighted avg": {
                "precision": 0.80,
                "recall": 0.80,
                "f1-score": 0.80,
                "support": 100,
            },
        },
    }


@pytest.fixture
def test_evaluation_file(tmp_path, monkeypatch):
    """Create a temporary valid test evaluation artifact."""

    path = tmp_path / "final-model-test-evaluation.json"

    path.write_text(
        json.dumps(valid_test_evaluation()),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        data,
        "TEST_EVALUATION_PATH",
        path,
    )

    return path

def valid_fairness_row(attribute="Gender", group="Female"):
    """Return a valid fairness group metrics record."""

    return {
        "attribute": attribute,
        "group": group,
        "sample_count": 100,
        "true_positive": 20,
        "true_negative": 60,
        "false_positive": 10,
        "false_negative": 10,
        "actual_positive_count": 30,
        "actual_negative_count": 70,
        "predicted_positive_count": 30,
        "actual_churn_rate": 0.30,
        "predicted_churn_rate": 0.30,
        "accuracy": 0.80,
        "precision": 0.666667,
        "tpr": 0.666667,
        "fpr": 0.142857,
        "fnr": 0.333333,
        "actual_churn_rate_ci_lower": 0.22,
        "actual_churn_rate_ci_upper": 0.40,
        "predicted_churn_rate_ci_lower": 0.22,
        "predicted_churn_rate_ci_upper": 0.40,
        "tpr_ci_lower": 0.45,
        "tpr_ci_upper": 0.82,
        "fpr_ci_lower": 0.08,
        "fpr_ci_upper": 0.24,
    }


def valid_fairness_summary():
    """Return a valid fairness summary JSON structure."""

    attributes = {}

    for attribute in ("Gender", "Geography", "AgeGroup"):
        attributes[attribute] = {
            "group_metrics": [
                {"group": "Example"}
            ],
            "fairness_summary": {
                "number_of_groups": 1,
                "demographic_parity_difference": 0.15,
                "equal_opportunity_difference": 0.20,
                "false_positive_rate_difference": 0.10,
                "equalized_odds_difference": 0.20,
                "disparate_impact_ratio": 0.75,
            },
        }

    return {
        "evaluation_dataset": "validation",
        "records": 2000,
        "model": "logistic_regression_balanced",
        "threshold": 0.65,
        "audit_attributes": ["Gender", "Geography", "AgeGroup"],
        "confidence_intervals": {
            "method": "Wilson score interval",
            "confidence_level": 0.95,
            "metrics": [
                "actual_churn_rate",
                "predicted_churn_rate",
                "tpr",
                "fpr",
            ],
        },
        "attributes": attributes,
    }


@pytest.fixture
def artifact_files(tmp_path, monkeypatch):
    """Create valid temporary artifacts and redirect loader paths."""

    paths = {
        "coefficients": tmp_path / "feature-coefficients.csv",
        "permutation": tmp_path / "permutation-importance.csv",
        "fairness_groups": tmp_path / "fairness-group-metrics.csv",
        "fairness_summary": tmp_path / "fairness-summary.json",
    }

    monkeypatch.setattr(
        data, "FEATURE_COEFFICIENTS_PATH", paths["coefficients"]
    )
    monkeypatch.setattr(
        data, "PERMUTATION_IMPORTANCE_PATH", paths["permutation"]
    )
    monkeypatch.setattr(
        data, "FAIRNESS_GROUP_METRICS_PATH", paths["fairness_groups"]
    )
    monkeypatch.setattr(
        data, "FAIRNESS_SUMMARY_PATH", paths["fairness_summary"]
    )

    pd.DataFrame([
        {
            "feature": "binary__IsActiveMember",
            "coefficient": -0.91,
            "odds_ratio": 0.40,
            "absolute_coefficient": 0.91,
            "direction": "negative",
        }
    ]).to_csv(paths["coefficients"], index=False)

    pd.DataFrame([
        {
            "feature": "Tenure",
            "importance_mean": -0.0001,
            "importance_std": 0.0002,
        }
    ]).to_csv(paths["permutation"], index=False)

    fairness_rows = [
        valid_fairness_row("Gender", "Female"),
        valid_fairness_row("Geography", "France"),
        valid_fairness_row("AgeGroup", "30-39"),
    ]

    pd.DataFrame(fairness_rows).to_csv(
        paths["fairness_groups"], index=False
    )

    paths["fairness_summary"].write_text(
        json.dumps(valid_fairness_summary()),
        encoding="utf-8",
    )

    return paths


def test_all_model_insight_artifacts_load_successfully(artifact_files):
    coefficients = data.load_feature_coefficients()
    permutation = data.load_permutation_importance()
    fairness_groups = data.load_fairness_group_metrics()
    fairness_summary = data.load_fairness_summary()

    assert len(coefficients) == 1
    assert len(permutation) == 1
    assert len(fairness_groups) == 3

    assert fairness_summary["evaluation_dataset"] == "validation"
    assert set(fairness_summary["attributes"]) == {
        "Gender",
        "Geography",
        "AgeGroup",
    }


def test_missing_artifact_raises_file_not_found(artifact_files):
    artifact_files["coefficients"].unlink()

    with pytest.raises(FileNotFoundError, match="Required artifact not found"):
        data.load_feature_coefficients()


def test_csv_artifact_missing_required_column_raises_error(artifact_files):
    pd.DataFrame([
        {
            "feature": "Age",
            "coefficient": 0.5,
        }
    ]).to_csv(artifact_files["coefficients"], index=False)

    with pytest.raises(ValueError, match="missing columns"):
        data.load_feature_coefficients()


def test_invalid_coefficient_value_raises_error(artifact_files):
    pd.DataFrame([
        {
            "feature": "Age",
            "coefficient": "invalid",
            "odds_ratio": 1.5,
            "absolute_coefficient": 0.5,
            "direction": "positive",
        }
    ]).to_csv(artifact_files["coefficients"], index=False)

    with pytest.raises((ValueError, TypeError)):
        data.load_feature_coefficients()


def test_infinite_coefficient_is_rejected(artifact_files):
    pd.DataFrame([
        {
            "feature": "Age",
            "coefficient": float("inf"),
            "odds_ratio": 1.5,
            "absolute_coefficient": 0.5,
            "direction": "positive",
        }
    ]).to_csv(artifact_files["coefficients"], index=False)

    with pytest.raises(ValueError, match="non-finite"):
        data.load_feature_coefficients()


def test_negative_permutation_importance_is_preserved(artifact_files):
    result = data.load_permutation_importance()

    assert result.loc[0, "importance_mean"] < 0


def test_fairness_group_metrics_accept_null_precision(artifact_files):
    row = valid_fairness_row()
    row["precision"] = None

    pd.DataFrame([row]).to_csv(
        artifact_files["fairness_groups"], index=False
    )

    result = data.load_fairness_group_metrics()

    assert pd.isna(result.loc[0, "precision"])


def test_fairness_group_metrics_reject_invalid_confidence_interval(
    artifact_files,
):
    row = valid_fairness_row()
    row["actual_churn_rate_ci_lower"] = 0.50
    row["actual_churn_rate_ci_upper"] = 0.40

    pd.DataFrame([row]).to_csv(
        artifact_files["fairness_groups"], index=False
    )

    with pytest.raises(ValueError, match="Invalid confidence interval"):
        data.load_fairness_group_metrics()


def test_fairness_group_metrics_reject_duplicate_groups(artifact_files):
    row = valid_fairness_row()

    pd.DataFrame([row, row]).to_csv(
        artifact_files["fairness_groups"], index=False
    )

    with pytest.raises(ValueError, match="duplicate attribute/group pairs"):
        data.load_fairness_group_metrics()


def test_fairness_summary_rejects_invalid_group_metrics_structure(
    artifact_files,
):
    summary = valid_fairness_summary()
    summary["attributes"]["Gender"]["fairness_summary"] = []

    artifact_files["fairness_summary"].write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be an object"):
        data.load_fairness_summary()


def test_fairness_summary_rejects_invalid_metric_value(artifact_files):
    summary = valid_fairness_summary()

    summary["attributes"]["Gender"]["fairness_summary"][
        "demographic_parity_difference"
    ] = 1.5

    artifact_files["fairness_summary"].write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid demographic_parity_difference"):
        data.load_fairness_summary()


def test_fairness_summary_rejects_duplicate_group_names(artifact_files):
    summary = valid_fairness_summary()

    summary["attributes"]["Gender"]["group_metrics"] = [
        {"group": "Female"},
        {"group": "Female"},
    ]
    summary["attributes"]["Gender"]["fairness_summary"][
        "number_of_groups"
    ] = 2

    artifact_files["fairness_summary"].write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate names"):
        data.load_fairness_summary()

def test_load_test_evaluation_accepts_valid_artifact(
    test_evaluation_file,
):
    result = data.load_test_evaluation()

    assert result["algorithm"] == "Balanced Logistic Regression"
    assert result["test_records"] == 100
    assert result["threshold"] == 0.65
    assert result["classification_report"]["Churned (1)"]["support"] == 20


def test_load_test_evaluation_rejects_missing_classification_report(
    test_evaluation_file,
):
    evaluation = valid_test_evaluation()
    del evaluation["classification_report"]

    test_evaluation_file.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="classification_report"):
        data.load_test_evaluation()


def test_load_test_evaluation_rejects_missing_class(
    test_evaluation_file,
):
    evaluation = valid_test_evaluation()
    del evaluation["classification_report"]["Churned (1)"]

    test_evaluation_file.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing entries"):
        data.load_test_evaluation()


def test_load_test_evaluation_rejects_invalid_metric(
    test_evaluation_file,
):
    evaluation = valid_test_evaluation()
    evaluation["metrics"]["roc_auc"] = 1.5

    test_evaluation_file.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="roc_auc"):
        data.load_test_evaluation()


def test_load_test_evaluation_rejects_invalid_confusion_matrix(
    test_evaluation_file,
):
    evaluation = valid_test_evaluation()
    evaluation["metrics"]["confusion_matrix"] = [[70, 10], [10]]

    test_evaluation_file.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="2x2"):
        data.load_test_evaluation()


def test_load_test_evaluation_rejects_inconsistent_confusion_matrix(
    test_evaluation_file,
):
    evaluation = valid_test_evaluation()
    evaluation["metrics"]["confusion_matrix"] = [
        [70, 10],
        [10, 11],
    ]

    test_evaluation_file.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match test_records"):
        data.load_test_evaluation()


def test_load_test_evaluation_rejects_inconsistent_class_support(
    test_evaluation_file,
):
    evaluation = valid_test_evaluation()
    evaluation["classification_report"]["Churned (1)"]["support"] = 21

    test_evaluation_file.write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="support does not match"):
        data.load_test_evaluation()


def test_load_test_evaluation_rejects_missing_artifact(
    tmp_path,
    monkeypatch,
):
    missing_path = tmp_path / "missing-evaluation.json"

    monkeypatch.setattr(
        data,
        "TEST_EVALUATION_PATH",
        missing_path,
    )

    with pytest.raises(FileNotFoundError):
        data.load_test_evaluation()