import numpy as np
import pandas as pd
import pytest

from src.fairness import (
    calculate_fairness_summary,
    calculate_group_metrics,
)
from src.fairness import calculate_binomial_confidence_interval

@pytest.fixture
def sample_predictions():
    y_true = pd.Series([1, 1, 0, 0, 1, 0])
    y_pred = pd.Series([1, 0, 1, 0, 1, 0])
    groups = pd.Series(["A", "A", "A", "A", "B", "B"])

    return y_true, y_pred, groups


def test_group_metrics_calculate_expected_values(sample_predictions):
    y_true, y_pred, groups = sample_predictions

    results = calculate_group_metrics(y_true, y_pred, groups)

    assert list(results.index) == ["A", "B"]

    assert results.loc["A", "sample_count"] == 4
    assert results.loc["A", "actual_churn_rate"] == 0.5
    assert results.loc["A", "predicted_churn_rate"] == 0.5
    assert results.loc["A", "accuracy"] == 0.5
    assert results.loc["A", "precision"] == 0.5
    assert results.loc["A", "f1_score"] == 0.5
    assert results.loc["A", "tpr"] == 0.5
    assert results.loc["A", "fpr"] == 0.5
    assert results.loc["A", "fnr"] == 0.5

    assert results.loc["B", "sample_count"] == 2
    assert results.loc["B", "actual_churn_rate"] == 0.5
    assert results.loc["B", "predicted_churn_rate"] == 0.5
    assert results.loc["B", "accuracy"] == 1.0
    assert results.loc["B", "precision"] == 1.0
    assert results.loc["B", "f1_score"] == 1.0
    assert results.loc["B", "tpr"] == 1.0
    assert results.loc["B", "fpr"] == 0.0
    assert results.loc["B", "fnr"] == 0.0


def test_fairness_summary_calculates_expected_values(
    sample_predictions,
):
    y_true, y_pred, groups = sample_predictions

    group_metrics = calculate_group_metrics(
        y_true,
        y_pred,
        groups,
    )

    summary = calculate_fairness_summary(group_metrics)

    assert summary["number_of_groups"] == 2
    assert summary["demographic_parity_difference"] == 0.0
    assert summary["equal_opportunity_difference"] == 0.5
    assert summary["false_positive_rate_difference"] == 0.5
    assert summary["equalized_odds_difference"] == 0.5
    assert summary["disparate_impact_ratio"] == 1.0


def test_group_metrics_handle_zero_predicted_positives():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 0, 0, 0])
    groups = pd.Series(["A", "A", "B", "B"])

    results = calculate_group_metrics(y_true, y_pred, groups)

    assert np.isnan(results.loc["A", "precision"])
    assert np.isnan(results.loc["A", "f1_score"])
    assert results.loc["A", "fpr"] == 0.0
    assert results.loc["B", "fnr"] == 1.0


def test_group_metrics_handle_missing_actual_classes():
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([1, 0, 1, 0])
    groups = pd.Series(["A", "A", "B", "B"])

    results = calculate_group_metrics(y_true, y_pred, groups)

    assert np.isnan(results.loc["A", "fpr"])
    assert np.isnan(results.loc["B", "tpr"])
    assert np.isnan(results.loc["B", "fnr"])


def test_disparate_impact_is_nan_when_all_groups_have_zero_positive_rate():
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 0, 0, 0])
    groups = pd.Series(["A", "A", "B", "B"])

    group_metrics = calculate_group_metrics(y_true, y_pred, groups)
    summary = calculate_fairness_summary(group_metrics)

    assert np.isnan(summary["disparate_impact_ratio"])


def test_inputs_must_have_equal_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        calculate_group_metrics(
            [0, 1],
            [0, 1],
            ["A"],
        )


def test_inputs_must_be_one_dimensional():
    with pytest.raises(ValueError, match="one-dimensional"):
        calculate_group_metrics(
            [[0, 1]],
            [0, 1],
            ["A", "B"],
        )


def test_inputs_must_not_be_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_group_metrics([], [], [])


@pytest.mark.parametrize(
    "y_true",
    [
        [0, 2],
        [0, -1],
        [0, np.nan],
    ],
)
def test_actual_labels_must_be_binary(y_true):
    with pytest.raises(ValueError, match="y_true"):
        calculate_group_metrics(y_true, [0, 1], ["A", "B"])


@pytest.mark.parametrize(
    "y_pred",
    [
        [0, 2],
        [0, -1],
        [0, np.nan],
    ],
)
def test_predictions_must_be_binary(y_pred):
    with pytest.raises(ValueError, match="y_pred"):
        calculate_group_metrics([0, 1], y_pred, ["A", "B"])


def test_group_membership_must_not_be_missing():
    with pytest.raises(ValueError, match="missing values"):
        calculate_group_metrics(
            [0, 1],
            [0, 1],
            ["A", None],
        )


def test_pandas_series_indexes_must_match():
    y_true = pd.Series([0, 1], index=[1, 2])
    y_pred = pd.Series([0, 1], index=[1, 2])
    groups = pd.Series(["A", "B"], index=[2, 3])

    with pytest.raises(ValueError, match="indexes must match"):
        calculate_group_metrics(y_true, y_pred, groups)


def test_fairness_summary_rejects_invalid_input():
    with pytest.raises(ValueError, match="pandas DataFrame"):
        calculate_fairness_summary({"tpr": 0.5})


def test_fairness_summary_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required"):
        calculate_fairness_summary(
            pd.DataFrame({"tpr": [0.5], "fpr": [0.2]})
        )


def test_fairness_summary_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_fairness_summary(
            pd.DataFrame(
                columns=[
                    "predicted_churn_rate",
                    "tpr",
                    "fpr",
                ]
            )
        )

def test_wilson_interval_for_fifty_successes_out_of_one_hundred():
    lower, upper = calculate_binomial_confidence_interval(50, 100)

    assert lower == pytest.approx(0.4038, abs=0.0001)
    assert upper == pytest.approx(0.5962, abs=0.0001)


def test_wilson_interval_for_zero_successes():
    lower, upper = calculate_binomial_confidence_interval(0, 100)

    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(0.03699, abs=0.0001)


def test_wilson_interval_for_all_successes():
    lower, upper = calculate_binomial_confidence_interval(100, 100)

    assert lower == pytest.approx(0.9630, abs=0.0001)
    assert upper == pytest.approx(1.0)


def test_wilson_interval_returns_nan_for_zero_observations():
    lower, upper = calculate_binomial_confidence_interval(0, 0)

    assert np.isnan(lower)
    assert np.isnan(upper)


@pytest.mark.parametrize(
    "successes, total, confidence",
    [
        (-1, 10, 0.95),
        (11, 10, 0.95),
        (5, -1, 0.95),
        (5, 10, 0),
        (5, 10, 1),
        (5, 10, 1.5),
    ],
)
def test_wilson_interval_rejects_invalid_inputs(
    successes,
    total,
    confidence,
):
    with pytest.raises(ValueError):
        calculate_binomial_confidence_interval(
            successes,
            total,
            confidence,
        )


def test_group_metrics_include_confusion_matrix_counts(sample_predictions):
    y_true, y_pred, groups = sample_predictions

    results = calculate_group_metrics(y_true, y_pred, groups)

    group_a = results.loc["A"]

    assert group_a["true_positive"] == 1
    assert group_a["true_negative"] == 1
    assert group_a["false_positive"] == 1
    assert group_a["false_negative"] == 1

    assert group_a["actual_positive_count"] == 2
    assert group_a["actual_negative_count"] == 2
    assert group_a["predicted_positive_count"] == 2


def test_group_metrics_include_valid_confidence_intervals(
    sample_predictions,
):
    y_true, y_pred, groups = sample_predictions

    results = calculate_group_metrics(y_true, y_pred, groups)

    for metric in ["actual_churn_rate", "predicted_churn_rate",
                   "tpr", "fpr"]:
        lower = results[f"{metric}_ci_lower"]
        upper = results[f"{metric}_ci_upper"]
        rate = results[metric]

        for group in results.index:
            if np.isnan(rate.loc[group]):
                assert np.isnan(lower.loc[group])
                assert np.isnan(upper.loc[group])
            else:
                assert 0 <= lower.loc[group] <= rate.loc[group]
                assert rate.loc[group] <= upper.loc[group] <= 1


def test_group_metrics_confidence_intervals_handle_missing_classes():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    groups = ["A", "A", "B", "B"]

    results = calculate_group_metrics(y_true, y_pred, groups)

    assert np.isnan(results.loc["A", "fpr_ci_lower"])
    assert np.isnan(results.loc["A", "fpr_ci_upper"])

    assert np.isnan(results.loc["B", "tpr_ci_lower"])
    assert np.isnan(results.loc["B", "tpr_ci_upper"])
