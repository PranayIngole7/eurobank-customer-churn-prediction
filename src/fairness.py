"""
Fairness metrics for binary customer churn classification.

This module calculates group-wise model performance,
confusion-matrix counts, confidence intervals, and
comparisons across demographic or other audit groups.

Class 1 represents a churned customer.
"""

from statistics import NormalDist

import numpy as np
import pandas as pd


def _validate_inputs(y_true, y_pred, groups):
    """Validate labels, predictions, and group membership."""

    if isinstance(y_true, pd.Series):
        if not isinstance(y_pred, pd.Series) or not isinstance(
            groups, pd.Series
        ):
            raise ValueError(
                "When y_true is a pandas Series, "
                "y_pred and groups must also be Series."
            )

        if not y_true.index.equals(y_pred.index):
            raise ValueError("y_true and y_pred indexes must match.")

        if not y_true.index.equals(groups.index):
            raise ValueError("y_true and groups indexes must match.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    if y_true.ndim != 1 or y_pred.ndim != 1 or groups.ndim != 1:
        raise ValueError("All inputs must be one-dimensional.")

    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("All inputs must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError(
            "y_pred must contain only binary predictions 0 and 1."
        )

    if pd.isna(groups).any():
        raise ValueError("Group membership must not contain missing values.")

    return y_true.astype(int), y_pred.astype(int), groups


def _safe_divide(numerator, denominator):
    """Divide safely, returning NaN when the denominator is zero."""

    if denominator == 0:
        return np.nan

    return numerator / denominator


def calculate_binomial_confidence_interval(
    successes,
    total,
    confidence=0.95,
):
    """
    Calculate a Wilson score confidence interval for a binomial rate.

    Parameters
    ----------
    successes : int
        Number of observed successes.
    total : int
        Total number of observations.
    confidence : float
        Confidence level, between 0 and 1.

    Returns
    -------
    tuple[float, float]
        Lower and upper confidence interval bounds.
        Returns (NaN, NaN) when total is zero.
    """

    if isinstance(successes, (bool, np.bool_)) or not isinstance(
        successes, (int, np.integer)
    ):
        raise ValueError("successes must be a non-negative integer.")

    if isinstance(total, (bool, np.bool_)) or not isinstance(
        total, (int, np.integer)
    ):
        raise ValueError("total must be a non-negative integer.")

    if successes < 0 or total < 0:
        raise ValueError("successes and total must be non-negative.")

    if successes > total:
        raise ValueError("successes cannot exceed total.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    if total == 0:
        return np.nan, np.nan

    observed_rate = successes / total

    z_score = NormalDist().inv_cdf(
        1 - (1 - confidence) / 2
    )

    z_squared = z_score ** 2

    denominator = 1 + z_squared / total

    center = (
        observed_rate + z_squared / (2 * total)
    ) / denominator

    margin = (
        z_score
        * np.sqrt(
            observed_rate * (1 - observed_rate) / total
            + z_squared / (4 * total ** 2)
        )
        / denominator
    )

    lower_bound = max(0.0, center - margin)
    upper_bound = min(1.0, center + margin)

    return float(lower_bound), float(upper_bound)


def calculate_group_metrics(y_true, y_pred, groups):
    """
    Calculate group-wise classification performance.

    Returns confusion-matrix counts, classification rates,
    and Wilson confidence intervals for selected rates.
    """

    y_true, y_pred, groups = _validate_inputs(
        y_true,
        y_pred,
        groups,
    )

    data = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group": groups,
    })

    results = []

    for group_name, group_data in data.groupby(
        "group",
        sort=False,
        observed=True,
    ):
        actual = group_data["y_true"].to_numpy()
        predicted = group_data["y_pred"].to_numpy()

        true_positive = int(
            np.sum((actual == 1) & (predicted == 1))
        )

        true_negative = int(
            np.sum((actual == 0) & (predicted == 0))
        )

        false_positive = int(
            np.sum((actual == 0) & (predicted == 1))
        )

        false_negative = int(
            np.sum((actual == 1) & (predicted == 0))
        )

        total = len(group_data)

        actual_positive = true_positive + false_negative
        actual_negative = true_negative + false_positive

        predicted_positive = true_positive + false_positive

        actual_churn_rate = _safe_divide(
            actual_positive,
            total,
        )

        predicted_churn_rate = _safe_divide(
            predicted_positive,
            total,
        )

        accuracy = _safe_divide(
            true_positive + true_negative,
            total,
        )

        precision = _safe_divide(
            true_positive,
            predicted_positive,
        )

        tpr = _safe_divide(
            true_positive,
            actual_positive,
        )

        fpr = _safe_divide(
            false_positive,
            actual_negative,
        )

        fnr = _safe_divide(
            false_negative,
            actual_positive,
        )

        actual_ci = calculate_binomial_confidence_interval(
            actual_positive,
            total,
        )

        predicted_ci = calculate_binomial_confidence_interval(
            predicted_positive,
            total,
        )

        tpr_ci = calculate_binomial_confidence_interval(
            true_positive,
            actual_positive,
        )

        fpr_ci = calculate_binomial_confidence_interval(
            false_positive,
            actual_negative,
        )

        results.append({
            "group": group_name,

            # Sample and confusion-matrix counts
            "sample_count": total,
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "actual_positive_count": actual_positive,
            "actual_negative_count": actual_negative,
            "predicted_positive_count": predicted_positive,

            # Classification metrics
            "actual_churn_rate": actual_churn_rate,
            "predicted_churn_rate": predicted_churn_rate,
            "accuracy": accuracy,
            "precision": precision,
            "tpr": tpr,
            "fpr": fpr,
            "fnr": fnr,

            # Wilson confidence intervals
            "actual_churn_rate_ci_lower": actual_ci[0],
            "actual_churn_rate_ci_upper": actual_ci[1],
            "predicted_churn_rate_ci_lower": predicted_ci[0],
            "predicted_churn_rate_ci_upper": predicted_ci[1],
            "tpr_ci_lower": tpr_ci[0],
            "tpr_ci_upper": tpr_ci[1],
            "fpr_ci_lower": fpr_ci[0],
            "fpr_ci_upper": fpr_ci[1],
        })

    return pd.DataFrame(results).set_index("group")


def calculate_fairness_summary(group_metrics):
    """
    Compare group-wise prediction and error rates.

    Demographic parity difference:
        Maximum minus minimum predicted-positive rate.

    Equal opportunity difference:
        Maximum minus minimum true positive rate.

    Equalized odds difference:
        Maximum of the TPR difference and FPR difference.

    Disparate impact ratio:
        Minimum divided by maximum predicted-positive rate.

    For multiple groups, disparate impact uses the minimum
    positive prediction rate divided by the maximum rate.

    Undefined comparisons return NaN.
    """

    if not isinstance(group_metrics, pd.DataFrame):
        raise ValueError("group_metrics must be a pandas DataFrame.")

    required_columns = {
        "predicted_churn_rate",
        "tpr",
        "fpr",
    }

    if not required_columns.issubset(group_metrics.columns):
        raise ValueError(
            "group_metrics is missing required metric columns."
        )

    if group_metrics.empty:
        raise ValueError("group_metrics must not be empty.")

    def calculate_difference(column):
        values = group_metrics[column].dropna()

        if len(values) < 2:
            return np.nan

        return float(values.max() - values.min())

    def calculate_disparate_impact():
        rates = group_metrics["predicted_churn_rate"].dropna()

        if len(rates) < 2:
            return np.nan

        maximum = rates.max()

        if maximum == 0:
            return np.nan

        return float(rates.min() / maximum)

    demographic_parity_difference = calculate_difference(
        "predicted_churn_rate"
    )

    equal_opportunity_difference = calculate_difference("tpr")

    fpr_difference = calculate_difference("fpr")

    if (
        np.isnan(equal_opportunity_difference)
        or np.isnan(fpr_difference)
    ):
        equalized_odds_difference = np.nan
    else:
        equalized_odds_difference = max(
            equal_opportunity_difference,
            fpr_difference,
        )

    return {
        "number_of_groups": int(len(group_metrics)),
        "demographic_parity_difference": (
            demographic_parity_difference
        ),
        "equal_opportunity_difference": (
            equal_opportunity_difference
        ),
        "false_positive_rate_difference": fpr_difference,
        "equalized_odds_difference": equalized_odds_difference,
        "disparate_impact_ratio": calculate_disparate_impact(),
    }