"""
Evaluation utilities for Eurobank customer churn prediction.

Provides:
1. Classification report generation.
2. Confusion matrix visualization.
3. ROC curve visualization.
4. Precision-Recall curve visualization.
"""
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

import numpy as np

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def _validate_binary_labels(y_true, y_pred):
    """
    Validate actual and predicted binary classification labels.
    """

    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError(
            "Actual and predicted labels must be one-dimensional."
        )

    if len(actual) == 0 or len(predicted) == 0:
        raise ValueError(
            "Actual and predicted labels must not be empty."
        )

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted labels must have equal lengths."
        )

    if (
        not np.isin(actual, [0, 1]).all()
        or not np.isin(predicted, [0, 1]).all()
    ):
        raise ValueError("Labels must contain only 0 and 1.")

    return actual, predicted


def generate_classification_report(y_true, y_pred) -> dict:
    """
    Generate a classification report for binary churn predictions.

    Class 0 represents retained customers.
    Class 1 represents churned customers.
    """

    actual, predicted = _validate_binary_labels(y_true, y_pred)

    return classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Retained (0)", "Churned (1)"],
        output_dict=True,
        zero_division=0,
    )


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_path=None,
):
    """
    Generate a confusion matrix visualization.

    Parameters
    ----------
    y_true:
        Actual target labels.
    y_pred:
        Predicted target labels.
    output_path:
        Optional path where the figure will be saved.

    Returns
    -------
    tuple
        Matplotlib figure and axes.
    """

    import matplotlib.pyplot as plt

    actual, predicted = _validate_binary_labels(y_true, y_pred)

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Retained (0)", "Churned (1)"],
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Customer Churn - Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax

def plot_roc_curve(y_true, y_score, output_path=None):
    """
    Generate a Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_true:
        Actual binary target labels.
    y_score:
        Model scores or probabilities for positive class 1.
    output_path:
        Optional path where the figure will be saved.

    Returns
    -------
    tuple
        Matplotlib figure, axes and ROC-AUC score.
    """

    import matplotlib.pyplot as plt

    actual = np.asarray(y_true)
    scores = np.asarray(y_score)

    if actual.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional.")

    if len(actual) == 0 or len(scores) == 0:
        raise ValueError("Labels and scores must not be empty.")

    if len(actual) != len(scores):
        raise ValueError("Labels and scores must have equal lengths.")

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    if not np.issubdtype(scores.dtype, np.number):
        raise ValueError("Scores must be numeric.")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values.")

    if np.unique(actual).size != 2:
        raise ValueError("ROC curve requires both target classes.")

    false_positive_rate, true_positive_rate, _ = roc_curve(
        actual,
        scores,
        pos_label=1,
    )

    auc_score = roc_auc_score(actual, scores)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC curve (AUC = {auc_score:.4f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_title("Customer Churn - ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax, auc_score

def plot_precision_recall_curve(y_true, y_score, output_path=None):
    """
    Generate a Precision-Recall curve.

    Parameters
    ----------
    y_true:
        Actual binary target labels.
    y_score:
        Model scores or probabilities for positive class 1.
    output_path:
        Optional path where the figure will be saved.

    Returns
    -------
    tuple
        Matplotlib figure, axes and average precision score.
    """

    import matplotlib.pyplot as plt

    actual = np.asarray(y_true)
    scores = np.asarray(y_score)

    if actual.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional.")

    if len(actual) == 0 or len(scores) == 0:
        raise ValueError("Labels and scores must not be empty.")

    if len(actual) != len(scores):
        raise ValueError("Labels and scores must have equal lengths.")

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    if not np.issubdtype(scores.dtype, np.number):
        raise ValueError("Scores must be numeric.")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values.")

    if np.unique(actual).size != 2:
        raise ValueError(
            "Precision-Recall curve requires both target classes."
        )

    precision, recall, _ = precision_recall_curve(
        actual,
        scores,
        pos_label=1,
    )

    average_precision = average_precision_score(
        actual,
        scores,
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        recall,
        precision,
        label=f"PR curve (AP = {average_precision:.4f})",
    )

    ax.set_title("Customer Churn - Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax, average_precision

def compare_thresholds(y_true, y_score, thresholds):
    """
    Compare classification performance across multiple probability thresholds.

    Parameters
    ----------
    y_true : array-like
        Actual binary class labels (0 or 1).

    y_score : array-like
        Predicted probabilities for the positive class.

    thresholds : array-like
        Classification thresholds between 0 and 1.

    Returns
    -------
    pandas.DataFrame
        Threshold-wise precision, recall, F1-score, and confusion matrix counts.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.asarray(thresholds)

    # Validate dimensions and non-empty inputs.
    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true must be a non-empty 1D array.")

    if y_score.ndim != 1 or y_score.size == 0:
        raise ValueError("y_score must be a non-empty 1D array.")

    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty 1D array.")

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")

    # Validate actual labels.
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("y_true must contain both classes: 0 and 1.")

    # Validate probability scores.
    try:
        y_score = y_score.astype(float)
    except (TypeError, ValueError):
        raise ValueError("y_score must contain numeric probability scores.")

    if not np.isfinite(y_score).all():
        raise ValueError("y_score must contain only finite values.")

    if ((y_score < 0) | (y_score > 1)).any():
        raise ValueError("y_score values must be between 0 and 1.")

    # Validate thresholds.
    if thresholds.dtype.kind not in "iuf":
        raise ValueError(
            "thresholds must contain numeric values, not strings or booleans."
        )

    thresholds = thresholds.astype(float)

    if not np.isfinite(thresholds).all():
        raise ValueError("thresholds must contain only finite values.")

    if ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("thresholds must be between 0 and 1.")

    results = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    return pd.DataFrame(results)

def plot_threshold_comparison(comparison, output_path=None):
    """
    Plot precision, recall, and F1-score across classification thresholds.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Results returned by compare_thresholds().

    output_path : str or Path, optional
        Location where the plot should be saved.

    Returns
    -------
    tuple
        Matplotlib figure and axes.
    """

    import matplotlib.pyplot as plt

    required_columns = {
        "threshold",
        "precision",
        "recall",
        "f1_score",
    }

    if not isinstance(comparison, pd.DataFrame):
        raise ValueError("comparison must be a pandas DataFrame.")

    if comparison.empty:
        raise ValueError("comparison DataFrame must not be empty.")

    if not required_columns.issubset(comparison.columns):
        raise ValueError(
            f"comparison must contain these columns: {required_columns}"
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        comparison["threshold"],
        comparison["precision"],
        marker="o",
        label="Precision",
    )

    ax.plot(
        comparison["threshold"],
        comparison["recall"],
        marker="o",
        label="Recall",
    )

    ax.plot(
        comparison["threshold"],
        comparison["f1_score"],
        marker="o",
        label="F1-score",
    )

    ax.set_title("Classification Performance Across Thresholds")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax