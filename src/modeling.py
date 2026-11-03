"""
Modeling module for Eurobank customer churn prediction.

Provides:
1. Majority-class Dummy Classifier.
2. Standard Logistic Regression.
3. Class-weighted Logistic Regression.
4. Configurable binary classification evaluation.
"""

import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_baseline_models() -> dict:
    """
    Build fresh, unfitted baseline classification models.

    Returns
    -------
    dict
        Dictionary containing Dummy Classifier,
        Standard Logistic Regression and
        Class-weighted Logistic Regression.
    """

    dummy_model = DummyClassifier(
        strategy="most_frequent",
    )

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight=None,
                ),
            ),
        ]
    )

    balanced_logistic_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return {
        "dummy": dummy_model,
        "logistic_regression": logistic_model,
        "logistic_regression_balanced": balanced_logistic_model,
    }


def evaluate_model(
    model,
    X_test,
    y_test,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a fitted binary classification model.

    Class 1 represents a churned customer.

    Parameters
    ----------
    model:
        Fitted binary classification model.
    X_test:
        Feature dataset.
    y_test:
        Actual target values.
    threshold:
        Probability threshold for predicting churn.
        Defaults to 0.5.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1-score,
        ROC-AUC and confusion matrix.
    """

    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(
            threshold,
            (int, float, np.integer, np.floating),
        )
    ):
        raise ValueError("Threshold must be a numeric value.")

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    probabilities = model.predict_proba(X_test)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model does not contain positive class 1.")

    positive_class_index = classes.index(1)

    churn_probabilities = probabilities[:, positive_class_index]

    if not np.isfinite(churn_probabilities).all():
        raise ValueError("Model produced invalid prediction probabilities.")

    predictions = (churn_probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            churn_probabilities,
        ),
        "confusion_matrix": matrix.tolist(),
    }