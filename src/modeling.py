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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

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


def build_candidate_models() -> dict:
    """Build fresh pipelines for all algorithms compared on validation data."""
    models = build_baseline_models()

    models["decision_tree"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=6,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    models["random_forest"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]
    )

    models["gradient_boosting"] = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=2,
                    min_samples_leaf=10,
                    subsample=0.8,
                ),
            ),
        ]
    )

    return models


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
        ROC-AUC, average precision and confusion matrix.
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

    probabilities = np.asarray(model.predict_proba(X_test))

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model does not contain positive class 1.")

    positive_class_index = classes.index(1)

    if probabilities.ndim != 2 or probabilities.shape != (
        len(X_test),
        len(classes),
    ):
        raise ValueError("Model produced an invalid probability array.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced invalid prediction probabilities.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model probabilities must be between 0 and 1.")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Model probabilities must sum to 1 for each row.")

    churn_probabilities = probabilities[:, positive_class_index]

    if not np.isfinite(churn_probabilities).all():
        raise ValueError("Model produced invalid prediction probabilities.")

    predictions = (churn_probabilities >= threshold).astype(int)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    result = {
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
        "average_precision": average_precision_score(
            y_test,
            churn_probabilities,
        ),
        "confusion_matrix": matrix.tolist(),
    }

    return result
