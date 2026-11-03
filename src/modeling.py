"""
Baseline modeling module for Eurobank customer churn prediction.

Provides:
1. Majority-class Dummy Classifier.
2. Logistic Regression pipeline.
3. Classification evaluation metrics.
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
        Dictionary containing Dummy Classifier and
        Logistic Regression pipeline.
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
                ),
            ),
        ]
    )

    return {
        "dummy": dummy_model,
        "logistic_regression": logistic_model,
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a fitted binary classification model.

    Class 1 represents a churned customer.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1-score,
        ROC-AUC and confusion matrix.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model does not contain positive class 1.")

    positive_class_index = classes.index(1)

    churn_probabilities = probabilities[:, positive_class_index]

    if not np.isfinite(churn_probabilities).all():
        raise ValueError("Model produced invalid prediction probabilities.")

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
