"""Customer churn risk scoring."""

import numpy as np
import pandas as pd


RISK_THRESHOLDS = [0.0, 0.30, 0.50, 0.70, np.nextafter(1.0, np.inf)]

RISK_LABELS = ["Low", "Moderate", "High", "Critical"]


def score_customers(model, X, threshold=0.65):
    """Generate churn probabilities, risk scores, categories, and predictions."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X must contain at least one customer.")

    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise TypeError("threshold must be numeric.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    if not hasattr(model, "predict_proba"):
        raise TypeError("model must support predict_proba.")

    if not hasattr(model, "classes_"):
        raise ValueError("model must be fitted before scoring.")

    classes = np.asarray(model.classes_)
    positive_class = np.flatnonzero(classes == 1)

    if len(positive_class) != 1:
        raise ValueError("model must contain exactly one positive class (1).")

    probabilities = np.asarray(model.predict_proba(X))

    if probabilities.ndim != 2 or probabilities.shape != (
        len(X), len(classes)
    ):
        raise ValueError("Model returned an invalid probability array.")

    churn_probabilities = probabilities[:, positive_class[0]]

    if not np.isfinite(churn_probabilities).all():
        raise ValueError("Predicted probabilities must be finite.")

    if ((churn_probabilities < 0) | (churn_probabilities > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    risk_scores = churn_probabilities * 100

    risk_categories = pd.cut(
        churn_probabilities,
        bins=RISK_THRESHOLDS,
        labels=RISK_LABELS,
        right=False,
    )

    return pd.DataFrame(
        {
            "churn_probability": churn_probabilities,
            "risk_score": risk_scores,
            "risk_category": risk_categories.astype("object"),
            "churn_prediction": (churn_probabilities >= threshold).astype(int),
        },
        index=X.index,
    )
