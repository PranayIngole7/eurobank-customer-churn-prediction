"""
Final model training and unseen-test evaluation.

The original test partition is used exclusively for evaluation.
The classification threshold is retained from validation-based
model selection.
"""

import numpy as np
from sklearn.metrics import average_precision_score

from src.data_split import split_dataset
from src.evaluation import generate_classification_report
from src.model_selection import MODEL_NAMES
from src.modeling import build_candidate_models, evaluate_model


def train_and_evaluate_final_model(
    processed_data,
    *,
    threshold: float,
    model_name: str,
) -> tuple:
    """
    Train the selected model on the original training partition
    and evaluate it on the untouched original test partition.

    Returns
    -------
    tuple
        Fitted model and evaluation results.
    """

    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError("threshold must be numeric and between 0 and 1.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be numeric and between 0 and 1.")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty candidate name.")

    models = build_candidate_models()
    if model_name not in models:
        raise ValueError(f"Unknown final model: {model_name}")

    # Reproduce the established stratified 80/20 split.
    X_train, X_test, y_train, y_test = split_dataset(processed_data)

    # Build a fresh instance of the explicitly selected model.
    model = models[model_name]

    # Fit exclusively on the original training partition.
    model.fit(X_train, y_train)

    # Evaluate using the previously selected threshold.
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold=threshold,
    )

    # Retrieve probabilities for the positive class (Exited = 1).
    classes = list(model.classes_)
    positive_class_index = classes.index(1)

    churn_probabilities = model.predict_proba(X_test)[
        :, positive_class_index
    ]

    # Derive predictions using the unchanged threshold.
    predictions = (churn_probabilities >= threshold).astype(int)

    # Calculate Average Precision separately from ROC-AUC.
    average_precision = average_precision_score(
        y_test,
        churn_probabilities,
    )

    # Generate the class-wise report using the same threshold.
    classification_results = generate_classification_report(
        y_test,
        predictions,
    )

    results = {
        "model_name": model_name,
        "algorithm": MODEL_NAMES[model_name],
        "threshold": float(threshold),
        "training_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "metrics": {
            **{
                key: float(value)
                for key, value in metrics.items()
                if key != "confusion_matrix"
            },
            "average_precision": float(average_precision),
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "classification_report": classification_results,
    }

    return model, results
