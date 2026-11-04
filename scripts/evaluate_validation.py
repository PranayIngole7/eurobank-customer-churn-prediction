"""
Generate actual validation evaluation artifacts.

The model is fitted only on the internal training split.
The validation split is used for evaluation.
The original test split is never evaluated by this script.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_split import split_dataset, split_training_data
from src.evaluation import (
    generate_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from src.feature_engineering import engineer_features
from src.modeling import build_baseline_models, evaluate_model


DATA_PATH = Path("data/raw/european-bank.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

SELECTED_THRESHOLD = 0.65


def main():
    # 1. Load and prepare the dataset.
    raw_data = pd.read_csv(DATA_PATH)
    processed_data = engineer_features(raw_data)

    # 2. Create the original training/test split.
    #    The original test data is deliberately discarded.
    X_train, _, y_train, _ = split_dataset(processed_data)

    # 3. Create internal training and validation splits.
    (
        X_internal_train,
        X_validation,
        y_internal_train,
        y_validation,
    ) = split_training_data(X_train, y_train)

    print("Dataset split")
    print("-------------")
    print(f"Internal training records: {len(X_internal_train)}")
    print(f"Validation records:        {len(X_validation)}")

    # 4. Build and fit the selected balanced Logistic Regression model.
    model = build_baseline_models()["logistic_regression_balanced"]

    model.fit(X_internal_train, y_internal_train)

    # 5. Generate validation probabilities and predictions.
    class_labels = list(model.classes_)
    positive_class_index = class_labels.index(1)

    probabilities = model.predict_proba(X_validation)[
        :, positive_class_index
    ]

    predictions = (
        probabilities >= SELECTED_THRESHOLD
    ).astype(int)

    # 6. Generate the classification report.
    classification_results = generate_classification_report(
        y_validation,
        predictions,
    )

    # 7. Generate evaluation metrics.
    evaluation_results = evaluate_model(
        model,
        X_validation,
        y_validation,
        threshold=SELECTED_THRESHOLD,
    )

    # 8. Generate actual validation visualizations.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_confusion_matrix(
        y_validation,
        predictions,
        FIGURES_DIR / "confusion-matrix-validation.png",
    )
    plt.close(fig)

    fig, _, roc_auc = plot_roc_curve(
        y_validation,
        probabilities,
        FIGURES_DIR / "roc-curve-validation.png",
    )
    plt.close(fig)

    fig, _, average_precision = plot_precision_recall_curve(
        y_validation,
        probabilities,
        FIGURES_DIR / "precision-recall-curve-validation.png",
    )
    plt.close(fig)

    # 9. Save the validation classification report and metrics.
    output = {
        "evaluation_dataset": "validation",
        "records": int(len(y_validation)),
        "threshold": SELECTED_THRESHOLD,
        "classification_report": classification_results,
        "accuracy": float(evaluation_results["accuracy"]),
        "precision": float(evaluation_results["precision"]),
        "recall": float(evaluation_results["recall"]),
        "f1_score": float(evaluation_results["f1_score"]),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "confusion_matrix": evaluation_results["confusion_matrix"],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "classification-report-validation.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    # 10. Display results.
    print("\nValidation Classification Report")
    print("--------------------------------")
    print(pd.DataFrame(classification_results).T.to_string())

    print("\nValidation Evaluation Metrics")
    print("-----------------------------")

    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ):
        print(f"{metric}: {output[metric]:.4f}")

    print(f"average_precision: {average_precision:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(evaluation_results["confusion_matrix"])

    print("\nGenerated Artifacts")
    print("-------------------")
    print(output_path)

    for filename in (
        "confusion-matrix-validation.png",
        "roc-curve-validation.png",
        "precision-recall-curve-validation.png",
    ):
        print(FIGURES_DIR / filename)

    print("\nValidation evaluation completed successfully.")


if __name__ == "__main__":
    main()
