"""Compare candidate models and select a threshold using validation data only."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_split import split_dataset
from src.evaluation import (
    generate_classification_report,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from src.feature_engineering import engineer_features
from src.model_selection import run_validation_selection
from src.modeling import evaluate_model


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "european-bank.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    processed_data = engineer_features(pd.read_csv(DATA_PATH))

    # The held-out test partition is separated and discarded before selection.
    X_train, _, y_train, _ = split_dataset(processed_data)
    selection_results = run_validation_selection(X_train, y_train)

    selection = selection_results["selection"]
    model = selection_results["models"][selection["model_name"]]
    X_validation = selection_results["X_validation"]
    y_validation = selection_results["y_validation"]
    threshold = selection["threshold"]

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        threshold=threshold,
    )
    probabilities = selection_results["validation_probabilities"][
        selection["model_name"]
    ]
    predictions = (probabilities >= threshold).astype(int)
    classification_report = generate_classification_report(
        y_validation,
        predictions,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    comparison = selection_results["comparison"]
    comparison.to_csv(
        REPORTS_DIR / "model-comparison-validation.csv",
        index=False,
    )
    all_thresholds = selection_results["threshold_comparison"]
    all_thresholds.to_csv(
        REPORTS_DIR / "model-threshold-comparison-validation.csv",
        index=False,
    )
    selected_thresholds = all_thresholds[
        all_thresholds["model_name"] == selection["model_name"]
    ].drop(columns=["model_name", "average_precision", "roc_auc"])
    selected_thresholds.to_csv(
        REPORTS_DIR / "threshold-comparison.csv",
        index=False,
    )

    selected_validation = {
        "evaluation_dataset": "validation",
        "selected_model": selection["model_name"],
        "algorithm": selection["algorithm"],
        "threshold": threshold,
        "records": int(len(y_validation)),
        "selection_rule": selection["selection_rule"],
        "metrics": {
            key: (
                value
                if key == "confusion_matrix"
                else float(value)
            )
            for key, value in metrics.items()
        },
        "classification_report": classification_report,
    }
    (REPORTS_DIR / "classification-report-validation.json").write_text(
        json.dumps(selected_validation, indent=2),
        encoding="utf-8",
    )

    selection_report = {
        "evaluation_dataset": "validation",
        "validation_records": int(len(y_validation)),
        "candidate_metrics_at_threshold_0_50": comparison.to_dict(
            orient="records"
        ),
        "selected_model": selection,
    }
    (REPORTS_DIR / "model-selection-validation.json").write_text(
        json.dumps(selection_report, indent=2),
        encoding="utf-8",
    )

    fig, _ = plot_confusion_matrix(
        y_validation,
        predictions,
        FIGURES_DIR / "confusion-matrix-validation.png",
    )
    plt.close(fig)
    fig, _, _ = plot_roc_curve(
        y_validation,
        probabilities,
        FIGURES_DIR / "roc-curve-validation.png",
    )
    plt.close(fig)
    fig, _, _ = plot_precision_recall_curve(
        y_validation,
        probabilities,
        FIGURES_DIR / "precision-recall-curve-validation.png",
    )
    plt.close(fig)

    print("Validation model comparison")
    print(comparison.to_string(index=False))
    print("\nSelected model and threshold")
    print(json.dumps(selection, indent=2))
    print("\nHeld-out test records were not evaluated or used for selection.")


if __name__ == "__main__":
    main()
