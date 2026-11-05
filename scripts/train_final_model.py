"""
Train the final selected churn model and evaluate it on the
untouched original test partition.

The classification threshold was selected using validation data
and is not modified using original test results.
"""

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import engineer_features
from src.final_model_training import train_and_evaluate_final_model
from src.model_persistence import create_model_metadata, save_model


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "european-bank.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

JSON_OUTPUT_PATH = REPORTS_DIR / "final-model-test-evaluation.json"
MARKDOWN_OUTPUT_PATH = REPORTS_DIR / "final-model-evaluation.md"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_OUTPUT_PATH = MODEL_DIR / "churn_model.joblib"
METADATA_OUTPUT_PATH = MODEL_DIR / "model_metadata.json"

def generate_markdown_report(results: dict) -> str:
    """Generate a readable final-model evaluation report."""

    metrics = results["metrics"]
    confusion_matrix = metrics["confusion_matrix"]
    classification_report = results["classification_report"]

    report = [
        "# Final Model Evaluation",
        "",
        "## Model Information",
        "",
        f"- Model: {results['model_name']}",
        f"- Algorithm: {results['algorithm']}",
        f"- Classification threshold: {results['threshold']}",
        f"- Training records: {results['training_records']}",
        f"- Test records: {results['test_records']}",
        "",
        "## Evaluation Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
    ):
        report.append(f"| {metric.replace('_', ' ').title()} | {metrics[metric]:.4f} |")

    report.extend([
        "",
        "## Confusion Matrix",
        "",
        "| Actual / Predicted | Retained (0) | Churned (1) |",
        "|---|---:|---:|",
        f"| Retained (0) | {confusion_matrix[0][0]} | {confusion_matrix[0][1]} |",
        f"| Churned (1) | {confusion_matrix[1][0]} | {confusion_matrix[1][1]} |",
        "",
        "## Classification Report",
        "",
        "| Class | Precision | Recall | F1-score | Support |",
        "|---|---:|---:|---:|---:|",
    ])

    for class_name in ("Retained (0)", "Churned (1)"):
        class_metrics = classification_report[class_name]

        report.append(
            f"| {class_name} "
            f"| {class_metrics['precision']:.4f} "
            f"| {class_metrics['recall']:.4f} "
            f"| {class_metrics['f1-score']:.4f} "
            f"| {int(class_metrics['support'])} |"
        )

    report.extend([
        "",
        "## Evaluation Methodology",
        "",
        "- The original dataset was split using stratified 80/20 sampling.",
        "- The model was fitted exclusively on the original training partition.",
        "- The original test partition was used exclusively for final evaluation.",
        "- The classification threshold of 0.65 was selected using validation data.",
        "- The test results were not used to tune the model, threshold, or features.",
        "- Feature preprocessing was fitted as part of the training pipeline.",
        "",
        "## Limitations",
        "",
        "These metrics represent performance on one held-out test partition. "
        "They do not guarantee equivalent performance on future customer "
        "populations or production data.",
        "",
    ])

    return "\n".join(report)


def main():
    # 1. Load the original customer dataset.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    raw_data = pd.read_csv(DATA_PATH)

    print(f"Raw customer records: {len(raw_data)}")

    # 2. Apply the established feature engineering process.
    processed_data = engineer_features(raw_data)

    print(f"Feature-engineered records: {len(processed_data)}")

    # 3. Train and evaluate the final selected model.
    model, results = train_and_evaluate_final_model(
        processed_data
    )

    # Generate and persist model metadata and fitted model.
    metadata = create_model_metadata(
        model,
       results,
   )

    save_model(
        model,
        metadata,
        MODEL_OUTPUT_PATH,
        METADATA_OUTPUT_PATH,
    )

    print("\nModel persistence completed successfully.")
    print(f"Model artifact: {MODEL_OUTPUT_PATH}")
    print(f"Model metadata: {METADATA_OUTPUT_PATH}")

    # 4. Save machine-readable evaluation results.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with JSON_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # 5. Generate the human-readable evaluation report.
    markdown_report = generate_markdown_report(results)

    MARKDOWN_OUTPUT_PATH.write_text(
        markdown_report,
        encoding="utf-8",
    )

    # 6. Display evaluation results.
    print("\nFinal Model Evaluation")
    print("======================")

    print(f"Model: {results['model_name']}")
    print(f"Algorithm: {results['algorithm']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Training records: {results['training_records']}")
    print(f"Test records: {results['test_records']}")

    print("\nEvaluation Metrics")
    print("------------------")

    for metric, value in results["metrics"].items():
        if metric == "confusion_matrix":
            continue

        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(results["metrics"]["confusion_matrix"])

    print("\nClassification Report")
    print("---------------------")
    print(
        pd.DataFrame(
            results["classification_report"]
        ).T.to_string()
    )

    print("\nGenerated Artifacts")
    print("-------------------")
    print(JSON_OUTPUT_PATH)
    print(MARKDOWN_OUTPUT_PATH)
    print(MODEL_OUTPUT_PATH)
    print(METADATA_OUTPUT_PATH)

    print("\nFinal model training and evaluation completed successfully.")


if __name__ == "__main__":
    main()
