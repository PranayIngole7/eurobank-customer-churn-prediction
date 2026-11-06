"""Load and validate saved model and evaluation artifacts."""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
TEST_EVALUATION_PATH = REPORTS_DIR / "final-model-test-evaluation.json"
THRESHOLD_COMPARISON_PATH = REPORTS_DIR / "threshold-comparison.csv"

REQUIRED_THRESHOLD_COLUMNS = {
    "threshold",
    "precision",
    "recall",
    "f1_score",
    "tn",
    "fp",
    "fn",
    "tp",
}

REQUIRED_TEST_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "average_precision",
    "confusion_matrix",
}


def _load_json(path):
    """Load a JSON artifact and ensure it contains an object."""
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")

    return data


def load_model_metadata():
    """Load and validate persisted model metadata."""
    metadata = _load_json(MODEL_METADATA_PATH)

    required_fields = {
        "model_name",
        "algorithm",
        "threshold",
        "training_records",
        "feature_columns",
    }

    missing = required_fields - metadata.keys()
    if missing:
        raise ValueError(
            f"Model metadata is missing fields: {sorted(missing)}"
        )

    threshold = metadata["threshold"]

    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Model threshold must be a number between 0 and 1.")

    if not isinstance(metadata["feature_columns"], list):
        raise ValueError("Model feature_columns must be a list.")

    return metadata


def load_test_evaluation():
    """Load and validate held-out test evaluation results."""
    evaluation = _load_json(TEST_EVALUATION_PATH)

    required_fields = {
        "model_name",
        "algorithm",
        "threshold",
        "training_records",
        "test_records",
        "metrics",
    }

    missing = required_fields - evaluation.keys()
    if missing:
        raise ValueError(
            f"Test evaluation is missing fields: {sorted(missing)}"
        )

    metrics = evaluation["metrics"]

    if not isinstance(metrics, dict):
        raise ValueError("Test evaluation metrics must be a JSON object.")

    missing_metrics = REQUIRED_TEST_METRICS - metrics.keys()
    if missing_metrics:
        raise ValueError(
            f"Test evaluation is missing metrics: {sorted(missing_metrics)}"
        )

    matrix = metrics["confusion_matrix"]

    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
    ):
        raise ValueError("Confusion matrix must be a 2x2 list.")

    return evaluation


def load_threshold_comparison():
    """Load and validate validation-based threshold comparison results."""
    if not THRESHOLD_COMPARISON_PATH.is_file():
        raise FileNotFoundError(
            f"Required artifact not found: {THRESHOLD_COMPARISON_PATH}"
        )

    try:
        data = pd.read_csv(THRESHOLD_COMPARISON_PATH)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Could not read threshold comparison CSV: {exc}"
        ) from exc

    missing = REQUIRED_THRESHOLD_COLUMNS - set(data.columns)

    if missing:
        raise ValueError(
            f"Threshold comparison CSV is missing columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("Threshold comparison CSV contains no records.")

    for column in REQUIRED_THRESHOLD_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if not data["threshold"].between(0, 1).all():
        raise ValueError("Threshold values must be between 0 and 1.")

    return data