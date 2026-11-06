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

FEATURE_COEFFICIENTS_PATH = REPORTS_DIR / "feature-coefficients.csv"

PERMUTATION_IMPORTANCE_PATH = (
    REPORTS_DIR / "permutation-importance.csv"
)

FAIRNESS_GROUP_METRICS_PATH = (
    REPORTS_DIR / "fairness-group-metrics.csv"
)

FAIRNESS_SUMMARY_PATH = REPORTS_DIR / "fairness-summary.json"

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

REQUIRED_COEFFICIENT_COLUMNS = {
    "feature",
    "coefficient",
    "odds_ratio",
    "absolute_coefficient",
    "direction",
}

REQUIRED_PERMUTATION_COLUMNS = {
    "feature",
    "importance_mean",
    "importance_std",
}

REQUIRED_FAIRNESS_COLUMNS = {
    "attribute",
    "group",
    "sample_count",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "actual_positive_count",
    "actual_negative_count",
    "predicted_positive_count",
    "actual_churn_rate",
    "predicted_churn_rate",
    "accuracy",
    "precision",
    "tpr",
    "fpr",
    "fnr",
    "actual_churn_rate_ci_lower",
    "actual_churn_rate_ci_upper",
    "predicted_churn_rate_ci_lower",
    "predicted_churn_rate_ci_upper",
    "tpr_ci_lower",
    "tpr_ci_upper",
    "fpr_ci_lower",
    "fpr_ci_upper",
}

FAIRNESS_ATTRIBUTES = {
    "Gender",
    "Geography",
    "AgeGroup",
}

REQUIRED_CLASSIFICATION_CLASSES = {
    "Retained (0)",
    "Churned (1)",
    "macro avg",
    "weighted avg",
}

REQUIRED_CLASSIFICATION_METRICS = {
    "precision",
    "recall",
    "f1-score",
    "support",
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

def _load_csv_artifact(path, required_columns, artifact_name):
    """Load a CSV artifact and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact not found: {path}"
        )

    try:
        data = pd.read_csv(path)
    except (
        pd.errors.ParserError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
    ) as exc:
        raise ValueError(
            f"Could not read {artifact_name}: {exc}"
        ) from exc

    if data.empty:
        raise ValueError(
            f"{artifact_name} contains no records."
        )

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"{artifact_name} is missing columns: "
            f"{sorted(missing)}"
        )

    return data

def load_feature_coefficients():
    """Load and validate logistic regression feature coefficients."""
    data = _load_csv_artifact(
        FEATURE_COEFFICIENTS_PATH,
        REQUIRED_COEFFICIENT_COLUMNS,
        "Feature coefficients",
    )

    numeric_columns = {
        "coefficient",
        "odds_ratio",
        "absolute_coefficient",
    }

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if data[list(numeric_columns)].isna().any().any():
        raise ValueError(
            "Feature coefficients contain missing numeric values."
        )

    if not data[list(numeric_columns)].apply(
        lambda column: column.map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        )
    ).all().all():
        raise ValueError(
            "Feature coefficients contain non-finite values."
        )

    if data["feature"].isna().any() or data["feature"].duplicated().any():
        raise ValueError(
            "Feature coefficients contain missing or duplicate features."
        )

    if data["direction"].isna().any():
        raise ValueError(
            "Feature coefficients contain missing directions."
        )

    if (data["odds_ratio"] <= 0).any():
        raise ValueError("Odds ratios must be positive.")

    if (data["absolute_coefficient"] < 0).any():
        raise ValueError(
            "Absolute coefficients cannot be negative."
        )

    return data

def load_permutation_importance():
    """Load and validate validation-based permutation importance."""
    data = _load_csv_artifact(
        PERMUTATION_IMPORTANCE_PATH,
        REQUIRED_PERMUTATION_COLUMNS,
        "Permutation importance",
    )

    numeric_columns = [
        "importance_mean",
        "importance_std",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if data[numeric_columns].isna().any().any():
        raise ValueError(
            "Permutation importance contains missing numeric values."
        )

    if not data[numeric_columns].apply(
        lambda column: column.map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        )
    ).all().all():
        raise ValueError(
            "Permutation importance contains non-finite values."
        )

    if data["feature"].isna().any() or data["feature"].duplicated().any():
        raise ValueError(
            "Permutation importance contains missing or duplicate features."
        )

    if (data["importance_std"] < 0).any():
        raise ValueError(
            "Permutation importance standard deviations cannot be negative."
        )

    return data

def load_fairness_group_metrics():
    """Load and validate validation-based fairness group metrics."""
    data = _load_csv_artifact(
        FAIRNESS_GROUP_METRICS_PATH,
        REQUIRED_FAIRNESS_COLUMNS,
        "Fairness group metrics",
    )

    count_columns = [
        "sample_count",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "actual_positive_count",
        "actual_negative_count",
        "predicted_positive_count",
    ]

    rate_columns = [
        "actual_churn_rate",
        "predicted_churn_rate",
        "accuracy",
        "precision",
        "tpr",
        "fpr",
        "fnr",
        "actual_churn_rate_ci_lower",
        "actual_churn_rate_ci_upper",
        "predicted_churn_rate_ci_lower",
        "predicted_churn_rate_ci_upper",
        "tpr_ci_lower",
        "tpr_ci_upper",
        "fpr_ci_lower",
        "fpr_ci_upper",
    ]

    numeric_columns = count_columns + rate_columns

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    required_numeric = data[
        count_columns + [column for column in rate_columns if column != "precision"]
    ]

    if required_numeric.isna().any().any():
        raise ValueError(
            "Fairness group metrics contain missing numeric values."
        )

    if not data[numeric_columns].apply(
        lambda column: column.map(
            lambda value: pd.isna(value) or abs(value) != float("inf")
        )
    ).all().all():
        raise ValueError(
            "Fairness group metrics contain non-finite values."
        )

    if (data[count_columns] < 0).any().any():
        raise ValueError("Fairness group counts cannot be negative.")

    if (data[rate_columns].drop(columns=["precision"]) < 0).any().any():
        raise ValueError("Fairness rates cannot be negative.")

    if (data[rate_columns].drop(columns=["precision"]) > 1).any().any():
        raise ValueError("Fairness rates cannot exceed 1.")

    if data["attribute"].isna().any() or data["group"].isna().any():
        raise ValueError(
            "Fairness group metrics contain missing attribute or group names."
        )

    if not set(data["attribute"]).issubset(FAIRNESS_ATTRIBUTES):
        raise ValueError(
            "Fairness group metrics contain unsupported audit attributes."
        )

    if data.duplicated(subset=["attribute", "group"]).any():
        raise ValueError(
            "Fairness group metrics contain duplicate attribute/group pairs."
        )

    confidence_intervals = [
        ("actual_churn_rate_ci_lower", "actual_churn_rate_ci_upper"),
        ("predicted_churn_rate_ci_lower", "predicted_churn_rate_ci_upper"),
        ("tpr_ci_lower", "tpr_ci_upper"),
        ("fpr_ci_lower", "fpr_ci_upper"),
    ]

    for lower, upper in confidence_intervals:
        if (data[lower] > data[upper]).any():
            raise ValueError(
                f"Invalid confidence interval: {lower} exceeds {upper}."
            )

    return data

def load_fairness_summary():
    """Load and validate the saved fairness analysis summary."""
    summary = _load_json(FAIRNESS_SUMMARY_PATH)

    required_fields = {
        "evaluation_dataset",
        "records",
        "model",
        "threshold",
        "audit_attributes",
        "confidence_intervals",
        "attributes",
    }

    missing = required_fields - summary.keys()

    if missing:
        raise ValueError(
            f"Fairness summary is missing fields: {sorted(missing)}"
        )

    if summary["evaluation_dataset"] != "validation":
        raise ValueError(
            "Fairness summary must use the validation dataset."
        )

    if (
        not isinstance(summary["records"], int)
        or summary["records"] <= 0
    ):
        raise ValueError(
            "Fairness summary records must be a positive integer."
        )

    threshold = summary["threshold"]

    if (
        not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        raise ValueError(
            "Fairness summary threshold must be between 0 and 1."
        )

    audit_attributes = summary["audit_attributes"]

    if (
        not isinstance(audit_attributes, list)
        or set(audit_attributes) != FAIRNESS_ATTRIBUTES
    ):
        raise ValueError(
            "Fairness summary must contain Gender, Geography, and AgeGroup."
        )

    confidence = summary["confidence_intervals"]

    if not isinstance(confidence, dict):
        raise ValueError(
            "Confidence interval metadata must be an object."
        )

    if confidence.get("method") != "Wilson score interval":
        raise ValueError(
            "Unexpected fairness confidence interval method."
        )

    if confidence.get("confidence_level") != 0.95:
        raise ValueError(
            "Fairness confidence level must be 0.95."
        )

    attributes = summary["attributes"]

    if (
        not isinstance(attributes, dict)
        or set(attributes) != FAIRNESS_ATTRIBUTES
    ):
        raise ValueError(
            "Fairness summary has invalid audit attribute entries."
        )

    required_summary_metrics = {
        "demographic_parity_difference",
        "equal_opportunity_difference",
        "false_positive_rate_difference",
        "equalized_odds_difference",
        "disparate_impact_ratio",
    }

    for attribute, details in attributes.items():
        if not isinstance(details, dict):
            raise ValueError(
                f"Invalid fairness summary for {attribute}."
            )

        if "group_metrics" not in details or "fairness_summary" not in details:
            raise ValueError(
                f"Fairness summary for {attribute} is incomplete."
            )

        groups = details["group_metrics"]

        if not isinstance(groups, list) or not groups:
            raise ValueError(
                f"Fairness groups for {attribute} must be a non-empty list."
            )

        if not all(isinstance(group, dict) for group in groups):
            raise ValueError(
                f"Fairness groups for {attribute} must contain objects."
            )

        if not all(
            isinstance(group.get("group"), str)
            and group["group"].strip()
            for group in groups
        ):
            raise ValueError(
                f"Fairness groups for {attribute} contain invalid group names."
            )

        group_names = [group["group"] for group in groups]

        if len(group_names) != len(set(group_names)):
            raise ValueError(
                f"Fairness groups for {attribute} contain duplicate names."
            )

        metrics = details["fairness_summary"]

        if not isinstance(metrics, dict):
            raise ValueError(
                f"Fairness indicators for {attribute} must be an object."
            )

        number_of_groups = metrics.get("number_of_groups")

        if (
            not isinstance(number_of_groups, int)
            or isinstance(number_of_groups, bool)
            or number_of_groups <= 0
        ):
            raise ValueError(
                f"Invalid number_of_groups for {attribute}."
            )

        if number_of_groups != len(groups):
            raise ValueError(
                f"Fairness group count mismatch for {attribute}."
            )

        missing_metrics = required_summary_metrics - metrics.keys()

        if missing_metrics:
            raise ValueError(
                f"Fairness summary for {attribute} is missing metrics: "
                f"{sorted(missing_metrics)}"
            )

        for metric in required_summary_metrics:
            value = metrics[metric]

            if (
                not isinstance(value, (int, float))
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"Invalid {metric} value for {attribute}."
                )

    return summary

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
        "classification_report",
    }

    missing = required_fields - evaluation.keys()

    if missing:
        raise ValueError(
            f"Test evaluation is missing fields: {sorted(missing)}"
        )

    if not isinstance(evaluation["model_name"], str) or not evaluation["model_name"]:
        raise ValueError("Test evaluation model_name must be a non-empty string.")

    if not isinstance(evaluation["algorithm"], str) or not evaluation["algorithm"]:
        raise ValueError("Test evaluation algorithm must be a non-empty string.")

    threshold = evaluation["threshold"]

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Test evaluation threshold must be between 0 and 1.")

    for field in ("training_records", "test_records"):
        value = evaluation[field]

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"Test evaluation {field} must be a positive integer."
            )

    metrics = evaluation["metrics"]

    if not isinstance(metrics, dict):
        raise ValueError("Test evaluation metrics must be a JSON object.")

    missing_metrics = REQUIRED_TEST_METRICS - metrics.keys()

    if missing_metrics:
        raise ValueError(
            f"Test evaluation is missing metrics: {sorted(missing_metrics)}"
        )

    for metric in REQUIRED_TEST_METRICS - {"confusion_matrix"}:
        value = metrics[metric]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not pd.notna(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"Invalid test evaluation metric: {metric}."
            )

    matrix = metrics["confusion_matrix"]

    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(
            not isinstance(row, list) or len(row) != 2
            for row in matrix
        )
    ):
        raise ValueError("Confusion matrix must be a 2x2 list.")

    matrix_values = [value for row in matrix for value in row]

    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in matrix_values
    ):
        raise ValueError(
            "Confusion matrix must contain non-negative integer counts."
        )

    if sum(matrix_values) != evaluation["test_records"]:
        raise ValueError(
            "Confusion matrix counts do not match test_records."
        )

    classification_report = evaluation["classification_report"]

    if not isinstance(classification_report, dict):
        raise ValueError(
            "Classification report must be a JSON object."
        )

    missing_classes = (
        REQUIRED_CLASSIFICATION_CLASSES
        - classification_report.keys()
    )

    if missing_classes:
        raise ValueError(
            "Classification report is missing entries: "
            f"{sorted(missing_classes)}"
        )

    for class_name in REQUIRED_CLASSIFICATION_CLASSES:
        class_metrics = classification_report[class_name]

        if not isinstance(class_metrics, dict):
            raise ValueError(
                f"Classification report entry {class_name} "
                "must be an object."
            )

        missing_class_metrics = (
            REQUIRED_CLASSIFICATION_METRICS - class_metrics.keys()
        )

        if missing_class_metrics:
            raise ValueError(
                f"Classification report entry {class_name} "
                f"is missing metrics: {sorted(missing_class_metrics)}"
            )

        for metric in REQUIRED_CLASSIFICATION_METRICS:
            value = class_metrics[metric]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not pd.notna(value)
                or value < 0
                or (
                    metric != "support"
                    and value > 1
                )
            ):
                raise ValueError(
                    f"Invalid {metric} for {class_name}."
                )

    accuracy = classification_report.get("accuracy")

    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not 0 <= accuracy <= 1
    ):
        raise ValueError(
            "Classification report accuracy must be between 0 and 1."
        )

    if classification_report["Retained (0)"]["support"] + (
        classification_report["Churned (1)"]["support"]
    ) != evaluation["test_records"]:
        raise ValueError(
            "Classification report class support does not match test_records."
        )

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