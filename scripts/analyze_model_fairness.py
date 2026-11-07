"""
Analyze model fairness across customer groups.

The script reproduces the existing validation workflow:
- Original train/test split
- Internal training/validation split
- Balanced Logistic Regression
- Probability threshold of 0.65

The original held-out test set is never used.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_split import split_dataset, split_training_data
from src.fairness import (
    calculate_fairness_summary,
    calculate_group_metrics,
)
from src.feature_engineering import engineer_features
from src.modeling import build_baseline_models


DATA_PATH = Path("data/raw/european-bank.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

SELECTED_THRESHOLD = 0.65

AUDIT_ATTRIBUTES = [
    "Gender",
    "Geography",
    "AgeGroup",
]


def make_json_safe(value):
    """Convert NumPy and pandas values into JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)

    if pd.isna(value):
        return None

    return value


def generate_fairness_plot(all_group_metrics):
    """
    Generate group comparisons with 95% Wilson confidence intervals.

    Metrics:
    - Predicted churn rate
    - True positive rate
    - False positive rate
    """

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(AUDIT_ATTRIBUTES),
        figsize=(18, 7),
    )

    metrics = [
        ("predicted_churn_rate", "predicted_churn_rate"),
        ("tpr", "tpr"),
        ("fpr", "fpr"),
    ]

    metric_labels = [
        "Predicted churn rate",
        "True positive rate",
        "False positive rate",
    ]

    for axis, attribute in zip(axes, AUDIT_ATTRIBUTES):
        group_metrics = all_group_metrics[attribute]

        x = np.arange(len(group_metrics))
        width = 0.25

        for index, ((metric, ci_prefix), label) in enumerate(
            zip(metrics, metric_labels)
        ):
            positions = x + (index - 1) * width

            rates = group_metrics[metric].to_numpy(dtype=float)

            lower = group_metrics[
                f"{ci_prefix}_ci_lower"
            ].to_numpy(dtype=float)

            upper = group_metrics[
                f"{ci_prefix}_ci_upper"
            ].to_numpy(dtype=float)

            valid = (
                np.isfinite(rates)
                & np.isfinite(lower)
                & np.isfinite(upper)
            )

            lower_errors = np.maximum(
            rates[valid] - lower[valid],
            0.0,
            )

            upper_errors = np.maximum(
            upper[valid] - rates[valid],
            0.0,
            )

            y_errors = np.vstack([
            lower_errors,
            upper_errors,
            ])

            axis.bar(
                positions[valid],
                rates[valid],
                width,
                label=label,
                yerr=y_errors,
                capsize=3,
            )

        axis.set_title(attribute)
        axis.set_xlabel("Customer group")
        axis.set_ylabel("Rate with 95% Wilson CI")
        axis.set_ylim(0, 1.08)

        axis.set_xticks(x)
        axis.set_xticklabels(
            group_metrics.index.astype(str),
            rotation=30,
            ha="right",
        )

        axis.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Customer Churn Model Fairness Comparison "
        "with 95% Confidence Intervals",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = FIGURES_DIR / "fairness-comparison.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path



def main():
    # 1. Load and prepare the dataset.
    raw_data = pd.read_csv(DATA_PATH)
    processed_data = engineer_features(raw_data)

    # 2. Reproduce the original training/test split.
    #    The original test data is deliberately discarded.
    X_train, _, y_train, _ = split_dataset(processed_data)

    # 3. Reproduce the internal training/validation split.
    (
        X_internal_train,
        X_validation,
        y_internal_train,
        y_validation,
    ) = split_training_data(X_train, y_train)

    print("Validation dataset")
    print("------------------")
    print(f"Internal training records: {len(X_internal_train)}")
    print(f"Validation records:        {len(X_validation)}")
    print(f"Threshold:                 {SELECTED_THRESHOLD}")

    # 4. Fit the exact existing model.
    model = build_baseline_models()["logistic_regression_balanced"]

    model.fit(X_internal_train, y_internal_train)

    # 5. Generate validation probabilities and predictions.
    class_labels = list(model.classes_)
    positive_class_index = class_labels.index(1)

    probabilities = model.predict_proba(X_validation)[
        :, positive_class_index
    ]

    predictions = pd.Series(
        (probabilities >= SELECTED_THRESHOLD).astype(int),
        index=X_validation.index,
        name="prediction",
    )

    # 6. Calculate group metrics for every audit attribute.
    all_group_metrics = {}
    fairness_summaries = {}
    combined_results = []

    for attribute in AUDIT_ATTRIBUTES:
        print(f"\nAnalyzing attribute: {attribute}")

        # Retrieve group labels using the original validation indexes.
        validation_groups = processed_data.loc[
            X_validation.index,
            attribute,
        ]

        if not validation_groups.index.equals(y_validation.index):
            raise ValueError(
                f"Validation index mismatch for {attribute}."
            )

        group_metrics = calculate_group_metrics(
            y_validation,
            predictions,
            validation_groups,
        )

        fairness_summary = calculate_fairness_summary(
            group_metrics
        )

        all_group_metrics[attribute] = group_metrics
        fairness_summaries[attribute] = fairness_summary

        print("\nGroup-wise performance")
        print(group_metrics.to_string(float_format=lambda x: f"{x:.4f}"))

        print("\nFairness summary")

        for metric, value in fairness_summary.items():
            if isinstance(value, (int, np.integer)):
                print(f"{metric}: {value}")
            elif pd.isna(value):
                print(f"{metric}: undefined")
            else:
                print(f"{metric}: {value:.4f}")

        # Prepare combined CSV records.
        for group_name, row in group_metrics.iterrows():
            record = {
                "attribute": attribute,
                "group": group_name,
                **row.to_dict(),
            }
            combined_results.append(record)

    # 7. Save group-wise metrics as CSV.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORTS_DIR / "fairness-group-metrics.csv"

    combined_dataframe = pd.DataFrame(combined_results)
    combined_dataframe.to_csv(csv_path, index=False)

    # 8. Save fairness summaries and group metrics as JSON.
    output = {
        "evaluation_dataset": "validation",
        "records": int(len(y_validation)),
        "model": "logistic_regression_balanced",
        "threshold": SELECTED_THRESHOLD,
        "audit_attributes": AUDIT_ATTRIBUTES,
        "confidence_intervals": {
            "method": "Wilson score interval",
            "confidence_level": 0.95,
            "metrics": [
                "actual_churn_rate",
                "predicted_churn_rate",
                "tpr",
                "fpr",
                ],
            },
        "attributes": {},
    }

    for attribute in AUDIT_ATTRIBUTES:
        output["attributes"][attribute] = {
            "group_metrics": (
                all_group_metrics[attribute]
                .reset_index()
                .to_dict(orient="records")
            ),
            "fairness_summary": fairness_summaries[attribute],
        }

    output_path = REPORTS_DIR / "fairness-summary.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_safe(output),
            file,
            indent=4,
            allow_nan=False,
        )

    # 9. Generate the fairness visualization.
    figure_path = generate_fairness_plot(all_group_metrics)

    # 10. Display generated artifacts.
    print("\nGenerated Artifacts")
    print("-------------------")
    print(csv_path)
    print(output_path)
    print(figure_path)

    print("\nFairness analysis completed successfully.")


if __name__ == "__main__":
    main()