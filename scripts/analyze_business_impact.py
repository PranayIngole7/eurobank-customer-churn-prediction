"""
Analyze the operational business impact of classification thresholds.

The analysis uses validation-set threshold comparison results.
It does not estimate financial savings or actual retention costs.
"""

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("reports/threshold-comparison.csv")
OUTPUT_PATH = Path("reports/business-impact-analysis.md")

SELECTED_THRESHOLDS = [0.30, 0.50, 0.65, 0.80]


def load_threshold_results():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Threshold results not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "tn",
        "fp",
        "fn",
        "tp",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if data.empty:
        raise ValueError("Threshold comparison data is empty.")

    if data["threshold"].duplicated().any():
        raise ValueError("Duplicate thresholds found.")

    count_columns = ["tn", "fp", "fn", "tp"]

    if (data[count_columns] < 0).any().any():
        raise ValueError("Confusion matrix counts cannot be negative.")

    return data


def calculate_business_metrics(data):
    selected = data[
        data["threshold"].apply(
            lambda value: any(
                np.isclose(value, threshold)
                for threshold in SELECTED_THRESHOLDS
            )
        )
    ].copy()

    if len(selected) != len(SELECTED_THRESHOLDS):
        raise ValueError(
            "Could not find all requested thresholds."
        )

    selected = selected.sort_values("threshold")

    total_customers = (
        selected.iloc[0][["tn", "fp", "fn", "tp"]].sum()
    )

    actual_churners = (
        selected.iloc[0]["fn"] + selected.iloc[0]["tp"]
    )

    results = []

    for _, row in selected.iterrows():
        tn = int(row["tn"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        tp = int(row["tp"])

        flagged = tp + fp

        results.append({
            "threshold": row["threshold"],
            "total_customers": int(total_customers),
            "customers_flagged": flagged,
            "genuine_churners_detected": tp,
            "false_alerts": fp,
            "missed_churners": fn,
            "intervention_rate": flagged / total_customers,
            "churn_detection_rate": tp / actual_churners,
            "false_discovery_rate": (
                fp / flagged if flagged else 0
            ),
            "precision": tp / flagged if flagged else 0,
            "recall": tp / actual_churners,
            "f1_score": row["f1_score"],
        })

    return pd.DataFrame(results)


def generate_report(results):
    selected = results[
        np.isclose(results["threshold"], 0.65)
    ].iloc[0]

    lines = [
        "# Business Impact Analysis",
        "",
        "## 1. Objective",
        "",
        "Evaluate how classification thresholds affect the "
        "operational workload of a hypothetical customer "
        "retention team.",
        "",
        "This analysis uses validation data only. The original "
        "test dataset remains reserved for final evaluation.",
        "",
        "## 2. Dataset Overview",
        "",
        f"- Validation customers: {int(selected['total_customers']):,}",
        f"- Actual churners: {int(selected['genuine_churners_detected'] + selected['missed_churners']):,}",
        f"- Actual retained customers: {int(selected['total_customers'] - selected['genuine_churners_detected'] - selected['missed_churners']):,}",
        "",
        "## 3. Threshold Scenario Comparison",
        "",
        "| Threshold | Customers Flagged | Genuine Churners Detected | False Alerts | Missed Churners |",
        "|---:|---:|---:|---:|---:|",
    ]

    for _, row in results.iterrows():
        lines.append(
            f"| {row['threshold']:.2f} "
            f"| {int(row['customers_flagged']):,} "
            f"| {int(row['genuine_churners_detected']):,} "
            f"| {int(row['false_alerts']):,} "
            f"| {int(row['missed_churners']):,} |"
        )

    lines.extend([
        "",
        "## 4. Operational Metrics",
        "",
        "| Threshold | Intervention Rate | Churn Detection Rate | False Discovery Rate | Precision | Recall | F1-score |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for _, row in results.iterrows():
        lines.append(
            f"| {row['threshold']:.2f} "
            f"| {row['intervention_rate']:.2%} "
            f"| {row['churn_detection_rate']:.2%} "
            f"| {row['false_discovery_rate']:.2%} "
            f"| {row['precision']:.4f} "
            f"| {row['recall']:.4f} "
            f"| {row['f1_score']:.4f} |"
        )

    lines.extend([
        "",
        "## 5. Selected Threshold: 0.65",
        "",
        f"At threshold 0.65, the model flags "
        f"**{int(selected['customers_flagged']):,} customers** "
        f"for potential retention intervention.",
        "",
        f"- Genuine churners detected: {int(selected['genuine_churners_detected']):,}",
        f"- False alerts: {int(selected['false_alerts']):,}",
        f"- Missed churners: {int(selected['missed_churners']):,}",
        f"- Intervention rate: {selected['intervention_rate']:.2%}",
        f"- Churn detection rate: {selected['churn_detection_rate']:.2%}",
        f"- False discovery rate: {selected['false_discovery_rate']:.2%}",
        "",
        "The threshold was selected using validation F1-score. "
        "It is not claimed to be the financially optimal threshold.",
        "",
        "## 6. Business Interpretation",
        "",
        "- A lower threshold identifies more churners but increases "
        "the number of customers requiring intervention.",
        "- A higher threshold reduces false alerts and intervention "
        "workload but misses more genuine churners.",
        "- The selected threshold represents a balance between "
        "precision and recall according to validation F1-score.",
        "- The appropriate operational threshold ultimately depends "
        "on retention-team capacity and the relative costs of false "
        "alerts and missed churners.",
        "",
        "## 7. Financial Impact Limitations",
        "",
        "Actual retention campaign costs, customer lifetime value, "
        "churn-related financial losses, and campaign success rates "
        "were not provided.",
        "",
        "Therefore, this report does not claim actual financial "
        "savings, revenue protection, or return on investment.",
        "Any future financial analysis must use documented business "
        "assumptions or verified company data.",
        "",
        "## 8. Data Integrity",
        "",
        "- Source: `reports/threshold-comparison.csv`.",
        "- Dataset: 2,000 validation records.",
        "- Threshold selection: validation F1-score.",
        "- Original 2,000-record test set: not used.",
        "",
    ])

    return "\n".join(lines)


def main():
    threshold_data = load_threshold_results()

    results = calculate_business_metrics(threshold_data)

    report = generate_report(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("\nBusiness impact analysis:")
    print(
        results.to_string(
            index=False,
            formatters={
                "intervention_rate": "{:.2%}".format,
                "churn_detection_rate": "{:.2%}".format,
                "false_discovery_rate": "{:.2%}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
            },
        )
    )

    print("\nReport generated:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
