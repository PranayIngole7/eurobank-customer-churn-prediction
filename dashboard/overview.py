"""Dashboard overview page."""

import pandas as pd
import streamlit as st

from dashboard.components import render_header
from dashboard.data_access import (
    load_model_metadata,
    load_test_evaluation,
    load_threshold_comparison,
)


def _format_percentage(value):
    """Format a decimal metric as a percentage."""
    return f"{value:.1%}"


def render_overview():
    """Render the model summary and evaluation overview."""

    render_header(
        "Customer Churn Dashboard",
        "Monitor the selected churn model, held-out test performance, "
        "and validation-based threshold analysis.",
    )

    # Load saved artifacts.
    try:
        metadata = load_model_metadata()
        evaluation = load_test_evaluation()
        thresholds = load_threshold_comparison()
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Unable to load dashboard data: {exc}")
        return

    metrics = evaluation["metrics"]
    selected_threshold = metadata["threshold"]

    # ---------------------------------------------------------
    # 1. Model summary
    # ---------------------------------------------------------
    st.header("Model Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = (
        st.columns(4)
    )

    with summary_col1:
        st.metric("Algorithm", metadata["algorithm"])

    with summary_col2:
        st.metric("Selected Threshold", f"{selected_threshold:.2f}")

    with summary_col3:
        st.metric(
            "Training Records",
            f"{metadata['training_records']:,}",
        )

    with summary_col4:
        st.metric(
            "Input Features",
            len(metadata["feature_columns"]),
        )

    st.divider()

    # ---------------------------------------------------------
    # 2. Held-out test performance
    # ---------------------------------------------------------
    st.header("Held-out Test Performance")

    st.caption(
        f"Evaluation on {evaluation['test_records']:,} previously "
        "unseen test records. These metrics are not used to tune "
        "the model or select its threshold."
    )

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = (
        st.columns(5)
    )

    with metric_col1:
        st.metric("Accuracy", _format_percentage(metrics["accuracy"]))

    with metric_col2:
        st.metric("Precision", _format_percentage(metrics["precision"]))

    with metric_col3:
        st.metric("Recall", _format_percentage(metrics["recall"]))

    with metric_col4:
        st.metric("F1 Score", _format_percentage(metrics["f1_score"]))

    with metric_col5:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.caption(
        "Precision measures how many predicted churners actually churned. "
        "Recall measures how many actual churners the model detected."
    )

    st.divider()

    # ---------------------------------------------------------
    # 3. Confusion matrix
    # ---------------------------------------------------------
    st.header("Confusion Matrix")

    st.caption(
        "Held-out test results. Rows represent actual outcomes; "
        "columns represent model predictions."
    )

    matrix = metrics["confusion_matrix"]

    confusion_matrix = pd.DataFrame(
        matrix,
        index=["Actual Retained", "Actual Churned"],
        columns=["Predicted Retained", "Predicted Churned"],
    )

    matrix_col1, matrix_col2 = st.columns([2, 3])

    with matrix_col1:
        st.table(confusion_matrix)

    with matrix_col2:
        tn, fp = matrix[0]
        fn, tp = matrix[1]

        st.metric("Correctly Identified Retained Customers", f"{tn:,}")
        st.metric("Correctly Detected Churners", f"{tp:,}")
        st.metric("False Churn Alerts", f"{fp:,}")
        st.metric("Missed Churners", f"{fn:,}")

    st.divider()

    # ---------------------------------------------------------
    # 4. Validation threshold analysis
    # ---------------------------------------------------------
    st.header("Threshold Analysis")

    st.caption(
        "The following comparison uses validation data, not the "
        "held-out test set. The selected threshold was chosen during "
        "model validation."
    )

    chart_data = thresholds.set_index("threshold")[
        ["precision", "recall", "f1_score"]
    ]

    st.line_chart(
        chart_data,
        x_label="Classification Threshold",
        y_label="Metric Score",
    )

    selected_rows = thresholds[
        (thresholds["threshold"] - selected_threshold).abs() < 1e-9
    ]

    if not selected_rows.empty:
        selected_row = selected_rows.iloc[0]

        st.subheader("Selected Threshold: Validation Results")

        threshold_col1, threshold_col2, threshold_col3 = st.columns(3)

        with threshold_col1:
            st.metric(
                "Validation Precision",
                _format_percentage(selected_row["precision"]),
            )

        with threshold_col2:
            st.metric(
                "Validation Recall",
                _format_percentage(selected_row["recall"]),
            )

        with threshold_col3:
            st.metric(
                "Validation F1 Score",
                _format_percentage(selected_row["f1_score"]),
            )

    st.subheader("Complete Threshold Comparison")

    display_columns = [
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    def highlight_selected_threshold(row):
        """Highlight the threshold selected during validation."""
        if abs(row["threshold"] - selected_threshold) < 1e-9:
            return [
                "background-color: #fff3cd; font-weight: bold"
            ] * len(row)

        return [""] * len(row)

    styled_thresholds = (
        thresholds[display_columns]
        .style
        .format(
            {
                "threshold": "{:.2f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1_score": "{:.3f}",
            }
        )
        .apply(highlight_selected_threshold, axis=1)
    )

    st.dataframe(
        styled_thresholds,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The selected threshold of 0.65 is based on validation F1 "
        "performance. It is a classification decision threshold, "
        "not a financial optimum or a guarantee of business outcomes."
    )