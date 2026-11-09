"""Model insights and explainability dashboard."""

import pandas as pd
import streamlit as st

from dashboard.components import render_header, render_section_header
from dashboard.data_access import (
    load_test_evaluation,
    load_model_metadata,
    load_feature_coefficients,
    load_permutation_importance,
    load_shap_global_importance,
    load_partial_dependence_figures,
    load_fairness_group_metrics,
    load_fairness_summary,
)

def render_evaluation_details(evaluation):
    """Display saved held-out test evaluation results."""

    st.header("Held-Out Test Evaluation")

    st.caption(
        f"Model: {evaluation['algorithm']} | "
        f"Training records: {evaluation['training_records']:,} | "
        f"Test records: {evaluation['test_records']:,} | "
        f"Classification threshold: {evaluation['threshold']:.2f}"
    )

    st.info(
        "These metrics come from the held-out test dataset. "
        "The classification threshold was selected using validation "
        "data before this final evaluation."
    )

    metrics = evaluation["metrics"]

    metric_definitions = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1_score"),
        ("ROC-AUC", "roc_auc"),
        ("Average Precision", "average_precision"),
    ]

    for start in (0, 3):
        columns = st.columns(3)

        for column, (label, key) in zip(
            columns,
            metric_definitions[start:start + 3],
        ):
            column.metric(
                label=label,
                value=f"{metrics[key]:.2%}",
            )

    st.subheader("Classification Report")

    classification_report = evaluation["classification_report"]

    display_order = [
        "Retained (0)",
        "Churned (1)",
        "macro avg",
        "weighted avg",
    ]

    rows = []

    for label in display_order:
        values = classification_report[label]

        rows.append(
            {
                "Class": label,
                "Precision": f"{values['precision']:.2%}",
                "Recall": f"{values['recall']:.2%}",
                "F1-score": f"{values['f1-score']:.2%}",
                "Support": int(values["support"]),
            }
        )

    report_df = pd.DataFrame(rows)

    st.dataframe(
        report_df,
        use_container_width=True,
        hide_index=True,
        # column_config={
        #     "Precision": st.column_config.NumberColumn(
        #         "Precision",
        #         format="percent",
        #     ),
        #     "Recall": st.column_config.NumberColumn(
        #         "Recall",
        #         format="percent",
        #     ),
        #     "F1-score": st.column_config.NumberColumn(
        #         "F1-score",
        #         format="percent",
        #     ),
        # },
    )

    st.warning(
        "Churn detection remains imperfect. Review the displayed "
        "precision, recall and confusion matrix when setting operational "
        "expectations; predictions should support prioritization rather "
        "than be treated as certain outcomes."
    )

def render_model_limitations(algorithm):
    """Explain limitations and responsible use of the model."""

    st.header("Model Limitations and Responsible Use")

    st.markdown(
        f"""
        **1. Imperfect churn detection**

        The final held-out test metrics are displayed above. Every
        classification threshold trades off false churn alerts against
        missed churners; use the reported precision, recall and confusion
        matrix when setting operational expectations.

        **2. Class imbalance**

        The test dataset contains substantially more retained customers
        than churned customers. Overall accuracy alone can therefore
        hide weaknesses in identifying the minority churn class.

        **3. No probability calibration guarantee**

        Predicted probabilities are model scores and should not
        automatically be interpreted as perfectly calibrated real-world
        churn probabilities.

        **4. Algorithm-specific assumptions**

        The selected model is **{algorithm}**. Its learned structure may
        not capture every relationship present in future customer
        populations. Explanations reflect this fitted model, not an
        independently verified business mechanism.

        **5. Correlated and overlapping features**

        Age and AgeGroup contain overlapping information. Their
        coefficients and feature-importance values should not be
        interpreted as independent effects.

        **6. Observed subgroup disparities**

        Validation-based fairness analysis identifies differences
        between demographic groups. These differences require further
        investigation and should not be interpreted as proof that the
        model is fair or unfair.

        **7. Limited external validation**

        Performance has been evaluated using the available dataset.
        Generalization to different banks, customer populations,
        geographic regions, or future time periods has not been
        established.

        **8. No causal interpretation**

        Model coefficients and feature importance describe learned
        statistical associations. They do not establish that changing
        a feature will cause a customer to remain with or leave a bank.
        """
    )

    st.warning(
        "Responsible-use guidance: use predictions as decision-support "
        "signals for human review. Do not automatically deny services, "
        "change customer terms, or take other consequential actions "
        "solely because of a churn prediction. Monitor model performance "
        "and subgroup outcomes before any real-world deployment."
    )

def render_coefficient_analysis(coefficients):
    """Display logistic regression coefficient and odds-ratio analysis."""

    render_section_header(
        "Logistic Regression Coefficients",
        "Coefficient direction and magnitude describe associations "
        "learned by the selected logistic regression model.",
    )

    chart_data = (
        coefficients
        .sort_values("coefficient")
        [["feature", "coefficient"]]
    )

    st.bar_chart(
        chart_data,
        x="coefficient",
        y="feature",
        horizontal=True,
        x_label="Coefficient (log-odds)",
        y_label="Feature",
    )

    display_data = (
        coefficients
        .sort_values("absolute_coefficient", ascending=False)
        [["feature", "coefficient", "odds_ratio", "direction"]]
    )

    st.subheader("Coefficient and Odds Ratio Details")

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "coefficient": st.column_config.NumberColumn(
                "Coefficient",
                format="%.4f",
            ),
            "odds_ratio": st.column_config.NumberColumn(
                "Odds Ratio",
                format="%.4f",
            ),
        },
    )

    st.info(
        "Positive coefficients indicate higher predicted churn odds; "
        "negative coefficients indicate lower predicted churn odds. "
        "Numerical coefficients represent one-standard-deviation changes. "
        "Categorical coefficients are relative to their reference category. "
        "Odds ratios describe changes in odds, not changes in probability."
    )


def render_permutation_importance(importance):
    """Display validation-based permutation feature importance."""

    render_section_header(
        "Permutation Feature Importance",
        "Importance is measured by the decrease in validation ROC-AUC "
        "when a feature is shuffled.",
    )

    chart_data = (
        importance
        .sort_values("importance_mean")
        [["feature", "importance_mean"]]
    )

    st.bar_chart(
        chart_data,
        x="importance_mean",
        y="feature",
        horizontal=True,
        x_label="Mean ROC-AUC decrease",
        y_label="Feature",
    )

    display_data = (
        importance
        .sort_values("importance_mean", ascending=False)
        .rename(
            columns={
                "importance_mean": "Mean ROC-AUC decrease",
                "importance_std": "Standard deviation",
            }
        )
    )

    st.subheader("Permutation Importance Details")

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mean ROC-AUC decrease": st.column_config.NumberColumn(
                "Mean ROC-AUC decrease",
                format="%.6f",
            ),
            "Standard deviation": st.column_config.NumberColumn(
                "Standard deviation (10 repeats)",
                format="%.6f",
            ),
        },
    )

    st.info(
        "Higher positive values indicate that shuffling a feature "
        "reduced validation ROC-AUC more substantially. Near-zero or "
        "negative values mean that shuffling did not reduce ROC-AUC "
        "in this experiment. Standard deviations describe variation "
        "across the ten permutation repeats; they are not confidence "
        "intervals."
    )


def render_shap_importance(shap_importance):
    """Display global mean absolute SHAP attributions."""
    render_section_header(
        "SHAP Global Feature Importance",
        "Mean absolute contribution to model-predicted churn probability "
        "on a reproducible validation sample.",
    )
    display = shap_importance.sort_values(
        "mean_absolute_shap",
        ascending=True,
    )
    st.bar_chart(
        display,
        x="mean_absolute_shap",
        y="feature",
        horizontal=True,
        x_label="Mean absolute SHAP value",
        y_label="Model input feature",
    )
    st.caption(
        "SHAP values explain model outputs, not causes. Local SHAP values "
        "describe one record's attribution relative to the sampled background."
    )


def render_explainability(coefficients, importance, shap_importance):
    """Render global model explainability sections."""

    st.header("Model Explainability")

    if coefficients is not None:
        render_coefficient_analysis(coefficients)
        st.divider()

    render_permutation_importance(importance)

    st.divider()
    render_shap_importance(shap_importance)

    st.warning(
        "Interpretation limitations: coefficients and permutation "
        "importance describe model behavior, not causation. Age and "
        "AgeGroup contain overlapping information, so their importance "
        "values should not be interpreted as independent contributions. "
        "Negative or near-zero importance alone is not sufficient "
        "evidence for removing a feature."
    )

def render_partial_dependence(figures):
    """Display saved PDP figures for the four engineered features."""

    st.subheader("Partial Dependence")
    st.caption(
        "Each curve averages predicted churn probability across the "
        "evaluation records while varying one feature. Engineered "
        "interactions and correlated inputs can create combinations that "
        "are uncommon in real customers; these plots are descriptive, "
        "not causal."
    )
    for feature, figure_path in figures.items():
        st.markdown(f"##### {feature}")
        st.image(str(figure_path), caption=f"Partial dependence for {feature}")


def format_rate(value):
    """Format a rate as a percentage, preserving missing values."""

    if pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def format_rate_with_interval(row, metric):
    """Format a rate and its 95% Wilson confidence interval."""

    value = row[metric]
    lower = row[f"{metric}_ci_lower"]
    upper = row[f"{metric}_ci_upper"]

    if pd.isna(value) or pd.isna(lower) or pd.isna(upper):
        return "N/A"

    return (
        f"{value:.2%} "
        f"({lower:.2%}–{upper:.2%})"
    )


def render_fairness_summary_metrics(fairness_summary):
    """Display saved fairness indicators."""

    columns = st.columns(5)

    metrics = [
        (
            "Demographic Parity Difference",
            "demographic_parity_difference",
        ),
        (
            "Equal Opportunity Difference",
            "equal_opportunity_difference",
        ),
        (
            "FPR Difference",
            "false_positive_rate_difference",
        ),
        (
            "Equalized Odds Difference",
            "equalized_odds_difference",
        ),
        (
            "Disparate Impact Ratio",
            "disparate_impact_ratio",
        ),
    ]

    for column, (label, key) in zip(columns, metrics):
        value = fairness_summary[key]

        if key == "disparate_impact_ratio":
            display_value = f"{value:.3f}"
        else:
            display_value = f"{value:.2%}"

        column.metric(
            label=label,
            value=display_value,
        )

def render_fairness_group_table(groups):
    """Display readable churn and classification performance tables."""

    churn_rows = []
    classification_rows = []

    for _, row in groups.iterrows():
        churn_rows.append(
            {
                "Group": row["group"],
                "Customers": int(row["sample_count"]),
                "Actual Churn": format_rate(row["actual_churn_rate"]),
                "Actual Churn (95% CI)": format_rate_with_interval(
                    row, "actual_churn_rate"
                ),
                "Predicted Churn": format_rate(
                    row["predicted_churn_rate"]
                ),
                "Predicted Churn (95% CI)": format_rate_with_interval(
                    row, "predicted_churn_rate"
                ),
            }
        )

        classification_rows.append(
            {
                "Group": row["group"],
                "Accuracy": format_rate(row["accuracy"]),
                "Precision": format_rate(row["precision"]),
                "F1-score": format_rate(row["f1_score"]),
                "TP": int(row["true_positive"]),
                "TN": int(row["true_negative"]),
                "FP": int(row["false_positive"]),
                "FN": int(row["false_negative"]),
                "TPR": format_rate(row["tpr"]),
                "TPR (95% CI)": format_rate_with_interval(row, "tpr"),
                "FPR": format_rate(row["fpr"]),
                "FPR (95% CI)": format_rate_with_interval(row, "fpr"),
                "FNR": format_rate(row["fnr"]),
            }
        )

    st.markdown("##### Churn Performance")

    st.dataframe(
        pd.DataFrame(churn_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("##### Classification Performance")

    st.dataframe(
        pd.DataFrame(classification_rows),
        use_container_width=True,
        hide_index=True,
    )

def render_fairness_attribute(
    attribute,
    groups,
    fairness_summary,
):
    """Render fairness analysis for one demographic attribute."""

    st.subheader(f"{attribute} Analysis")

    st.caption(
        f"Held-out test dataset | {len(groups)} groups | "
        f"{int(groups['sample_count'].sum())} customers"
    )

    render_fairness_summary_metrics(fairness_summary)

    st.markdown("#### Group Performance Comparison")

    render_fairness_group_table(groups)


def render_fairness_overview(groups, summary):
    """Render the complete demographic fairness dashboard."""

    st.header("Fairness and Group Performance")

    st.warning(
        "This is a post-selection audit on the held-out test dataset. "
        "Test results were not used to fit, tune, or select the model or "
        "threshold. Differences indicate observed performance variation; "
        "they do not independently establish discrimination or prove "
        "that a model is fair or unfair."
    )

    st.caption(
        f"Evaluation dataset: {summary['evaluation_dataset']} | "
        f"Records: {summary['records']} | "
        f"Threshold: {summary['threshold']:.2f} | "
        f"Confidence intervals: "
        f"{summary['confidence_intervals']['method']} "
        f"({summary['confidence_intervals']['confidence_level']:.0%})"
    )

    tabs = st.tabs(summary["audit_attributes"])

    for tab, attribute in zip(tabs, summary["audit_attributes"]):
        with tab:
            attribute_groups = groups[
                groups["attribute"] == attribute
            ].copy()

            if attribute == "AgeGroup":
                age_order = [
                    "<30",
                    "30-39",
                    "40-49",
                    "50-59",
                    "60+",
                ]

                attribute_groups["group"] = pd.Categorical(
                    attribute_groups["group"],
                    categories=age_order,
                    ordered=True,
                )

                attribute_groups = attribute_groups.sort_values("group")

            fairness_summary = summary["attributes"][attribute][
                "fairness_summary"
            ]

            render_fairness_attribute(
                attribute,
                attribute_groups,
                fairness_summary,
            )

    st.divider()

    st.subheader("How to Interpret Fairness Indicators")

    st.markdown(
        """
        - **Demographic Parity Difference:** Difference between the
          highest and lowest group predicted-positive rates.
        - **Equal Opportunity Difference:** Difference between group
          true positive rates.
        - **FPR Difference:** Difference between group false positive rates.
        - **Equalized Odds Difference:** Maximum of the TPR and FPR gaps.
        - **Disparate Impact Ratio:** Ratio of the lowest to highest
          group predicted-positive rates.
        """
    )

    st.info(
        "A disparate-impact ratio of 0.80 is sometimes used as a "
        "four-fifths screening heuristic. It is not a universal legal "
        "standard or a complete fairness assessment. Group sizes, "
        "confidence intervals, business context and applicable law "
        "must also be considered."
    )

def render_model_insights():
    """Render the model insights page."""

    render_header(
        "Model Insights",
        "Explore model evaluation, explainability, and fairness.",
    )

    try:
        evaluation = load_test_evaluation()
        metadata = load_model_metadata()
        coefficients = (
            load_feature_coefficients()
            if "Logistic Regression" in metadata["algorithm"]
            else None
        )
        importance = load_permutation_importance()
        shap_importance = load_shap_global_importance()
        pdp_figures = load_partial_dependence_figures()
        fairness_groups = load_fairness_group_metrics()
        fairness_summary = load_fairness_summary()

    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Unable to load model insights data: {exc}")
        return

    render_evaluation_details(evaluation)

    st.divider()

    render_model_limitations(metadata["algorithm"])

    st.divider()

    render_explainability(coefficients, importance, shap_importance)

    st.divider()
    render_partial_dependence(pdp_figures)

    st.divider()

    render_fairness_overview(
        fairness_groups,
        fairness_summary,
    )