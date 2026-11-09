"""Tests for Model Insights dashboard rendering."""

import pandas as pd

from dashboard import model_insights


def sample_evaluation():
    return {
        "algorithm": "Gradient Boosting",
        "training_records": 8000,
        "test_records": 2000,
        "threshold": 0.30,
        "metrics": {
            "accuracy": 0.8465,
            "precision": 0.6244,
            "recall": 0.6167,
            "f1_score": 0.6205,
            "roc_auc": 0.8615,
            "average_precision": 0.7032,
        },
            "classification_report": {
            "Retained (0)": {
                "precision": 0.9024,
                "recall": 0.9052,
                "f1-score": 0.9038,
                "support": 1593,
            },
            "Churned (1)": {
                "precision": 0.6244,
                "recall": 0.6167,
                "f1-score": 0.6205,
                "support": 407,
            },
            "macro avg": {
                "precision": 0.7634,
                "recall": 0.7610,
                "f1-score": 0.7622,
                "support": 2000,
            },
            "weighted avg": {
                "precision": 0.8458,
                "recall": 0.8465,
                "f1-score": 0.8461,
                "support": 2000,
            },
            "accuracy": 0.8465,
        },
    }


def test_render_evaluation_details(monkeypatch):
    """Evaluation metrics and classification report render correctly."""

    captured = {
        "headers": [],
        "metrics": [],
        "dataframes": [],
    }

    class FakeColumn:
        def metric(self, label, value):
            captured["metrics"].append((label, value))

    monkeypatch.setattr(
        model_insights.st,
        "header",
        lambda value: captured["headers"].append(value),
    )
    monkeypatch.setattr(model_insights.st, "caption", lambda value: None)
    monkeypatch.setattr(model_insights.st, "info", lambda value: None)
    monkeypatch.setattr(model_insights.st, "subheader", lambda value: None)
    monkeypatch.setattr(
        model_insights.st,
        "columns",
        lambda count: [FakeColumn() for _ in range(count)],
    )
    monkeypatch.setattr(
        model_insights.st,
        "dataframe",
        lambda dataframe, **kwargs: captured["dataframes"].append(dataframe),
    )
    monkeypatch.setattr(model_insights.st, "warning", lambda value: None)

    model_insights.render_evaluation_details(sample_evaluation())

    assert "Held-Out Test Evaluation" in captured["headers"]
    assert len(captured["metrics"]) == 6

    report = captured["dataframes"][0]

    assert list(report["Class"]) == [
        "Retained (0)",
        "Churned (1)",
        "macro avg",
        "weighted avg",
    ]

    assert list(report["Support"]) == [1593, 407, 2000, 2000]
    assert report.iloc[0]["Precision"] == "90.24%"
    assert report.iloc[1]["Recall"] == "61.67%"


def test_render_partial_dependence_displays_all_figures(monkeypatch):
    captured = {"images": [], "captions": []}
    figures = {
        "BalanceToSalaryRatio": "/tmp/pdp-ratio.png",
        "ProductDensityIndicator": "/tmp/pdp-density.png",
    }

    monkeypatch.setattr(model_insights.st, "subheader", lambda value: None)
    monkeypatch.setattr(
        model_insights.st, "caption", captured["captions"].append
    )
    monkeypatch.setattr(model_insights.st, "markdown", lambda value: None)
    monkeypatch.setattr(
        model_insights.st,
        "image",
        lambda path, caption: captured["images"].append((path, caption)),
    )

    model_insights.render_partial_dependence(figures)

    assert [image[0] for image in captured["images"]] == list(figures.values())
    assert "not causal" in captured["captions"][0]


def test_fairness_group_table_shows_f1_and_confusion_counts(monkeypatch):
    captured = {"frames": []}
    group = pd.DataFrame([{
        "group": "Group A",
        "sample_count": 8,
        "actual_churn_rate": 0.5,
        "actual_churn_rate_ci_lower": 0.2,
        "actual_churn_rate_ci_upper": 0.8,
        "predicted_churn_rate": 0.5,
        "predicted_churn_rate_ci_lower": 0.2,
        "predicted_churn_rate_ci_upper": 0.8,
        "accuracy": 0.75,
        "precision": 0.67,
        "f1_score": 0.67,
        "true_positive": 2,
        "true_negative": 4,
        "false_positive": 1,
        "false_negative": 1,
        "tpr": 0.67,
        "tpr_ci_lower": 0.2,
        "tpr_ci_upper": 0.9,
        "fpr": 0.2,
        "fpr_ci_lower": 0.01,
        "fpr_ci_upper": 0.6,
        "fnr": 0.33,
    }])

    monkeypatch.setattr(model_insights.st, "markdown", lambda value: None)
    monkeypatch.setattr(
        model_insights.st,
        "dataframe",
        lambda frame, **kwargs: captured["frames"].append(frame),
    )

    model_insights.render_fairness_group_table(group)

    classification = captured["frames"][1]
    assert classification.loc[0, "F1-score"] == "67.00%"
    assert classification.loc[0, ["TP", "TN", "FP", "FN"]].tolist() == [
        2, 4, 1, 1
    ]


def test_render_model_limitations(monkeypatch):
    """All model limitations and responsible-use guidance render."""

    captured = {
        "headers": [],
        "markdown": [],
        "warnings": [],
    }

    monkeypatch.setattr(
        model_insights.st,
        "header",
        lambda value: captured["headers"].append(value),
    )
    monkeypatch.setattr(
        model_insights.st,
        "markdown",
        lambda value: captured["markdown"].append(value),
    )
    monkeypatch.setattr(
        model_insights.st,
        "warning",
        lambda value: captured["warnings"].append(value),
    )

    model_insights.render_model_limitations("Gradient Boosting")

    assert captured["headers"] == [
        "Model Limitations and Responsible Use"
    ]

    limitations_text = captured["markdown"][0]

    for topic in [
        "Imperfect churn detection",
        "Class imbalance",
        "No probability calibration guarantee",
        "Algorithm-specific assumptions",
        "Gradient Boosting",
        "Correlated and overlapping features",
        "Observed subgroup disparities",
        "Limited external validation",
        "No causal interpretation",
    ]:
        assert topic in limitations_text

    assert len(captured["warnings"]) == 1
    assert "human review" in captured["warnings"][0]


def test_render_model_insights_integrates_all_sections(monkeypatch):
    """Model Insights loads artifacts and renders every section."""

    calls = []

    monkeypatch.setattr(model_insights, "render_header", lambda *args: None)

    monkeypatch.setattr(
        model_insights,
        "load_test_evaluation",
        lambda: sample_evaluation(),
    )
    monkeypatch.setattr(
        model_insights,
        "load_feature_coefficients",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        model_insights,
        "load_model_metadata",
        lambda: {"algorithm": "Gradient Boosting"},
    )
    monkeypatch.setattr(
        model_insights,
        "load_permutation_importance",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        model_insights,
        "load_shap_global_importance",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        model_insights,
        "load_partial_dependence_figures",
        lambda: {"Age": "/tmp/pdp-age.png"},
    )
    monkeypatch.setattr(
        model_insights,
        "load_fairness_group_metrics",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        model_insights,
        "load_fairness_summary",
        lambda: {},
    )

    monkeypatch.setattr(
        model_insights.st,
        "divider",
        lambda: calls.append("divider"),
    )
    monkeypatch.setattr(
        model_insights.st,
        "error",
        lambda message: calls.append("error"),
    )

    monkeypatch.setattr(
        model_insights,
        "render_evaluation_details",
        lambda data: calls.append("evaluation"),
    )
    monkeypatch.setattr(
        model_insights,
        "render_model_limitations",
        lambda algorithm: calls.append("limitations"),
    )
    monkeypatch.setattr(
        model_insights,
        "render_explainability",
        lambda coefficients, importance, shap_importance: calls.append(
            "explainability"
        ),
    )
    monkeypatch.setattr(
        model_insights,
        "render_partial_dependence",
        lambda figures: calls.append("pdp"),
    )
    monkeypatch.setattr(
        model_insights,
        "render_fairness_overview",
        lambda groups, summary: calls.append("fairness"),
    )

    model_insights.render_model_insights()

    assert calls == [
        "evaluation",
        "divider",
        "limitations",
        "divider",
        "explainability",
        "divider",
        "pdp",
        "divider",
        "fairness",
    ]
