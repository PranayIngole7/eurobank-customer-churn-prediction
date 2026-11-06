"""Tests for Model Insights dashboard rendering."""

import pandas as pd

from dashboard import model_insights


def sample_evaluation():
    return {
        "algorithm": "Balanced Logistic Regression",
        "training_records": 8000,
        "test_records": 2000,
        "threshold": 0.65,
        "metrics": {
            "accuracy": 0.80,
            "precision": 0.5081,
            "recall": 0.5405,
            "f1_score": 0.5238,
            "roc_auc": 0.7948,
            "average_precision": 0.5460,
        },
        "classification_report": {
            "Retained (0)": {
                "precision": 0.8807,
                "recall": 0.8663,
                "f1-score": 0.8734,
                "support": 1593,
            },
            "Churned (1)": {
                "precision": 0.5081,
                "recall": 0.5405,
                "f1-score": 0.5238,
                "support": 407,
            },
            "macro avg": {
                "precision": 0.6944,
                "recall": 0.7034,
                "f1-score": 0.6986,
                "support": 2000,
            },
            "weighted avg": {
                "precision": 0.8048,
                "recall": 0.8000,
                "f1-score": 0.8023,
                "support": 2000,
            },
            "accuracy": 0.80,
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
    assert report.iloc[0]["Precision"] == "88.07%"
    assert report.iloc[1]["Recall"] == "54.05%"


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

    model_insights.render_model_limitations()

    assert captured["headers"] == [
        "Model Limitations and Responsible Use"
    ]

    limitations_text = captured["markdown"][0]

    for topic in [
        "Moderate churn detection performance",
        "Class imbalance",
        "No probability calibration guarantee",
        "Logistic regression assumptions",
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
        "load_permutation_importance",
        lambda: pd.DataFrame(),
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
        lambda: calls.append("limitations"),
    )
    monkeypatch.setattr(
        model_insights,
        "render_explainability",
        lambda coefficients, importance: calls.append("explainability"),
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
        "fairness",
    ]