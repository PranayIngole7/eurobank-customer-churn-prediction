import pytest
import numpy as np

from src.evaluation import generate_classification_report


def test_classification_report_returns_expected_structure():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    results = generate_classification_report(y_true, y_pred)

    expected_keys = {
        "Retained (0)",
        "Churned (1)",
        "accuracy",
        "macro avg",
        "weighted avg",
    }

    assert set(results.keys()) == expected_keys


def test_classification_report_calculates_expected_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    results = generate_classification_report(y_true, y_pred)

    assert results["Retained (0)"]["precision"] == pytest.approx(1.0)
    assert results["Retained (0)"]["recall"] == pytest.approx(0.5)
    assert results["Retained (0)"]["f1-score"] == pytest.approx(2 / 3)
    assert results["Retained (0)"]["support"] == 2

    assert results["Churned (1)"]["precision"] == pytest.approx(2 / 3)
    assert results["Churned (1)"]["recall"] == pytest.approx(1.0)
    assert results["Churned (1)"]["f1-score"] == pytest.approx(0.8)
    assert results["Churned (1)"]["support"] == 2

    assert results["accuracy"] == pytest.approx(0.75)


def test_classification_report_includes_both_classes_when_predictions_have_one_class():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]

    results = generate_classification_report(y_true, y_pred)

    assert "Retained (0)" in results
    assert "Churned (1)" in results

    assert results["Churned (1)"]["precision"] == 0.0
    assert results["Churned (1)"]["recall"] == 0.0


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([], []),
        ([0, 1], [0]),
        ([[0, 1]], [[0, 1]]),
        ([0, 1, 2], [0, 1, 0]),
        ([0, 1], [0, 2]),
        ([0, None], [0, 1]),
    ],
)
def test_classification_report_rejects_invalid_inputs(y_true, y_pred):
    with pytest.raises(ValueError):
        generate_classification_report(y_true, y_pred)


def test_confusion_matrix_returns_figure_and_axes():
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    from src.evaluation import plot_confusion_matrix

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    fig, ax = plot_confusion_matrix(y_true, y_pred)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == "Predicted Class"
    assert ax.get_ylabel() == "Actual Class"

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_confusion_matrix_contains_correct_counts():
    import matplotlib.pyplot as plt

    from src.evaluation import plot_confusion_matrix

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    fig, ax = plot_confusion_matrix(y_true, y_pred)

    annotations = [
        text.get_text()
        for text in ax.texts
    ]

    assert annotations == ["1", "1", "0", "2"]

    plt.close(fig)


def test_confusion_matrix_saves_figure(tmp_path):
    import matplotlib.pyplot as plt

    from src.evaluation import plot_confusion_matrix

    output_path = tmp_path / "figures" / "confusion_matrix.png"

    fig, ax = plot_confusion_matrix(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([], []),
        ([0, 1], [0]),
        ([[0, 1]], [[0, 1]]),
        ([0, 1, 2], [0, 1, 0]),
        ([0, 1], [0, 2]),
    ],
)
def test_confusion_matrix_rejects_invalid_inputs(y_true, y_pred):
    from src.evaluation import plot_confusion_matrix

    with pytest.raises(ValueError):
        plot_confusion_matrix(y_true, y_pred)

def test_roc_curve_returns_figure_axes_and_auc():
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    from src.evaluation import plot_roc_curve

    y_true = [0, 0, 1, 1]
    y_score = [0.10, 0.40, 0.60, 0.90]

    fig, ax, auc_score = plot_roc_curve(y_true, y_score)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert auc_score == pytest.approx(1.0)

    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate"

    plt.close(fig)


def test_roc_curve_saves_figure(tmp_path):
    import matplotlib.pyplot as plt

    from src.evaluation import plot_roc_curve

    output_path = tmp_path / "figures" / "roc_curve.png"

    fig, ax, auc_score = plot_roc_curve(
        [0, 0, 1, 1],
        [0.10, 0.40, 0.60, 0.90],
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


def test_roc_curve_rejects_single_class():
    from src.evaluation import plot_roc_curve

    with pytest.raises(ValueError, match="both target classes"):
        plot_roc_curve(
            [0, 0, 0, 0],
            [0.10, 0.20, 0.30, 0.40],
        )


@pytest.mark.parametrize(
    "y_true, y_score",
    [
        ([], []),
        ([0, 1], [0.2]),
        ([[0, 1]], [[0.2, 0.8]]),
        ([0, 1, 2], [0.2, 0.8, 0.5]),
        ([0, 1], [0.2, np.nan]),
        ([0, 1], [0.2, np.inf]),
        ([0, 1], ["low", "high"]),
    ],
)
def test_roc_curve_rejects_invalid_inputs(y_true, y_score):
    from src.evaluation import plot_roc_curve

    with pytest.raises(ValueError):
        plot_roc_curve(y_true, y_score)

def test_precision_recall_curve_returns_figure_axes_and_ap():
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    from src.evaluation import plot_precision_recall_curve

    y_true = [0, 0, 1, 1]
    y_score = [0.10, 0.40, 0.60, 0.90]

    fig, ax, average_precision = plot_precision_recall_curve(
        y_true,
        y_score,
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert average_precision == pytest.approx(1.0)

    assert ax.get_xlabel() == "Recall"
    assert ax.get_ylabel() == "Precision"

    plt.close(fig)


def test_precision_recall_curve_saves_figure(tmp_path):
    import matplotlib.pyplot as plt

    from src.evaluation import plot_precision_recall_curve

    output_path = tmp_path / "figures" / "precision_recall_curve.png"

    fig, ax, average_precision = plot_precision_recall_curve(
        [0, 0, 1, 1],
        [0.10, 0.40, 0.60, 0.90],
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


def test_precision_recall_curve_rejects_single_class():
    from src.evaluation import plot_precision_recall_curve

    with pytest.raises(ValueError, match="both target classes"):
        plot_precision_recall_curve(
            [0, 0, 0, 0],
            [0.10, 0.20, 0.30, 0.40],
        )


@pytest.mark.parametrize(
    "y_true, y_score",
    [
        ([], []),
        ([0, 1], [0.2]),
        ([[0, 1]], [[0.2, 0.8]]),
        ([0, 1, 2], [0.2, 0.8, 0.5]),
        ([0, 1], [0.2, np.nan]),
        ([0, 1], [0.2, np.inf]),
        ([0, 1], ["low", "high"]),
    ],
)
def test_precision_recall_curve_rejects_invalid_inputs(
    y_true,
    y_score,
):
    from src.evaluation import plot_precision_recall_curve

    with pytest.raises(ValueError):
        plot_precision_recall_curve(y_true, y_score)

def test_compare_thresholds_returns_expected_dataframe():
    import pandas as pd

    from src.evaluation import compare_thresholds

    y_true = [0, 0, 1, 1]
    y_score = [0.20, 0.40, 0.60, 0.80]
    thresholds = [0.30, 0.50, 0.70]

    results = compare_thresholds(
        y_true,
        y_score,
        thresholds,
    )

    assert isinstance(results, pd.DataFrame)

    expected_columns = [
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    assert list(results.columns) == expected_columns
    assert len(results) == 3


def test_compare_thresholds_calculates_expected_metrics():
    from src.evaluation import compare_thresholds

    y_true = [0, 0, 1, 1]
    y_score = [0.20, 0.40, 0.60, 0.80]

    results = compare_thresholds(
        y_true,
        y_score,
        [0.30, 0.50, 0.70],
    )

    # Threshold 0.30: predictions [0, 1, 1, 1]
    first = results.iloc[0]

    assert first["precision"] == pytest.approx(2 / 3)
    assert first["recall"] == pytest.approx(1.0)
    assert first["f1_score"] == pytest.approx(0.8)
    assert first["tn"] == 1
    assert first["fp"] == 1
    assert first["fn"] == 0
    assert first["tp"] == 2

    # Threshold 0.50: predictions [0, 0, 1, 1]
    second = results.iloc[1]

    assert second["precision"] == pytest.approx(1.0)
    assert second["recall"] == pytest.approx(1.0)
    assert second["f1_score"] == pytest.approx(1.0)

    # Threshold 0.70: predictions [0, 0, 0, 1]
    third = results.iloc[2]

    assert third["precision"] == pytest.approx(1.0)
    assert third["recall"] == pytest.approx(0.5)
    assert third["f1_score"] == pytest.approx(2 / 3)


def test_compare_thresholds_preserves_threshold_order():
    from src.evaluation import compare_thresholds

    results = compare_thresholds(
        [0, 0, 1, 1],
        [0.20, 0.40, 0.60, 0.80],
        [0.70, 0.30, 0.50],
    )

    assert results["threshold"].tolist() == [
        0.70,
        0.30,
        0.50,
    ]


@pytest.mark.parametrize(
    "y_true, y_score, thresholds",
    [
        ([], [], [0.5]),
        ([0, 1], [0.2], [0.5]),
        ([0, 1], [0.2, 0.8], []),
        ([0, 1], [0.2, 0.8], [1.1]),
        ([0, 1], [0.2, 0.8], [-0.1]),
        ([0, 1], [0.2, 0.8], [np.nan]),
        ([0, 1], [0.2, 0.8], [np.inf]),
        ([0, 1], [0.2, np.nan], [0.5]),
        ([0, 1], [0.2, np.inf], [0.5]),
        ([0, 2], [0.2, 0.8], [0.5]),
        ([0, 0], [0.2, 0.8], [0.5]),
        ([[0, 1]], [[0.2, 0.8]], [0.5]),
        ([0, 1], ["low", "high"], [0.5]),
        ([0, 1], [0.2, 0.8], ["0.5"]),
        ([0, 1], [0.2, 0.8], [True]),
    ],
)
def test_compare_thresholds_rejects_invalid_inputs(
    y_true,
    y_score,
    thresholds,
):
    from src.evaluation import compare_thresholds

    with pytest.raises(ValueError):
        compare_thresholds(
            y_true,
            y_score,
            thresholds,
        )


def test_plot_threshold_comparison_returns_figure_and_axes():
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    from src.evaluation import (
        compare_thresholds,
        plot_threshold_comparison,
    )

    results = compare_thresholds(
        [0, 0, 1, 1],
        [0.20, 0.40, 0.60, 0.80],
        [0.30, 0.50, 0.70],
    )

    fig, ax = plot_threshold_comparison(results)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    assert ax.get_xlabel() == "Classification Threshold"
    assert ax.get_ylabel() == "Score"
    assert len(ax.lines) == 3

    plt.close(fig)


def test_plot_threshold_comparison_saves_figure(tmp_path):
    import matplotlib.pyplot as plt

    from src.evaluation import (
        compare_thresholds,
        plot_threshold_comparison,
    )

    results = compare_thresholds(
        [0, 0, 1, 1],
        [0.20, 0.40, 0.60, 0.80],
        [0.30, 0.50, 0.70],
    )

    output_path = tmp_path / "figures" / "threshold_comparison.png"

    fig, ax = plot_threshold_comparison(
        results,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)