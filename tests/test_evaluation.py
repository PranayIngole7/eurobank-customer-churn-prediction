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