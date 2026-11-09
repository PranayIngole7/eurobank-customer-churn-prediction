"""Validation-only candidate comparison and model/threshold selection."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_split import split_training_data
from src.evaluation import compare_thresholds
from src.modeling import build_candidate_models, evaluate_model
from src.preprocessing import validate_feature_data


DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.10, 0.901, 0.05), 2))
MINIMUM_SELECTION_PRECISION = 0.50

MODEL_NAMES = {
    "dummy": "Majority-Class Dummy",
    "logistic_regression": "Logistic Regression",
    "logistic_regression_balanced": "Balanced Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def compare_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[dict, pd.DataFrame, dict]:
    """Fit candidates on training data and compare on one validation split.

    Returns fitted models, a threshold-0.50 comparison table, and positive-class
    validation probabilities. The function accepts no test dataset.
    """
    if not isinstance(X_train, pd.DataFrame) or not isinstance(
        X_validation, pd.DataFrame
    ):
        raise TypeError("Training and validation features must be DataFrames.")
    if not isinstance(y_train, pd.Series) or not isinstance(
        y_validation, pd.Series
    ):
        raise TypeError("Training and validation targets must be Series.")
    if X_train.empty or X_validation.empty:
        raise ValueError("Training and validation data cannot be empty.")
    if not X_train.columns.equals(X_validation.columns):
        raise ValueError("Training and validation feature columns must match.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("Training feature and target indexes must match.")
    if not X_validation.index.equals(y_validation.index):
        raise ValueError("Validation feature and target indexes must match.")
    if not set(X_train.index).isdisjoint(X_validation.index):
        raise ValueError("Training and validation row indexes must not overlap.")

    validate_feature_data(X_train)
    validate_feature_data(X_validation)

    if not y_train.isin([0, 1]).all() or y_train.nunique() != 2:
        raise ValueError("Training target must contain both binary classes.")
    if not y_validation.isin([0, 1]).all() or y_validation.nunique() != 2:
        raise ValueError("Validation target must contain both binary classes.")

    models = build_candidate_models()
    probabilities: dict[str, np.ndarray] = {}
    rows = []

    for model_key, model in models.items():
        model.fit(X_train, y_train)
        class_labels = list(model.classes_)
        if 1 not in class_labels:
            raise ValueError(f"Candidate '{model_key}' lacks positive class 1.")

        model_probabilities = np.asarray(model.predict_proba(X_validation))
        if model_probabilities.shape != (len(X_validation), len(class_labels)):
            raise ValueError(
                f"Candidate '{model_key}' returned an invalid probability shape."
            )
        if (
            not np.isfinite(model_probabilities).all()
            or ((model_probabilities < 0) | (model_probabilities > 1)).any()
            or not np.allclose(model_probabilities.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError(
                f"Candidate '{model_key}' returned invalid probabilities."
            )

        positive_probabilities = model_probabilities[:, class_labels.index(1)]
        probabilities[model_key] = positive_probabilities
        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
            threshold=0.5,
        )
        tn, fp = metrics["confusion_matrix"][0]
        fn, tp = metrics["confusion_matrix"][1]
        rows.append({
            "model_name": model_key,
            "algorithm": MODEL_NAMES[model_key],
            **{
                key: float(value)
                for key, value in metrics.items()
                if key != "confusion_matrix"
            },
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "threshold": 0.5,
        })

    comparison = (
        pd.DataFrame(rows)
        .sort_values(
            ["f1_score", "average_precision", "precision"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    return models, comparison, probabilities


def select_model_and_threshold(
    models: Mapping,
    validation_probabilities: Mapping[str, np.ndarray],
    y_validation: pd.Series,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    minimum_precision: float = MINIMUM_SELECTION_PRECISION,
) -> tuple[dict, pd.DataFrame]:
    """Select by validation F1 subject to a precision floor.

    This uses only validation labels and probabilities. It prioritizes a
    balance between missed churners and false alerts while requiring at least
    the configured 0.50 precision when any candidate/threshold can meet it.
    """
    if not isinstance(y_validation, pd.Series):
        raise TypeError("y_validation must be a pandas Series.")
    if y_validation.empty or y_validation.nunique() != 2:
        raise ValueError("Validation target must contain both binary classes.")
    if not np.isfinite(minimum_precision) or not 0 <= minimum_precision <= 1:
        raise ValueError("minimum_precision must be between 0 and 1.")
    if set(models) != set(validation_probabilities):
        raise ValueError("Models and validation probabilities must have matching keys.")

    selected_rows = []
    threshold_tables = []
    for model_name, model in models.items():
        scores = np.asarray(validation_probabilities[model_name], dtype=float)
        if scores.ndim != 1 or len(scores) != len(y_validation):
            raise ValueError(f"Invalid validation probability shape for {model_name}.")
        if (
            not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()
        ):
            raise ValueError(f"Invalid validation probabilities for {model_name}.")

        threshold_table = compare_thresholds(
            y_validation.to_numpy(),
            scores,
            thresholds,
        )
        threshold_table.insert(0, "model_name", model_name)
        ap = float(average_precision_score(y_validation, scores))
        auc = float(roc_auc_score(y_validation, scores))
        threshold_table["average_precision"] = ap
        threshold_table["roc_auc"] = auc
        threshold_tables.append(threshold_table)

        eligible = threshold_table[
            threshold_table["precision"] >= minimum_precision
        ]
        used_precision_floor = not eligible.empty
        candidate_rows = eligible if used_precision_floor else threshold_table
        best = candidate_rows.sort_values(
            ["f1_score", "precision", "fp", "threshold"],
            ascending=[False, False, True, False],
        ).iloc[0]
        selected_rows.append({
            "model_name": model_name,
            "algorithm": MODEL_NAMES.get(model_name, model_name),
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1_score": float(best["f1_score"]),
            "roc_auc": auc,
            "average_precision": ap,
            "tn": int(best["tn"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
            "tp": int(best["tp"]),
            "precision_floor_met": bool(used_precision_floor),
        })

    selection_table = pd.DataFrame(selected_rows)
    chosen = selection_table.sort_values(
        ["f1_score", "average_precision", "precision", "fp", "model_name"],
        ascending=[False, False, False, True, True],
    ).iloc[0]

    chosen_name = str(chosen["model_name"])
    selection = {
        "model_name": chosen_name,
        "algorithm": MODEL_NAMES.get(chosen_name, chosen_name),
        "threshold": float(chosen["threshold"]),
        "validation_metrics": {
            key: (
                bool(chosen[key])
                if key == "precision_floor_met"
                else (
                    int(chosen[key])
                    if key in {"tn", "fp", "fn", "tp"}
                    else float(chosen[key])
                )
            )
            for key in (
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "average_precision",
                "tn",
                "fp",
                "fn",
                "tp",
                "precision_floor_met",
            )
        },
        "selection_rule": (
            "For each model, select the threshold with highest validation "
            "F1-score among thresholds with precision >= "
            f"{minimum_precision:.2f}; if none meet the floor, use maximum "
            "F1-score across its thresholds. Select the model with highest "
            "resulting F1-score; break ties by Average Precision, precision, "
            "fewer false positives, then model name."
        ),
    }

    threshold_results = pd.concat(threshold_tables, ignore_index=True)
    return selection, threshold_results


def run_validation_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    minimum_precision: float = MINIMUM_SELECTION_PRECISION,
) -> dict:
    """Split original training data and perform all selection on validation.

    The API intentionally accepts only original training features/labels, so
    the held-out test data cannot be passed into candidate fitting or selection.
    """
    (
        X_internal_train,
        X_validation,
        y_internal_train,
        y_validation,
    ) = split_training_data(X_train, y_train)
    models, comparison, probabilities = compare_candidate_models(
        X_internal_train,
        y_internal_train,
        X_validation,
        y_validation,
    )
    selection, threshold_comparison = select_model_and_threshold(
        models,
        probabilities,
        y_validation,
        thresholds=thresholds,
        minimum_precision=minimum_precision,
    )
    return {
        "models": models,
        "comparison": comparison,
        "validation_probabilities": probabilities,
        "selection": selection,
        "threshold_comparison": threshold_comparison,
        "X_internal_train": X_internal_train,
        "X_validation": X_validation,
        "y_internal_train": y_internal_train,
        "y_validation": y_validation,
    }
