"""
Analyze global feature influence for the selected churn model.

The model is fitted on the internal training split.
Coefficients and permutation importance are analyzed without
using the original test split.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_split import split_dataset, split_training_data
from src.explainability import (
    calculate_permutation_importance,
    extract_model_coefficients,
)
from src.feature_engineering import engineer_features
from src.modeling import build_baseline_models


DATA_PATH = Path("data/raw/european-bank.csv")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

TOP_FEATURES = 15
N_REPEATS = 10
RANDOM_STATE = 42


def plot_top_coefficients(coefficients, output_path):
    """Plot the features with the largest absolute coefficients."""

    top = (
        coefficients
        .head(TOP_FEATURES)
        .sort_values("coefficient")
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top["feature"],
        top["coefficient"],
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title("Logistic Regression: Top Feature Coefficients")
    ax.set_xlabel("Coefficient (log-odds scale)")
    ax.set_ylabel("Transformed feature")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_importance(importance, output_path):
    """Plot the most important original input features."""

    top = (
        importance
        .head(TOP_FEATURES)
        .sort_values("importance_mean")
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        capsize=3,
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Mean decrease in validation ROC-AUC")
    ax.set_ylabel("Original input feature")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # 1. Load and engineer the original dataset.
    raw_data = pd.read_csv(DATA_PATH)
    processed_data = engineer_features(raw_data)

    # 2. Create the original training/test split.
    #    The original test split is deliberately discarded.
    X_train, _, y_train, _ = split_dataset(processed_data)

    # 3. Create the internal training/validation split.
    (
        X_internal_train,
        X_validation,
        y_internal_train,
        y_validation,
    ) = split_training_data(X_train, y_train)

    print("Explainability dataset split")
    print("----------------------------")
    print(f"Internal training records: {len(X_internal_train)}")
    print(f"Validation records:        {len(X_validation)}")
    print("Original test records:     excluded")

    # 4. Fit the selected balanced Logistic Regression pipeline.
    model = build_baseline_models()["logistic_regression_balanced"]
    model.fit(X_internal_train, y_internal_train)

    # 5. Extract coefficients and odds ratios.
    coefficients = extract_model_coefficients(model)

    # 6. Calculate permutation importance on validation data.
    importance = calculate_permutation_importance(
        model,
        X_validation,
        y_validation,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    # 7. Save numerical results.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    coefficients_path = REPORTS_DIR / "feature-coefficients.csv"
    importance_path = REPORTS_DIR / "permutation-importance.csv"

    coefficients.to_csv(coefficients_path, index=False)
    importance.to_csv(importance_path, index=False)

    # 8. Generate visualizations.
    coefficients_figure = FIGURES_DIR / "feature-coefficients.png"
    importance_figure = FIGURES_DIR / "permutation-importance.png"

    plot_top_coefficients(coefficients, coefficients_figure)
    plot_permutation_importance(importance, importance_figure)

    # 9. Display the leading results.
    print("\nTop Features by Absolute Coefficient")
    print("------------------------------------")
    print(
        coefficients[
            ["feature", "coefficient", "odds_ratio", "direction"]
        ].head(TOP_FEATURES).to_string(index=False)
    )

    print("\nTop Features by Permutation Importance")
    print("--------------------------------------")
    print(
        importance.head(TOP_FEATURES).to_string(index=False)
    )

    print("\nGenerated Artifacts")
    print("-------------------")
    for path in (
        coefficients_path,
        importance_path,
        coefficients_figure,
        importance_figure,
    ):
        print(path)

    print("\nGlobal explainability analysis completed.")


if __name__ == "__main__":
    main()
