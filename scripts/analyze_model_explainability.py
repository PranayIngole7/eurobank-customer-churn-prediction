"""Generate selected-algorithm explanations using validation records only."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.data_split import split_dataset, split_training_data
from src.explainability import (
    calculate_permutation_importance,
    calculate_shap_values,
    generate_partial_dependence,
)
from src.feature_engineering import engineer_features
from src.model_persistence import get_model_metadata
from src.modeling import build_candidate_models


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "european-bank.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PDP_FEATURES = (
    "BalanceToSalaryRatio",
    "ProductDensityIndicator",
    "EngagementProductInteraction",
    "AgeTenureInteraction",
)
SHAP_MAX_EVALS = 33


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    processed_data = engineer_features(pd.read_csv(DATA_PATH))
    X_train, _, y_train, _ = split_dataset(processed_data)
    X_internal_train, X_validation, y_internal_train, y_validation = (
        split_training_data(X_train, y_train)
    )
    metadata = get_model_metadata()
    model = build_candidate_models()[metadata["model_name"]]
    model.fit(X_internal_train, y_internal_train)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    permutation = calculate_permutation_importance(
        model,
        X_validation,
        y_validation,
        n_repeats=10,
        random_state=42,
    )
    shap_results = calculate_shap_values(
        model,
        X_validation,
        max_samples=100,
        background_size=50,
        random_state=42,
        max_evals=SHAP_MAX_EVALS,
    )

    permutation.to_csv(
        REPORTS_DIR / "permutation-importance.csv",
        index=False,
    )
    shap_results["global_importance"].to_csv(
        REPORTS_DIR / "shap-global-importance.csv",
        index=False,
    )
    local_explanation = pd.DataFrame({
        "feature": shap_results["feature_names"],
        "customer_value": shap_results["data"].iloc[0].to_numpy(),
        "shap_value": shap_results["values"][0],
    }).sort_values("shap_value", key=lambda values: values.abs(), ascending=False)
    local_explanation.to_csv(
        REPORTS_DIR / "shap-local-example.csv",
        index=False,
    )

    global_importance = shap_results["global_importance"].head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    global_importance.sort_values("mean_absolute_shap").plot.barh(
        x="feature",
        y="mean_absolute_shap",
        ax=ax,
        legend=False,
    )
    ax.set_title("Global SHAP Importance for Churn Probability")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Model input feature")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "shap-global-importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    for feature in PDP_FEATURES:
        result = generate_partial_dependence(
            model,
            X_validation,
            feature,
            grid_size=20,
            output_path=FIGURES_DIR / f"pdp-{feature}.png",
        )
        plt.close(result["figure"])

    print(f"Selected algorithm: {metadata['algorithm']}")
    print(f"Explanation model training records: {len(X_internal_train)}")
    print(f"Validation records used: {len(X_validation)}")
    print(
        "The final persisted model was not explained directly; the selected "
        "algorithm was refit on internal training rows to keep validation "
        "records outside model fitting."
    )
    print("Original held-out test records were not used.")
    print(f"Global SHAP: {REPORTS_DIR / 'shap-global-importance.csv'}")
    print(f"Local SHAP example: {REPORTS_DIR / 'shap-local-example.csv'}")
    print(f"PDP figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
