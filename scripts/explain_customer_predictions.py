"""Generate SHAP explanations for the highest-risk validation customers."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_split import split_dataset, split_training_data
from src.explainability import calculate_shap_values
from src.feature_engineering import engineer_features
from src.model_persistence import get_model_metadata
from src.modeling import build_candidate_models


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "european-bank.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TOP_CUSTOMERS = 5
SHAP_MAX_EVALS = 33


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    processed_data = engineer_features(pd.read_csv(DATA_PATH))
    X_train, _, y_train, _ = split_dataset(processed_data)
    X_internal_train, X_validation, y_internal_train, _ = split_training_data(
        X_train,
        y_train,
    )
    metadata = get_model_metadata()

    model = build_candidate_models()[metadata["model_name"]]
    model.fit(X_internal_train, y_internal_train)
    probabilities = model.predict_proba(X_validation)[:, list(model.classes_).index(1)]
    selected_positions = np.argsort(-probabilities, kind="stable")[:TOP_CUSTOMERS]
    selected_indices = X_validation.index[selected_positions]

    explanations = calculate_shap_values(
        model,
        X_validation,
        max_samples=TOP_CUSTOMERS,
        background_size=50,
        random_state=42,
        sample_indices=selected_indices,
        max_evals=SHAP_MAX_EVALS,
    )

    summaries = []
    contributions = []
    for rank, position in enumerate(selected_positions, start=1):
        explanation_values = explanations["values"][rank - 1]
        customer = explanations["data"].iloc[rank - 1]
        contribution_frame = pd.DataFrame({
            "feature": explanations["feature_names"],
            "customer_value": customer.to_numpy(),
            "shap_value": explanation_values,
        })
        contribution_frame["absolute_shap"] = (
            contribution_frame["shap_value"].abs()
        )
        contribution_frame["direction"] = np.where(
            contribution_frame["shap_value"] > 0,
            "Increases modeled churn probability",
            np.where(
                contribution_frame["shap_value"] < 0,
                "Decreases modeled churn probability",
                "No modeled contribution",
            ),
        )
        contribution_frame.insert(0, "risk_rank", rank)
        contribution_frame.insert(
            1,
            "validation_record_index",
            int(selected_indices[rank - 1]),
        )
        contributions.append(contribution_frame)
        summaries.append({
            "risk_rank": rank,
            "validation_record_index": int(selected_indices[rank - 1]),
            "predicted_probability": float(probabilities[position]),
        })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "local-customer-summary.csv"
    contributions_path = REPORTS_DIR / "local-customer-contributions.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.concat(contributions, ignore_index=True).to_csv(
        contributions_path,
        index=False,
    )

    print(f"Selected algorithm: {metadata['algorithm']}")
    print(f"Internal training records: {len(X_internal_train)}")
    print(f"Validation customers explained: {len(summaries)}")
    print("Validation labels used for customer selection: No")
    print("Original held-out test data used: No")
    print(f"Local explanations: {contributions_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
