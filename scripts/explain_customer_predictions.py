"""
Generate local explanations for high-risk validation customers.

The model is trained only on the internal training split.
Validation labels are not used to select or explain customers.
The original test split is excluded.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_split import split_dataset, split_training_data
from src.explainability import explain_customer_prediction
from src.feature_engineering import engineer_features
from src.modeling import build_baseline_models


DATA_PATH = Path("data/raw/european-bank.csv")
REPORTS_DIR = Path("reports")

TOP_CUSTOMERS = 5
TOP_CONTRIBUTIONS = 5


def main():
    # 1. Load and prepare the original dataset.
    raw_data = pd.read_csv(DATA_PATH)
    processed_data = engineer_features(raw_data)

    # 2. Create the original training/test split.
    #    The original test split is deliberately discarded.
    X_train, _, y_train, _ = split_dataset(processed_data)

    # 3. Create internal training/validation splits.
    (
        X_internal_train,
        X_validation,
        y_internal_train,
        _,
    ) = split_training_data(X_train, y_train)

    # 4. Fit the selected model on internal training data only.
    model = build_baseline_models()["logistic_regression_balanced"]
    model.fit(X_internal_train, y_internal_train)

    # 5. Predict validation probabilities.
    probabilities = model.predict_proba(X_validation)[:, 1]

    # Select customers using predicted probabilities only.
    selected_positions = np.argsort(
        -probabilities,
        kind="stable",
    )[:TOP_CUSTOMERS]

    summaries = []
    all_contributions = []

    # 6. Explain each selected customer.
    for rank, position in enumerate(selected_positions, start=1):
        customer_features = X_validation.iloc[[position]]
        record_index = customer_features.index[0]

        explanation = explain_customer_prediction(
            model,
            customer_features,
        )

        summary = {
            "risk_rank": rank,
            "validation_record_index": record_index,
            "predicted_probability": explanation[
                "predicted_probability"
            ],
            "predicted_log_odds": explanation[
                "predicted_log_odds"
            ],
            "base_log_odds": explanation["base_log_odds"],
        }

        summaries.append(summary)

        contributions = explanation["contributions"].copy()
        contributions.insert(0, "risk_rank", rank)
        contributions.insert(1, "validation_record_index", record_index)

        all_contributions.append(contributions)

    # 7. Save the explanation results.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = REPORTS_DIR / "local-customer-summary.csv"
    contributions_path = REPORTS_DIR / "local-customer-contributions.csv"

    summary_df = pd.DataFrame(summaries)
    contributions_df = pd.concat(
        all_contributions,
        ignore_index=True,
    )

    summary_df.to_csv(summary_path, index=False)
    contributions_df.to_csv(contributions_path, index=False)

    # 8. Display results.
    print("Local Explainability Analysis")
    print("-----------------------------")
    print(f"Internal training records: {len(X_internal_train)}")
    print(f"Validation records:        {len(X_validation)}")
    print(f"Customers explained:       {len(summary_df)}")
    print("Validation labels used:    No")
    print("Original test records:     Excluded")

    print("\nSelected Customers")
    print("------------------")
    print(summary_df.to_string(index=False))

    for rank in range(1, len(summary_df) + 1):
        customer_contributions = (
            contributions_df[
                contributions_df["risk_rank"] == rank
            ]
            .sort_values(
                "absolute_contribution",
                ascending=False,
            )
            .head(TOP_CONTRIBUTIONS)
        )

        print(f"\nCustomer Risk Rank {rank}: Top Contributions")
        print("--------------------------------------------")
        print(
            customer_contributions[
                [
                    "feature",
                    "contribution",
                    "direction",
                ]
            ].to_string(index=False)
        )

    print("\nGenerated Artifacts")
    print("-------------------")
    print(summary_path)
    print(contributions_path)

    print("\nLocal explainability analysis completed.")


if __name__ == "__main__":
    main()
