"""Generate churn risk scores for held-out customers."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_split import split_dataset
from src.feature_engineering import engineer_features
from src.modeling import build_baseline_models
from src.risk_scoring import score_customers

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "european-bank.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "customer-risk-scores.csv"

CHURN_THRESHOLD = 0.65


def main():
    # Load and prepare customer data.
    raw_data = pd.read_csv(DATA_PATH)
    processed_data = engineer_features(raw_data)

    # Reproduce the original stratified train/test split.
    X_train, X_test, y_train, _ = split_dataset(processed_data)

    # Train the selected balanced Logistic Regression model.
    model = build_baseline_models()["logistic_regression_balanced"]
    model.fit(X_train, y_train)

    # Preserve customer IDs separately from model features.
    customer_ids = processed_data.loc[X_test.index, "CustomerId"]

    # Actual churn labels are deliberately not passed to the scoring function.
    scores = score_customers(
        model,
        X_test,
        threshold=CHURN_THRESHOLD,
    )

    results = scores.copy()
    results.insert(0, "CustomerId", customer_ids)

    # Sort customers by descending risk score.
    # CustomerId provides deterministic ordering when scores are tied.
    results = results.sort_values(
        ["risk_score", "CustomerId"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # Assign sequential ranks after sorting.
    results.insert(1, "risk_rank", range(1, len(results) + 1))

    # Save the customer risk scores.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"Customers scored: {len(results)}")
    print(f"Classification threshold: {CHURN_THRESHOLD}")

    print("\nRisk category distribution:")
    print(
        results["risk_category"]
        .value_counts()
        .reindex(["Low", "Moderate", "High", "Critical"], fill_value=0)
    )

    print("\nPredicted churn distribution:")
    print(results["churn_prediction"].value_counts().sort_index())

    print(f"\nRisk scores saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
