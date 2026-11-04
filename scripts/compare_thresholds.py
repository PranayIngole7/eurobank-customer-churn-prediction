"""
Reproduce validation-based classification threshold comparison.

Data separation:
- 6,000 records: internal training
- 2,000 records: validation
- 2,000 records: original test (never used here)

The balanced Logistic Regression pipeline is fitted only
on the internal training dataset.
"""

import numpy as np
import pandas as pd

from src.data_split import split_dataset, split_training_data
from src.evaluation import (
    compare_thresholds,
    plot_threshold_comparison,
)
from src.feature_engineering import engineer_features
from src.modeling import build_baseline_models


RAW_DATA_PATH = "data/raw/european-bank.csv"
RESULTS_PATH = "reports/threshold-comparison.csv"
FIGURE_PATH = "reports/figures/threshold-comparison.png"


def main():
    # 1. Load the original dataset.
    raw_data = pd.read_csv(RAW_DATA_PATH)

    print("Original dataset shape:", raw_data.shape)

    # 2. Apply the existing feature-engineering workflow.
    processed_data = engineer_features(raw_data)

    print("Feature-engineered dataset shape:", processed_data.shape)

    # 3. Create the original stratified 80/20 split.
    X_train, X_test, y_train, y_test = split_dataset(
        processed_data
    )

    print("\nOriginal split:")
    print("Training records:", len(X_train))
    print("Reserved test records:", len(X_test))

    # 4. Split original training data into internal training
    #    and validation datasets.
    (
        X_internal_train,
        X_validation,
        y_internal_train,
        y_validation,
    ) = split_training_data(X_train, y_train)

    print("\nInternal split:")
    print("Internal training records:", len(X_internal_train))
    print("Validation records:", len(X_validation))

    print("\nValidation class distribution:")
    print(y_validation.value_counts().sort_index())

    # 5. Build and fit the balanced Logistic Regression pipeline.
    models = build_baseline_models()

    balanced_model = models["logistic_regression_balanced"]

    balanced_model.fit(
        X_internal_train,
        y_internal_train,
    )

    print("\nBalanced Logistic Regression fitted successfully.")

    # 6. Generate churn probabilities on validation data only.
    classes = list(balanced_model.classes_)
    positive_class_index = classes.index(1)

    churn_probabilities = balanced_model.predict_proba(
        X_validation
    )[:, positive_class_index]

    print(
        "Validation probabilities generated:",
        len(churn_probabilities),
    )

    # 7. Compare thresholds from 0.10 to 0.90.
    thresholds = np.round(
        np.arange(0.10, 0.901, 0.05),
        2,
    )

    comparison = compare_thresholds(
        y_validation,
        churn_probabilities,
        thresholds,
    )

    # 8. Save comparison results.
    comparison.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # 9. Save threshold comparison visualization.
    fig, ax = plot_threshold_comparison(
        comparison,
        output_path=FIGURE_PATH,
    )

    import matplotlib.pyplot as plt
    plt.close(fig)

    # 10. Display results.
    print("\nThreshold comparison results:")
    print(
        comparison.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1_score": "{:.4f}".format,
            },
        )
    )

    best_row = comparison.loc[
        comparison["f1_score"].idxmax()
    ]

    print("\nBest validation threshold by F1-score:")
    print(f"Threshold : {best_row['threshold']:.2f}")
    print(f"Precision : {best_row['precision']:.4f}")
    print(f"Recall    : {best_row['recall']:.4f}")
    print(f"F1-score  : {best_row['f1_score']:.4f}")

    print("\nConfusion matrix counts at selected threshold:")
    print(f"TN: {int(best_row['tn'])}")
    print(f"FP: {int(best_row['fp'])}")
    print(f"FN: {int(best_row['fn'])}")
    print(f"TP: {int(best_row['tp'])}")

    print("\nArtifacts saved:")
    print(RESULTS_PATH)
    print(FIGURE_PATH)

    print(
        "\nOriginal test data was reserved and not used "
        "for threshold selection."
    )


if __name__ == "__main__":
    main()
