import numpy as np
import pandas as pd
import pytest

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.explainability import (
    calculate_permutation_importance,
    calculate_shap_values,
    explain_customer_prediction,
    extract_model_coefficients,
    generate_partial_dependence,
)
from src.modeling import build_baseline_models
from src.preprocessing import FEATURE_COLUMNS


@pytest.fixture
def sample_training_data():
    rows = []

    for index in range(100):
        rows.append({
            "CreditScore": 600 + index,
            "Age": 25 + (index % 40),
            "Tenure": index % 11,
            "Balance": 1000.0 * index,
            "NumOfProducts": 1 + (index % 2),
            "EstimatedSalary": 30000.0 + index,
            "Geography": ["France", "Germany", "Spain"][index % 3],
            "Gender": ["Male", "Female"][index % 2],
            "AgeGroup": [
                "<30",
                "30-39",
                "40-49",
                "50-59",
                "60+",
            ][index % 5],
            "HasCrCard": index % 2,
            "IsActiveMember": (index + 1) % 2,
            "HasBalance": int(index > 0),
            "BalanceToSalaryRatio": (1000.0 * index) / (30000.0 + index),
            "ProductDensityIndicator": (
                (1 + (index % 2)) / (index % 11)
                if index % 11 > 0 else 0.0
            ),
            "EngagementProductInteraction": (
                (index + 1) % 2
            ) * (1 + (index % 2)),
            "AgeTenureInteraction": (25 + (index % 40)) * (index % 11),
            "Exited": index % 2,
        })

    df = pd.DataFrame(rows)

    X = df[FEATURE_COLUMNS]
    y = df["Exited"]

    return X, y


@pytest.fixture
def fitted_model(sample_training_data):
    X, y = sample_training_data

    model = build_baseline_models()["logistic_regression_balanced"]
    model.fit(X, y)

    return model


def test_extract_model_coefficients_returns_dataframe(fitted_model):
    results = extract_model_coefficients(fitted_model)

    assert isinstance(results, pd.DataFrame)

    assert list(results.columns) == [
        "feature",
        "coefficient",
        "odds_ratio",
        "absolute_coefficient",
        "direction",
    ]


def test_extract_model_coefficients_returns_all_features(fitted_model):
    results = extract_model_coefficients(fitted_model)

    expected_features = (
        fitted_model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    assert set(results["feature"]) == set(expected_features)
    assert len(results) == len(expected_features)


def test_extract_model_coefficients_calculates_odds_ratios(
    fitted_model,
):
    results = extract_model_coefficients(fitted_model)

    expected_odds_ratios = np.exp(results["coefficient"])

    np.testing.assert_allclose(
        results["odds_ratio"],
        expected_odds_ratios,
    )


def test_extract_model_coefficients_calculates_absolute_values(
    fitted_model,
):
    results = extract_model_coefficients(fitted_model)

    np.testing.assert_allclose(
        results["absolute_coefficient"],
        np.abs(results["coefficient"]),
    )


def test_extract_model_coefficients_sorts_by_absolute_value(
    fitted_model,
):
    results = extract_model_coefficients(fitted_model)

    assert results["absolute_coefficient"].is_monotonic_decreasing


def test_extract_model_coefficients_assigns_correct_directions(
    fitted_model,
):
    results = extract_model_coefficients(fitted_model)

    for _, row in results.iterrows():
        if row["coefficient"] > 0:
            assert row["direction"] == "Higher churn odds"

        elif row["coefficient"] < 0:
            assert row["direction"] == "Lower churn odds"

        else:
            assert row["direction"] == "No directional association"


def test_extract_model_coefficients_rejects_non_pipeline():
    with pytest.raises(TypeError, match="Pipeline"):
        extract_model_coefficients("invalid model")


def test_extract_model_coefficients_rejects_missing_steps():
    model = Pipeline([])

    with pytest.raises(ValueError, match="preprocessor"):
        extract_model_coefficients(model)


def test_extract_model_coefficients_rejects_unsupported_classifier(
    sample_training_data,
):
    X, y = sample_training_data

    model = build_baseline_models()["logistic_regression"]
    model.set_params(classifier=DummyClassifier())

    model.fit(X, y)

    with pytest.raises(ValueError, match="LogisticRegression"):
        extract_model_coefficients(model)


def test_extract_model_coefficients_rejects_unfitted_classifier():
    model = build_baseline_models()["logistic_regression_balanced"]

    with pytest.raises(ValueError, match="fitted"):
        extract_model_coefficients(model)


def test_extract_model_coefficients_rejects_missing_preprocessor():
    model = Pipeline([
        ("classifier", DummyClassifier())
    ])

    with pytest.raises(ValueError, match="preprocessor"):
        extract_model_coefficients(model)

def test_permutation_importance_returns_dataframe(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    results = calculate_permutation_importance(
        fitted_model,
        X,
        y,
    )

    assert isinstance(results, pd.DataFrame)

    assert list(results.columns) == [
        "feature",
        "importance_mean",
        "importance_std",
    ]


def test_permutation_importance_returns_all_original_features(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    results = calculate_permutation_importance(
        fitted_model,
        X,
        y,
    )

    assert set(results["feature"]) == set(X.columns)
    assert len(results) == len(X.columns)


def test_permutation_importance_returns_finite_values(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    results = calculate_permutation_importance(
        fitted_model,
        X,
        y,
    )

    assert np.isfinite(results["importance_mean"]).all()
    assert np.isfinite(results["importance_std"]).all()


def test_permutation_importance_sorts_by_mean_importance(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    results = calculate_permutation_importance(
        fitted_model,
        X,
        y,
    )

    assert results["importance_mean"].is_monotonic_decreasing


def test_permutation_importance_is_reproducible(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    first_result = calculate_permutation_importance(
        fitted_model,
        X,
        y,
        n_repeats=5,
        random_state=42,
    )

    second_result = calculate_permutation_importance(
        fitted_model,
        X,
        y,
        n_repeats=5,
        random_state=42,
    )

    pd.testing.assert_frame_equal(
        first_result,
        second_result,
    )


def test_permutation_importance_rejects_non_pipeline(
    sample_training_data,
):
    X, y = sample_training_data

    with pytest.raises(TypeError, match="Pipeline"):
        calculate_permutation_importance(
            "invalid model",
            X,
            y,
        )


def test_permutation_importance_rejects_invalid_feature_type(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    with pytest.raises(TypeError, match="DataFrame"):
        calculate_permutation_importance(
            fitted_model,
            X.to_numpy(),
            y,
        )


def test_permutation_importance_rejects_invalid_target_type(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    with pytest.raises(TypeError, match="Series"):
        calculate_permutation_importance(
            fitted_model,
            X,
            y.to_numpy(),
        )


def test_permutation_importance_rejects_mismatched_lengths(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    with pytest.raises(ValueError, match="equal lengths"):
        calculate_permutation_importance(
            fitted_model,
            X,
            y.iloc[:-1],
        )


def test_permutation_importance_rejects_mismatched_indexes(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    mismatched_y = y.copy()
    mismatched_y.index = range(1000, 1100)

    with pytest.raises(ValueError, match="indexes must match"):
        calculate_permutation_importance(
            fitted_model,
            X,
            mismatched_y,
        )


def test_permutation_importance_rejects_single_class_target(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    single_class_y = pd.Series(
        np.zeros(len(y), dtype=int),
        index=y.index,
    )

    with pytest.raises(ValueError, match="both classes"):
        calculate_permutation_importance(
            fitted_model,
            X,
            single_class_y,
        )


@pytest.mark.parametrize(
    "n_repeats",
    [0, -1, 1.5, True, "10", None],
)
def test_permutation_importance_rejects_invalid_repeats(
    fitted_model,
    sample_training_data,
    n_repeats,
):
    X, y = sample_training_data

    with pytest.raises(ValueError, match="n_repeats"):
        calculate_permutation_importance(
            fitted_model,
            X,
            y,
            n_repeats=n_repeats,
        )


@pytest.mark.parametrize(
    "random_state",
    [-1, 1.5, True, "42", None],
)
def test_permutation_importance_rejects_invalid_random_state(
    fitted_model,
    sample_training_data,
    random_state,
):
    X, y = sample_training_data

    with pytest.raises(ValueError, match="random_state"):
        calculate_permutation_importance(
            fitted_model,
            X,
            y,
            random_state=random_state,
        )


def test_permutation_importance_rejects_unfitted_model(
    sample_training_data,
):
    X, y = sample_training_data

    model = build_baseline_models()["logistic_regression_balanced"]

    with pytest.raises(Exception):
        calculate_permutation_importance(
            model,
            X,
            y,
        )


def test_permutation_importance_rejects_wrong_feature_columns(
    fitted_model,
    sample_training_data,
):
    X, y = sample_training_data

    X_wrong = X.rename(
        columns={"Age": "CustomerAge"}
    )

    with pytest.raises(
        ValueError,
        match="Missing required feature columns",
    ):
        calculate_permutation_importance(
            fitted_model,
            X_wrong,
            y,
        )

def test_explain_customer_prediction_returns_expected_structure(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    result = explain_customer_prediction(
        fitted_model,
        X.iloc[[0]],
    )

    assert isinstance(result, dict)

    assert set(result.keys()) == {
        "base_log_odds",
        "predicted_log_odds",
        "predicted_probability",
        "contributions",
    }

    assert isinstance(result["contributions"], pd.DataFrame)

    assert list(result["contributions"].columns) == [
        "feature",
        "transformed_value",
        "coefficient",
        "contribution",
        "direction",
        "absolute_contribution",
    ]


def test_explain_customer_prediction_returns_valid_probability(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    result = explain_customer_prediction(
        fitted_model,
        X.iloc[[0]],
    )

    assert 0 <= result["predicted_probability"] <= 1


def test_explain_customer_prediction_reconstructs_log_odds(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    result = explain_customer_prediction(
        fitted_model,
        X.iloc[[0]],
    )

    contributions = result["contributions"]["contribution"].sum()

    reconstructed_log_odds = (
        result["base_log_odds"] + contributions
    )

    np.testing.assert_allclose(
        reconstructed_log_odds,
        result["predicted_log_odds"],
        rtol=1e-7,
        atol=1e-9,
    )


def test_explain_customer_prediction_probability_matches_log_odds(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    result = explain_customer_prediction(
        fitted_model,
        X.iloc[[0]],
    )

    log_odds = result["predicted_log_odds"]

    expected_probability = 1 / (1 + np.exp(-log_odds))

    np.testing.assert_allclose(
        result["predicted_probability"],
        expected_probability,
    )


def test_explain_customer_prediction_returns_all_transformed_features(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    result = explain_customer_prediction(
        fitted_model,
        X.iloc[[0]],
    )

    expected_features = (
        fitted_model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    assert set(result["contributions"]["feature"]) == set(
        expected_features
    )


def test_explain_customer_prediction_sorts_by_absolute_contribution(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    result = explain_customer_prediction(
        fitted_model,
        X.iloc[[0]],
    )

    assert result["contributions"][
        "absolute_contribution"
    ].is_monotonic_decreasing


def test_explain_customer_prediction_assigns_correct_directions(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    result = explain_customer_prediction(
        fitted_model,
        X.iloc[[0]],
    )

    for _, row in result["contributions"].iterrows():
        if row["contribution"] > 0:
            assert row["direction"] == "Increases churn log odds"

        elif row["contribution"] < 0:
            assert row["direction"] == "Decreases churn log odds"

        else:
            assert row["direction"] == "No contribution"


def test_explain_customer_prediction_rejects_non_pipeline():
    with pytest.raises(TypeError, match="Pipeline"):
        explain_customer_prediction(
            "invalid model",
            pd.DataFrame(),
        )


def test_explain_customer_prediction_rejects_invalid_feature_type(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    with pytest.raises(TypeError, match="DataFrame"):
        explain_customer_prediction(
            fitted_model,
            X.iloc[0].to_numpy(),
        )


def test_explain_customer_prediction_rejects_empty_features(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    with pytest.raises(ValueError, match="cannot be empty"):
        explain_customer_prediction(
            fitted_model,
            X.iloc[0:0],
        )


def test_explain_customer_prediction_rejects_multiple_customers(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    with pytest.raises(ValueError, match="exactly one customer"):
        explain_customer_prediction(
            fitted_model,
            X.iloc[:2],
        )


def test_explain_customer_prediction_rejects_unfitted_model(
    sample_training_data,
):
    X, _ = sample_training_data

    model = build_baseline_models()["logistic_regression_balanced"]

    with pytest.raises(ValueError, match="Fitted model"):
        explain_customer_prediction(
            model,
            X.iloc[[0]],
        )


def test_explain_customer_prediction_rejects_invalid_features(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    invalid_customer = X.iloc[[0]].drop(columns=["Age"])

    with pytest.raises(ValueError, match="Missing required feature"):
        explain_customer_prediction(
            fitted_model,
            invalid_customer,
        )


def test_shap_values_are_positive_class_and_feature_aligned(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    results = calculate_shap_values(
        fitted_model,
        X.iloc[:12],
        max_samples=4,
        background_size=4,
        random_state=11,
    )

    assert results["values"].shape == (4, len(FEATURE_COLUMNS))
    assert results["feature_names"] == list(FEATURE_COLUMNS)
    assert list(results["global_importance"]["feature"]) == list(
        results["global_importance"]
        .sort_values("mean_absolute_shap", ascending=False)["feature"]
    )
    assert np.isfinite(results["values"]).all()
    np.testing.assert_allclose(
        results["base_values"] + results["values"].sum(axis=1),
        results["predictions"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert np.max(np.abs(results["additivity_residual"])) <= 1e-6
    assert results["max_evals"] == 2 * len(FEATURE_COLUMNS) + 1


def test_shap_evaluation_budget_is_configurable(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    results = calculate_shap_values(
        fitted_model,
        X.iloc[:2],
        max_samples=2,
        background_size=3,
        max_evals=330,
    )

    assert results["max_evals"] == 330


@pytest.mark.parametrize(
    "max_evals",
    [0, 32, 1.5, True, "330"],
)
def test_shap_rejects_invalid_evaluation_budgets(
    fitted_model,
    sample_training_data,
    max_evals,
):
    X, _ = sample_training_data
    with pytest.raises(ValueError, match="max_evals"):
        calculate_shap_values(
            fitted_model,
            X.iloc[:2],
            max_samples=2,
            background_size=3,
            max_evals=max_evals,
        )


def test_shap_values_can_explain_explicit_validation_rows(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data
    requested_rows = pd.Index([4, 2])

    results = calculate_shap_values(
        fitted_model,
        X,
        max_samples=2,
        background_size=5,
        sample_indices=requested_rows,
    )

    pd.testing.assert_frame_equal(
        results["data"],
        X.loc[requested_rows].reset_index(drop=True),
    )
    assert results["values"].shape == (2, len(FEATURE_COLUMNS))


def test_shap_rejects_sample_indices_not_in_input(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    with pytest.raises(ValueError, match="refer to rows in X"):
        calculate_shap_values(
            fitted_model,
            X,
            sample_indices=pd.Index([1000]),
        )


def test_shap_rejects_feature_order_mismatch(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data
    reversed_features = X.iloc[:4].loc[:, list(reversed(X.columns))]

    with pytest.raises(ValueError, match="columns must match"):
        calculate_shap_values(fitted_model, reversed_features)


def test_shap_dependency_failure_is_reported(
    fitted_model,
    sample_training_data,
    monkeypatch,
):
    import sys

    X, _ = sample_training_data
    monkeypatch.setitem(sys.modules, "shap", None)

    with pytest.raises(ImportError, match="SHAP is required"):
        calculate_shap_values(fitted_model, X.iloc[:2])


def test_partial_dependence_uses_fitted_model_and_saves_plot(
    fitted_model,
    sample_training_data,
    tmp_path,
):
    X, _ = sample_training_data
    output_path = tmp_path / "pdp" / "balance-ratio.png"

    result = generate_partial_dependence(
        fitted_model,
        X,
        "BalanceToSalaryRatio",
        grid_size=5,
        output_path=output_path,
    )

    assert result["feature"] == "BalanceToSalaryRatio"
    assert result["figure"] is not None
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_partial_dependence_rejects_unknown_feature(
    fitted_model,
    sample_training_data,
):
    X, _ = sample_training_data

    with pytest.raises(ValueError, match="not present"):
        generate_partial_dependence(
            fitted_model,
            X,
            "unknown_feature",
        )
