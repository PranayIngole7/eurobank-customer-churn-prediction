import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from src.preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    validate_feature_data,
)


@pytest.fixture
def sample_features():
    """Return a small, valid customer feature dataset."""

    return pd.DataFrame(
        {
            "CreditScore": [600, 650, 700, 750],
            "Age": [25, 35, 45, 55],
            "Tenure": [2, 4, 6, 8],
            "Balance": [0.0, 1000.0, 2000.0, 3000.0],
            "NumOfProducts": [1, 2, 1, 2],
            "EstimatedSalary": [30000.0, 40000.0, 50000.0, 60000.0],
            "Geography": ["France", "Germany", "France", "Germany"],
            "Gender": ["Female", "Male", "Female", "Male"],
            "AgeGroup": ["<30", "30-39", "40-49", "50-59"],
            "HasCrCard": [1, 0, 1, 0],
            "IsActiveMember": [1, 1, 0, 0],
            "HasBalance": [0, 1, 1, 1],
            "BalanceToSalaryRatio": [0.0, 0.025, 0.04, 0.05],
            "ProductDensityIndicator": [1.0, 0.5, 1 / 6, 0.25],
            "EngagementProductInteraction": [1, 2, 0, 0],
            "AgeTenureInteraction": [50, 140, 270, 440],
        }
    )


def test_valid_feature_data_passes(sample_features):
    validate_feature_data(sample_features)


def test_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_feature_data([1, 2, 3])


def test_missing_feature_column_raises_value_error(sample_features):
    invalid_data = sample_features.drop(columns=["Age"])

    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_feature_data(invalid_data)


def test_missing_values_raise_value_error(sample_features):
    invalid_data = sample_features.copy()
    invalid_data.loc[0, "Balance"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        validate_feature_data(invalid_data)


@pytest.mark.parametrize(
    "invalid_value",
    [np.inf, -np.inf, "not-a-number"],
)
def test_invalid_numerical_values_raise_value_error(
    sample_features,
    invalid_value,
):
    invalid_data = sample_features.copy()
    invalid_data["CreditScore"] = invalid_data["CreditScore"].astype(object)
    invalid_data.loc[0, "CreditScore"] = invalid_value

    with pytest.raises(ValueError, match="finite numerical values"):
        validate_feature_data(invalid_data)


@pytest.mark.parametrize(
    "column",
    ["HasCrCard", "IsActiveMember", "HasBalance"],
)
def test_binary_features_reject_invalid_values(
    sample_features,
    column,
):
    invalid_data = sample_features.copy()
    invalid_data.loc[0, column] = 2

    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_feature_data(invalid_data)


def test_extra_columns_are_allowed_but_not_required(sample_features):
    extended_data = sample_features.copy()
    extended_data["CustomerId"] = [1001, 1002, 1003, 1004]
    extended_data["Exited"] = [0, 1, 0, 1]

    validate_feature_data(extended_data)


def test_feature_columns_contain_expected_features():
    assert len(FEATURE_COLUMNS) == 16
    assert len(set(FEATURE_COLUMNS)) == 16


def test_build_preprocessor_returns_unfitted_transformer():
    preprocessor = build_preprocessor()

    assert not hasattr(preprocessor, "transformers_")


def test_build_preprocessor_returns_fresh_instances():
    first = build_preprocessor()
    second = build_preprocessor()

    assert first is not second


def test_preprocessor_transforms_features(sample_features):
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(sample_features)

    assert transformed.shape[0] == len(sample_features)
    assert transformed.shape[1] > len(FEATURE_COLUMNS)


def test_preprocessor_does_not_modify_original_data(sample_features):
    original = sample_features.copy(deep=True)

    preprocessor = build_preprocessor()
    preprocessor.fit_transform(sample_features)

    pd.testing.assert_frame_equal(sample_features, original)


def test_preprocessor_handles_unknown_categories(sample_features):
    train_data = sample_features.iloc[:3].copy()

    test_data = sample_features.iloc[[3]].copy()
    test_data["Geography"] = "UnknownCountry"
    test_data["Gender"] = "UnknownGender"
    test_data["AgeGroup"] = "UnknownAgeGroup"

    preprocessor = build_preprocessor()
    preprocessor.fit(train_data)

    transformed = preprocessor.transform(test_data)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == preprocessor.transform(train_data).shape[1]


def test_binary_features_pass_through_unchanged(sample_features):
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(sample_features)

    if issparse(transformed):
        transformed = transformed.toarray()

    binary_start = (
        len(preprocessor.transformers_[0][2])
        + len(preprocessor.named_transformers_["categorical"]
              .named_steps["encoder"].get_feature_names_out())
    )

    expected_binary = sample_features[
        ["HasCrCard", "IsActiveMember", "HasBalance"]
    ].to_numpy()

    actual_binary = transformed[:, binary_start:]

    np.testing.assert_array_equal(actual_binary, expected_binary)
