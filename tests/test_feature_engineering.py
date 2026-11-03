import pandas as pd
import pytest

from src.feature_engineering import engineer_features


@pytest.fixture
def sample_dataset():
    return pd.DataFrame({
        "Age": [25, 35, 45, 55, 65],
        "Balance": [0, 25000, 50000, 0, 100000],
    })


def test_engineered_features_are_created(sample_dataset):

    result = engineer_features(sample_dataset)

    assert "HasBalance" in result.columns
    assert "AgeGroup" in result.columns


def test_has_balance_feature(sample_dataset):

    result = engineer_features(sample_dataset)

    assert result["HasBalance"].tolist() == [0, 1, 1, 0, 1]


def test_age_group_feature(sample_dataset):

    result = engineer_features(sample_dataset)

    assert result["AgeGroup"].astype(str).tolist() == [
        "<30",
        "30-39",
        "40-49",
        "50-59",
        "60+",
    ]


def test_original_dataset_is_not_modified(sample_dataset):

    original = sample_dataset.copy(deep=True)

    engineer_features(sample_dataset)

    pd.testing.assert_frame_equal(
        sample_dataset,
        original,
    )


def test_missing_age_column_raises_error():

    dataset = pd.DataFrame({
        "Balance": [1000]
    })

    with pytest.raises(ValueError, match="Missing required columns"):

        engineer_features(dataset)


def test_missing_balance_column_raises_error():

    dataset = pd.DataFrame({
        "Age": [30]
    })

    with pytest.raises(ValueError, match="Missing required columns"):

        engineer_features(dataset)


def test_missing_values_raise_error():

    dataset = pd.DataFrame({
        "Age": [30, None],
        "Balance": [1000, 2000],
    })

    with pytest.raises(ValueError, match="missing or non-numeric"):

        engineer_features(dataset)


def test_age_below_valid_range_raises_error():

    dataset = pd.DataFrame({
        "Age": [17],
        "Balance": [1000],
    })

    with pytest.raises(ValueError, match="between 18 and 100"):

        engineer_features(dataset)


def test_age_above_valid_range_raises_error():

    dataset = pd.DataFrame({
        "Age": [101],
        "Balance": [1000],
    })

    with pytest.raises(ValueError, match="between 18 and 100"):

        engineer_features(dataset)


def test_negative_balance_raises_error():

    dataset = pd.DataFrame({
        "Age": [30],
        "Balance": [-1000],
    })

    with pytest.raises(ValueError, match="Balance cannot"):

        engineer_features(dataset)


def test_non_numeric_age_raises_error():

    dataset = pd.DataFrame({
        "Age": ["unknown"],
        "Balance": [1000],
    })

    with pytest.raises(ValueError, match="missing or non-numeric"):

        engineer_features(dataset)


def test_age_boundary_values():

    dataset = pd.DataFrame({
        "Age": [18, 29, 30, 39, 40, 49, 50, 59, 60, 100],
        "Balance": [1000] * 10,
    })

    result = engineer_features(dataset)

    assert result["AgeGroup"].astype(str).tolist() == [
        "<30",
        "<30",
        "30-39",
        "30-39",
        "40-49",
        "40-49",
        "50-59",
        "50-59",
        "60+",
        "60+",
    ]


def test_infinite_values_raise_error():

    dataset = pd.DataFrame({
        "Age": [30],
        "Balance": [float("inf")],
    })

    with pytest.raises(ValueError, match="infinite values"):

        engineer_features(dataset)


def test_invalid_input_type_raises_error():

    with pytest.raises(TypeError, match="pandas DataFrame"):

        engineer_features([30, 40])

def test_fractional_age_raises_error():
    dataset = pd.DataFrame({
        "Age": [18.5],
        "Balance": [1000]
    })

    with pytest.raises(ValueError, match="integer"):
        engineer_features(dataset)
