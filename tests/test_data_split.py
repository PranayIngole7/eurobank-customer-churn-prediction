import pandas as pd
import pytest

from src.data_split import split_dataset
from src.preprocessing import FEATURE_COLUMNS


@pytest.fixture
def sample_dataset():
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
            "AgeGroup": ["<30", "30-39", "40-49", "50-59", "60+"][index % 5],
            "HasCrCard": index % 2,
            "IsActiveMember": (index + 1) % 2,
            "HasBalance": int(index > 0),
            "Exited": index % 2,
        })

    return pd.DataFrame(rows)


def test_split_returns_four_datasets(sample_dataset):
    result = split_dataset(sample_dataset)

    assert len(result) == 4


def test_split_has_expected_sizes(sample_dataset):
    X_train, X_test, y_train, y_test = split_dataset(sample_dataset)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_split_preserves_feature_columns(sample_dataset):
    X_train, X_test, _, _ = split_dataset(sample_dataset)

    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_test.columns) == FEATURE_COLUMNS


def test_split_preserves_both_target_classes(sample_dataset):
    _, _, y_train, y_test = split_dataset(sample_dataset)

    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_split_preserves_target_proportions(sample_dataset):
    _, _, y_train, y_test = split_dataset(sample_dataset)

    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    assert abs(train_ratio - test_ratio) <= 0.01


def test_split_is_reproducible(sample_dataset):
    first = split_dataset(sample_dataset)
    second = split_dataset(sample_dataset)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_series_equal(first[2], second[2])
    pd.testing.assert_series_equal(first[3], second[3])


def test_split_does_not_modify_input(sample_dataset):
    original = sample_dataset.copy(deep=True)

    split_dataset(sample_dataset)

    pd.testing.assert_frame_equal(sample_dataset, original)


def test_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError):
        split_dataset([1, 2, 3])


def test_missing_target_raises_value_error(sample_dataset):
    sample_dataset = sample_dataset.drop(columns=["Exited"])

    with pytest.raises(ValueError, match="Missing required columns"):
        split_dataset(sample_dataset)


def test_missing_feature_raises_value_error(sample_dataset):
    sample_dataset = sample_dataset.drop(columns=["Age"])

    with pytest.raises(ValueError, match="Missing required columns"):
        split_dataset(sample_dataset)


def test_missing_target_values_raise_value_error(sample_dataset):
    sample_dataset.loc[0, "Exited"] = None

    with pytest.raises(ValueError, match="Target column contains missing"):
        split_dataset(sample_dataset)


def test_invalid_target_values_raise_value_error(sample_dataset):
    sample_dataset.loc[0, "Exited"] = 2

    with pytest.raises(ValueError, match="Target must contain only 0 and 1"):
        split_dataset(sample_dataset)


def test_single_target_class_raises_value_error(sample_dataset):
    sample_dataset["Exited"] = 0

    with pytest.raises(ValueError, match="must contain both classes"):
        split_dataset(sample_dataset)


def test_missing_feature_values_raise_value_error(sample_dataset):
    sample_dataset.loc[0, "Age"] = None

    with pytest.raises(ValueError, match="Feature data contains missing"):
        split_dataset(sample_dataset)
