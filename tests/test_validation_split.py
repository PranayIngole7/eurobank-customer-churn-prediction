import pandas as pd
import pytest

from src.data_split import split_training_data
from src.preprocessing import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
)


def create_sample_data():
    """Create a valid dataset for validation split tests."""

    size = 100

    data = {}

    for column in NUMERICAL_FEATURES:
        data[column] = list(range(size))

    data["Geography"] = (["France", "Germany", "Spain"] * 34)[:size]
    data["Gender"] = ["Male", "Female"] * 50
    data["AgeGroup"] = ["30-39", "40-49", "50-59", "<30", "60+"] * 20

    data["HasCrCard"] = [0, 1] * 50
    data["IsActiveMember"] = [1, 0] * 50
    data["HasBalance"] = [1, 0] * 50

    X = pd.DataFrame(data, columns=FEATURE_COLUMNS)

    y = pd.Series(
        [0] * 80 + [1] * 20,
        name="Exited",
    )

    return X, y


def test_returns_four_datasets():
    X, y = create_sample_data()

    result = split_training_data(X, y)

    assert len(result) == 4


def test_split_has_expected_sizes():
    X, y = create_sample_data()

    X_train, X_validation, y_train, y_validation = (
        split_training_data(X, y)
    )

    assert len(X_train) == 75
    assert len(X_validation) == 25
    assert len(y_train) == 75
    assert len(y_validation) == 25


def test_preserves_feature_columns():
    X, y = create_sample_data()

    X_train, X_validation, _, _ = split_training_data(X, y)

    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_validation.columns) == FEATURE_COLUMNS


def test_preserves_both_target_classes():
    X, y = create_sample_data()

    _, _, y_train, y_validation = split_training_data(X, y)

    assert set(y_train.unique()) == {0, 1}
    assert set(y_validation.unique()) == {0, 1}


def test_preserves_class_proportions():
    X, y = create_sample_data()

    _, _, y_train, y_validation = split_training_data(X, y)

    assert abs(y_train.mean() - y.mean()) < 0.02
    assert abs(y_validation.mean() - y.mean()) < 0.02


def test_split_is_reproducible():
    X, y = create_sample_data()

    first = split_training_data(X, y)
    second = split_training_data(X, y)

    for first_part, second_part in zip(first, second):
        if isinstance(first_part, pd.DataFrame):
            pd.testing.assert_frame_equal(first_part, second_part)
        else:
            pd.testing.assert_series_equal(first_part, second_part)


def test_does_not_modify_input_data():
    X, y = create_sample_data()

    original_X = X.copy(deep=True)
    original_y = y.copy(deep=True)

    split_training_data(X, y)

    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_internal_training_and_validation_do_not_overlap():
    X, y = create_sample_data()

    X_train, X_validation, _, _ = split_training_data(X, y)

    assert set(X_train.index).isdisjoint(set(X_validation.index))


def test_rejects_invalid_feature_input_type():
    _, y = create_sample_data()

    with pytest.raises(TypeError, match="X_train must be a pandas DataFrame"):
        split_training_data([], y)


def test_rejects_invalid_target_input_type():
    X, _ = create_sample_data()

    with pytest.raises(TypeError, match="y_train must be a pandas Series"):
        split_training_data(X, [0, 1])


def test_rejects_unequal_lengths():
    X, y = create_sample_data()

    with pytest.raises(ValueError, match="equal lengths"):
        split_training_data(X, y.iloc[:-1])


def test_rejects_mismatched_indexes():
    X, y = create_sample_data()
    y.index = y.index + 1

    with pytest.raises(ValueError, match="indexes must match"):
        split_training_data(X, y)


def test_rejects_missing_feature_values():
    X, y = create_sample_data()
    X.loc[0, "Age"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        split_training_data(X, y)


def test_rejects_missing_target_values():
    X, y = create_sample_data()
    y.iloc[0] = float("nan")

    with pytest.raises(ValueError, match="Target contains missing values"):
        split_training_data(X, y)


def test_rejects_invalid_target_values():
    X, y = create_sample_data()
    y.iloc[0] = 2

    with pytest.raises(ValueError, match="only 0 and 1"):
        split_training_data(X, y)


def test_rejects_single_target_class():
    X, y = create_sample_data()
    y[:] = 0

    with pytest.raises(ValueError, match="both classes"):
        split_training_data(X, y)
