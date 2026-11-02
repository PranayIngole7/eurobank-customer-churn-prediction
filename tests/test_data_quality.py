import pandas as pd
import pytest

from src.data_quality import validate_dataset


@pytest.fixture
def valid_dataset():
    """Create a small valid dataset for testing."""

    return pd.DataFrame({
        "Year": [2025, 2025],
        "CustomerId": [10001, 10002],
        "Surname": ["Smith", "Brown"],
        "CreditScore": [650, 720],
        "Geography": ["France", "Germany"],
        "Gender": ["Male", "Female"],
        "Age": [35, 42],
        "Tenure": [5, 7],
        "Balance": [50000.0, 100000.0],
        "NumOfProducts": [1, 2],
        "HasCrCard": [1, 0],
        "IsActiveMember": [1, 0],
        "EstimatedSalary": [75000.0, 90000.0],
        "Exited": [0, 1],
    })


def get_status(results, check_name):
    """Return the status of a particular validation check."""

    for result in results:
        if result["check"] == check_name:
            return result["status"]

    raise AssertionError(f"Check not found: {check_name}")


def test_valid_dataset_passes_all_checks(valid_dataset):
    results = validate_dataset(valid_dataset)

    assert len(results) == 26
    assert all(result["status"] == "PASS" for result in results)


def test_missing_required_column_is_detected(valid_dataset):
    dataset = valid_dataset.drop(columns=["Age"])

    results = validate_dataset(dataset)

    assert get_status(results, "Required columns") == "FAIL"


def test_missing_values_are_detected(valid_dataset):
    dataset = valid_dataset.copy()
    dataset.loc[0, "Age"] = None

    results = validate_dataset(dataset)

    assert get_status(results, "Missing values") == "FAIL"


def test_duplicate_rows_are_detected(valid_dataset):
    dataset = pd.concat(
        [valid_dataset, valid_dataset.iloc[[0]]],
        ignore_index=True,
    )

    results = validate_dataset(dataset)

    assert get_status(results, "Duplicate rows") == "FAIL"


def test_duplicate_customer_ids_are_detected(valid_dataset):
    dataset = valid_dataset.copy()
    dataset.loc[1, "CustomerId"] = dataset.loc[0, "CustomerId"]

    results = validate_dataset(dataset)

    assert get_status(results, "Customer ID uniqueness") == "FAIL"


def test_invalid_target_values_are_detected(valid_dataset):
    dataset = valid_dataset.copy()
    dataset.loc[0, "Exited"] = 2

    results = validate_dataset(dataset)

    assert get_status(results, "Target values") == "FAIL"


def test_invalid_binary_values_are_detected(valid_dataset):
    dataset = valid_dataset.copy()
    dataset.loc[0, "HasCrCard"] = 3

    results = validate_dataset(dataset)

    assert get_status(results, "HasCrCard binary validation") == "FAIL"


def test_invalid_numerical_range_is_detected(valid_dataset):
    dataset = valid_dataset.copy()
    dataset.loc[0, "Age"] = -5

    results = validate_dataset(dataset)

    assert get_status(results, "Age range validation") == "FAIL"


def test_validation_does_not_modify_original_dataset(valid_dataset):
    original = valid_dataset.copy(deep=True)

    validate_dataset(valid_dataset)

    pd.testing.assert_frame_equal(valid_dataset, original)

def test_invalid_geography_is_detected(valid_dataset):
    dataset = valid_dataset.copy()
    dataset.loc[0, "Geography"] = "Italy"

    results = validate_dataset(dataset)

    assert get_status(
        results,
        "Geography categorical validation",
    ) == "FAIL"


def test_invalid_gender_is_detected(valid_dataset):
    dataset = valid_dataset.copy()
    dataset.loc[0, "Gender"] = "Unknown"

    results = validate_dataset(dataset)

    assert get_status(
        results,
        "Gender categorical validation",
    ) == "FAIL"


def test_fractional_age_is_detected(valid_dataset):
    dataset = valid_dataset.copy()
    dataset["Age"] = dataset["Age"].astype(float)
    dataset.loc[0, "Age"] = 35.7

    results = validate_dataset(dataset)

    assert get_status(
        results,
        "Age integer validation",
    ) == "FAIL"