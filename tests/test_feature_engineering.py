import pandas as pd
import pytest

from src.feature_engineering import engineer_features


@pytest.fixture
def sample_dataset():
    return pd.DataFrame({
        "Age": [25, 35, 45, 55, 65],
        "Balance": [0, 25000, 50000, 0, 100000],
    })


@pytest.fixture
def sample_dataset_with_optional_features():
    return pd.DataFrame({
        "Age": [25, 35, 45, 55],
        "Balance": [5000.0, 20000.0, 15000.0, 50000.0],
        "EstimatedSalary": [20000.0, 50000.0, 25000.0, 100000.0],
        "Tenure": [1, 2, 3, 5],
        "NumOfProducts": [1, 2, 2, 3],
        "IsActiveMember": [0, 1, 1, 0],
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


def test_balance_to_salary_ratio_computes_expected_value(
    sample_dataset_with_optional_features,
):
    result = engineer_features(sample_dataset_with_optional_features)

    expected = [
        0.25,
        0.4,
        0.6,
        0.5,
    ]

    assert result["BalanceToSalaryRatio"].tolist() == pytest.approx(expected)
    assert result["BalanceToSalary"].tolist() == pytest.approx(expected)


def test_balance_to_salary_ratio_zero_salary_returns_zero():
    dataset = pd.DataFrame({
        "Age": [30, 40],
        "Balance": [1000.0, 5000.0],
        "EstimatedSalary": [0.0, 40000.0],
    })

    result = engineer_features(dataset)

    assert result["BalanceToSalaryRatio"].tolist() == [0.0, 0.125]


def test_balance_to_salary_ratio_negative_salary_raises_error():
    dataset = pd.DataFrame({
        "Age": [30],
        "Balance": [1000.0],
        "EstimatedSalary": [-1.0],
    })

    with pytest.raises(ValueError, match="EstimatedSalary cannot contain negative values"):
        engineer_features(dataset)


def test_balance_to_salary_ratio_non_numeric_salary_raises_error():
    dataset = pd.DataFrame({
        "Age": [30],
        "Balance": [1000.0],
        "EstimatedSalary": ["invalid"],
    })

    with pytest.raises(ValueError, match="missing or non-numeric"):
        engineer_features(dataset)


def test_product_density_indicator_computes_expected_value(
    sample_dataset_with_optional_features,
):
    result = engineer_features(sample_dataset_with_optional_features)

    expected = [1.0, 1.0, 0.6666666667, 0.6]

    assert result["ProductDensityIndicator"].tolist() == pytest.approx(expected)
    assert result["ProductDensity"].tolist() == pytest.approx(expected)


def test_product_density_indicator_zero_tenure_returns_zero():
    dataset = pd.DataFrame({
        "Age": [30, 40],
        "Balance": [1000.0, 2000.0],
        "Tenure": [0, 3],
        "NumOfProducts": [2, 2],
    })

    result = engineer_features(dataset)

    assert result["ProductDensityIndicator"].tolist() == pytest.approx([0.0, 0.6666666667])


def test_tenure_and_product_count_domain_boundaries():
    dataset = pd.DataFrame({
        "Age": [30, 40],
        "Balance": [1000.0, 2000.0],
        "Tenure": [0, 10],
        "NumOfProducts": [1, 4],
    })

    result = engineer_features(dataset)

    assert result["ProductDensityIndicator"].tolist() == pytest.approx(
        [0.0, 0.4]
    )
    assert result["EngagementProductInteraction"].tolist() == [0.0, 0.0]


def test_product_density_indicator_negative_tenure_raises_error():
    dataset = pd.DataFrame({
        "Age": [30],
        "Balance": [1000.0],
        "Tenure": [-1],
        "NumOfProducts": [2],
    })

    with pytest.raises(ValueError, match="Tenure must be between 0 and 10"):
        engineer_features(dataset)


@pytest.mark.parametrize(
    "tenure",
    [1.5, 11, float("inf"), None],
)
def test_tenure_rejects_fractional_out_of_range_and_invalid_values(tenure):
    dataset = pd.DataFrame({
        "Age": [30],
        "Balance": [1000.0],
        "Tenure": [tenure],
    })

    with pytest.raises(ValueError, match="Tenure"):
        engineer_features(dataset)


@pytest.mark.parametrize(
    "products",
    [0, -1, 1.5, 5, float("inf"), None],
)
def test_product_count_rejects_invalid_values(products):
    dataset = pd.DataFrame({
        "Age": [30],
        "Balance": [1000.0],
        "NumOfProducts": [products],
    })

    with pytest.raises(ValueError, match="NumOfProducts"):
        engineer_features(dataset)


def test_engagement_product_interaction_computes_expected_value(
    sample_dataset_with_optional_features,
):
    result = engineer_features(sample_dataset_with_optional_features)

    expected = [0, 2, 2, 0]

    assert result["EngagementProductInteraction"].tolist() == expected
    assert result["EngagementProduct"].tolist() == expected


def test_engagement_product_interaction_invalid_active_member_raises_error():
    dataset = pd.DataFrame({
        "Age": [30],
        "Balance": [1000.0],
        "IsActiveMember": [2],
        "NumOfProducts": [1],
    })

    with pytest.raises(ValueError, match="IsActiveMember must contain only 0 and 1"):
        engineer_features(dataset)


def test_age_tenure_interaction_computes_expected_value(
    sample_dataset_with_optional_features,
):
    result = engineer_features(sample_dataset_with_optional_features)

    expected = [25, 70, 135, 275]

    assert result["AgeTenureInteraction"].tolist() == expected
    assert result["AgeTenure"].tolist() == expected


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
        "Balance": [1000],
    })

    with pytest.raises(ValueError, match="integer"):
        engineer_features(dataset)
