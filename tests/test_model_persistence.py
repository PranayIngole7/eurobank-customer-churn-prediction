import json

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_persistence import (
    create_model_metadata,
    get_model_metadata,
    load_model,
    save_model,
    validate_metadata,
)

from src.preprocessing import FEATURE_COLUMNS


@pytest.fixture
def fitted_model():
    X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5, 6],
            "feature_2": [2, 1, 4, 3, 6, 5],
        }
    )

    y = [0, 0, 0, 1, 1, 1]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
        ]
    )

    model.fit(X, y)

    return model


@pytest.fixture
def training_results():
    return {
        "model_name": "logistic_regression_balanced",
        "algorithm": "Balanced Logistic Regression",
        "threshold": 0.65,
        "training_records": 8000,
    }


@pytest.fixture
def metadata(fitted_model, training_results):
    return create_model_metadata(
        fitted_model,
        training_results,
    )


def test_create_model_metadata_contains_required_fields(
    fitted_model,
    training_results,
):
    metadata = create_model_metadata(
        fitted_model,
        training_results,
    )

    assert metadata["model_name"] == "logistic_regression_balanced"
    assert metadata["threshold"] == 0.65
    assert metadata["training_records"] == 8000
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert metadata["model_classes"] == [0, 1]


def test_create_model_metadata_rejects_unfitted_model(
    training_results,
):
    model = LogisticRegression()

    with pytest.raises(ValueError, match="fitted"):
        create_model_metadata(model, training_results)


def test_create_model_metadata_rejects_missing_training_results(
    fitted_model,
):
    with pytest.raises(ValueError, match="missing fields"):
        create_model_metadata(fitted_model, {})


def test_validate_metadata_accepts_valid_metadata(metadata):
    assert validate_metadata(metadata) is True


def test_validate_metadata_rejects_missing_field(metadata):
    invalid_metadata = metadata.copy()
    del invalid_metadata["threshold"]

    with pytest.raises(ValueError, match="missing fields"):
        validate_metadata(invalid_metadata)


def test_validate_metadata_rejects_invalid_threshold(metadata):
    invalid_metadata = metadata.copy()
    invalid_metadata["threshold"] = 1.5

    with pytest.raises(ValueError, match="threshold"):
        validate_metadata(invalid_metadata)


def test_validate_metadata_rejects_boolean_threshold(metadata):
    invalid_metadata = metadata.copy()
    invalid_metadata["threshold"] = True

    with pytest.raises(TypeError, match="numeric"):
        validate_metadata(invalid_metadata)


def test_validate_metadata_rejects_incompatible_features(metadata):
    invalid_metadata = metadata.copy()
    invalid_metadata["feature_columns"] = ["wrong_feature"]

    with pytest.raises(ValueError, match="feature columns"):
        validate_metadata(invalid_metadata)


def test_validate_metadata_rejects_invalid_training_count(metadata):
    invalid_metadata = metadata.copy()
    invalid_metadata["training_records"] = 0

    with pytest.raises(ValueError, match="training_records"):
        validate_metadata(invalid_metadata)


def test_save_and_load_model_round_trip(
    fitted_model,
    metadata,
    tmp_path,
):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    save_model(
        fitted_model,
        metadata,
        model_path,
        metadata_path,
    )

    loaded_model, loaded_metadata = load_model(
        model_path,
        metadata_path,
    )

    X = pd.DataFrame(
        {
            "feature_1": [2, 5],
            "feature_2": [1, 6],
        }
    )

    original_probabilities = fitted_model.predict_proba(X)
    loaded_probabilities = loaded_model.predict_proba(X)

    np.testing.assert_allclose(
        original_probabilities,
        loaded_probabilities,
    )

    assert loaded_metadata == metadata


def test_save_model_rejects_unfitted_model(metadata, tmp_path):
    model = LogisticRegression()

    with pytest.raises(ValueError, match="fitted"):
        save_model(
            model,
            metadata,
            tmp_path / "model.joblib",
            tmp_path / "metadata.json",
        )


def test_load_model_rejects_missing_model(metadata, tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(FileNotFoundError, match="Model artifact"):
        load_model(
            tmp_path / "missing.joblib",
            metadata_path,
        )

def test_load_model_rejects_corrupted_artifact(
    metadata,
    tmp_path,
):
    model_path = tmp_path / "corrupted_model.joblib"
    metadata_path = tmp_path / "metadata.json"

    metadata_path.write_text(json.dumps(metadata))
    model_path.write_bytes(b"this is not a valid joblib artifact")

    with pytest.raises(
        ValueError,
        match="Unable to load model artifact",
    ):
        load_model(model_path, metadata_path)

def test_get_model_metadata_rejects_invalid_json(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{invalid json")

    with pytest.raises(ValueError, match="invalid JSON"):
        get_model_metadata(metadata_path)

@pytest.mark.parametrize(
    "invalid_structure",
    [
        [],
        "invalid metadata",
        123,
        None,
    ],
)
def test_get_model_metadata_rejects_invalid_json_structure(
    invalid_structure,
    tmp_path,
):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(invalid_structure))

    with pytest.raises(TypeError, match="Metadata must be a dictionary"):
        get_model_metadata(metadata_path)

def test_get_model_metadata_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_model_metadata(tmp_path / "missing.json")


def test_load_model_rejects_incompatible_model_classes(
    fitted_model,
    metadata,
    tmp_path,
):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    save_model(
        fitted_model,
        metadata,
        model_path,
        metadata_path,
    )

    incompatible_model = LogisticRegression()
    incompatible_model.fit(
        [[1], [2], [3], [4]],
        [1, 1, 2, 2],
    )

    joblib.dump(incompatible_model, model_path)

    with pytest.raises(ValueError, match="classes"):
        load_model(model_path, metadata_path)