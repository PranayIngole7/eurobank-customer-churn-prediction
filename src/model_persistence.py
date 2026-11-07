"""
Model persistence utilities for Eurobank customer churn prediction.

Responsibilities:
1. Generate model metadata.
2. Validate metadata compatibility.
3. Persist fitted models using joblib.
4. Load previously persisted models.
5. Read metadata independently of model loading.
"""

import json
import platform
import tempfile

from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from src.preprocessing import FEATURE_COLUMNS


# Project model artifact directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY = PROJECT_ROOT / "models"

DEFAULT_MODEL_PATH = MODEL_DIRECTORY / "churn_model.joblib"
DEFAULT_METADATA_PATH = MODEL_DIRECTORY / "model_metadata.json"

METADATA_SCHEMA_VERSION = 1


def create_model_metadata(model, training_results):
    """
    Generate metadata describing a fitted model.

    Parameters
    ----------
    model:
        Fitted scikit-learn pipeline.

    training_results:
        Results returned by train_and_evaluate_final_model().
    """

    if not hasattr(model, "classes_"):
        raise ValueError("Model must be fitted before metadata creation.")

    required_result_fields = {
        "model_name",
        "algorithm",
        "threshold",
        "training_records",
    }

    missing_fields = required_result_fields - set(training_results)

    if missing_fields:
        raise ValueError(
            f"Training results are missing fields: {sorted(missing_fields)}"
        )

    metadata = {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "model_name": training_results["model_name"],
        "algorithm": training_results["algorithm"],
        "threshold": float(training_results["threshold"]),
        "feature_columns": list(FEATURE_COLUMNS),
        "training_records": int(training_results["training_records"]),
        "random_state": 42,
        "target_column": "Exited",
        "positive_class": 1,
        "model_classes": [
            int(value) for value in model.classes_
        ],
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    validate_metadata(metadata)

    return metadata


def validate_metadata(metadata):
    """
    Validate metadata structure and compatibility.

    Raises
    ------
    TypeError:
        If metadata is not a dictionary.

    ValueError:
        If required metadata is missing or incompatible.
    """

    if not isinstance(metadata, dict):
        raise TypeError("Metadata must be a dictionary.")

    required_fields = {
        "metadata_schema_version",
        "model_name",
        "algorithm",
        "threshold",
        "feature_columns",
        "training_records",
        "random_state",
        "target_column",
        "positive_class",
        "model_classes",
        "python_version",
        "scikit_learn_version",
        "joblib_version",
        "created_at",
    }

    missing_fields = required_fields - set(metadata)

    if missing_fields:
        raise ValueError(
            f"Metadata is missing fields: {sorted(missing_fields)}"
        )

    if metadata["metadata_schema_version"] != METADATA_SCHEMA_VERSION:
        raise ValueError("Unsupported metadata schema version.")

    if not isinstance(metadata["model_name"], str) or not metadata["model_name"]:
        raise ValueError("model_name must be a non-empty string.")

    if not isinstance(metadata["algorithm"], str) or not metadata["algorithm"]:
        raise ValueError("algorithm must be a non-empty string.")

    threshold = metadata["threshold"]

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be numeric.")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    if metadata["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature columns do not match the current application."
        )

    if (
        isinstance(metadata["training_records"], bool)
        or not isinstance(metadata["training_records"], int)
        or metadata["training_records"] <= 0
    ):
        raise ValueError("training_records must be a positive integer.")

    if metadata["target_column"] != "Exited":
        raise ValueError("Unexpected target column.")

    if metadata["positive_class"] != 1:
        raise ValueError("Positive class must be 1.")

    if metadata["model_classes"] != [0, 1]:
        raise ValueError("Model classes must contain [0, 1].")

    for field in (
        "python_version",
        "scikit_learn_version",
        "joblib_version",
        "created_at",
    ):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"{field} must be a non-empty string.")

    return True


def save_model(
    model,
    metadata,
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
):
    """
    Persist a fitted model and its metadata.

    Both files are written using temporary files followed by
    atomic replacement.
    """

    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("A fitted probability-capable model is required.")

    validate_metadata(metadata)

    if list(model.classes_) != metadata["model_classes"]:
        raise ValueError("Model classes do not match metadata.")

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if model_path.resolve() == metadata_path.resolve():
        raise ValueError("Model and metadata paths must be different.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    model_temp_path = None
    metadata_temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            model_temp_path = Path(temporary_file.name)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=metadata_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            metadata_temp_path = Path(temporary_file.name)

            json.dump(
                metadata,
                temporary_file,
                indent=4,
            )

        joblib.dump(model, model_temp_path)

        # Replace only after both temporary artifacts are written.
        model_temp_path.replace(model_path)
        metadata_temp_path.replace(metadata_path)

    finally:
        for temporary_path in (
            model_temp_path,
            metadata_temp_path,
        ):
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    return model_path, metadata_path


def load_model(
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
):
    """
    Load a persisted model and validate its metadata.

    Returns
    -------
    tuple
        Loaded model and metadata dictionary.
    """

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    metadata = get_model_metadata(metadata_path)

    # joblib uses pickle-based serialization. Only load trusted artifacts.
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(
            "Unable to load model artifact; it may be corrupted or incompatible."
        ) from exc

    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError("Loaded artifact is not a fitted probability-capable model.")

    if list(model.classes_) != metadata["model_classes"]:
        raise ValueError("Loaded model classes do not match metadata.")

    return model, metadata


def get_model_metadata(metadata_path=DEFAULT_METADATA_PATH):
    """
    Read and validate metadata without loading the model artifact.
    """

    metadata_path = Path(metadata_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}"
        )

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError("Model metadata contains invalid JSON.") from error

    validate_metadata(metadata)

    return metadata