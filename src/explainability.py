"""
Model explainability utilities for customer churn prediction.

Responsibilities:
1. Extract Logistic Regression coefficients and odds ratios.
2. Calculate permutation feature importance.
3. Explain individual predictions using log-odds contributions.
4. Provide optional model-agnostic explainability wrappers for SHAP and
   partial dependence analyses.
"""

import numpy as np
import pandas as pd

from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.preprocessing import validate_feature_data


def extract_model_coefficients(model: Pipeline) -> pd.DataFrame:
    """
    Extract Logistic Regression coefficients and odds ratios.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted pipeline containing preprocessor and classifier steps.

    Returns
    -------
    pandas.DataFrame
        Feature coefficients, odds ratios, and direction of association.

    Raises
    ------
    TypeError
        If the supplied model is not a Pipeline.

    ValueError
        If the pipeline is not fitted or has an unsupported classifier.
    """

    if not isinstance(model, Pipeline):
        raise TypeError("Model must be a scikit-learn Pipeline.")

    if "preprocessor" not in model.named_steps:
        raise ValueError("Pipeline must contain a preprocessor step.")

    if "classifier" not in model.named_steps:
        raise ValueError("Pipeline must contain a classifier step.")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    if not isinstance(classifier, LogisticRegression):
        raise ValueError("Classifier must be LogisticRegression.")

    if not hasattr(preprocessor, "get_feature_names_out"):
        raise ValueError("Preprocessor does not support feature names.")

    if not hasattr(classifier, "coef_"):
        raise ValueError("Classifier must be fitted before extraction.")

    if not hasattr(classifier, "classes_"):
        raise ValueError("Classifier must be fitted before extraction.")

    if not np.array_equal(classifier.classes_, np.array([0, 1])):
        raise ValueError(
            "Classifier classes must be ordered as [0, 1]."
        )

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_

    if coefficients.shape != (1, len(feature_names)):
        raise ValueError(
            "Unexpected Logistic Regression coefficient shape."
        )

    coefficients = coefficients[0]

    if not np.isfinite(coefficients).all():
        raise ValueError("Model coefficients contain invalid values.")

    odds_ratios = np.exp(coefficients)

    if not np.isfinite(odds_ratios).all():
        raise ValueError("Odds ratios contain invalid values.")

    directions = np.where(
        coefficients > 0,
        "Higher churn odds",
        np.where(
            coefficients < 0,
            "Lower churn odds",
            "No directional association",
        ),
    )

    results = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "odds_ratio": odds_ratios,
        "absolute_coefficient": np.abs(coefficients),
        "direction": directions,
    })

    return (
        results
        .sort_values(
            by="absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_permutation_importance(
    model: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate permutation feature importance using validation ROC-AUC.

    Each original input feature is shuffled independently. Its importance
    is measured by the resulting decrease in ROC-AUC.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted Logistic Regression pipeline.

    X_validation : pandas.DataFrame
        Internal validation features.

    y_validation : pandas.Series
        Internal validation target.

    n_repeats : int, default=10
        Number of times each feature is shuffled.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Feature importance mean and standard deviation, sorted by
        descending mean importance.

    Raises
    ------
    TypeError
        If input types are invalid.

    ValueError
        If validation data or parameters are invalid.
    """

    if not isinstance(model, Pipeline):
        raise TypeError("Model must be a scikit-learn Pipeline.")

    if not isinstance(X_validation, pd.DataFrame):
        raise TypeError("X_validation must be a pandas DataFrame.")

    if not isinstance(y_validation, pd.Series):
        raise TypeError("y_validation must be a pandas Series.")

    if X_validation.empty:
        raise ValueError("Validation features cannot be empty.")

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "Validation features and target must have equal lengths."
        )

    if not X_validation.index.equals(y_validation.index):
        raise ValueError(
            "Validation features and target indexes must match."
        )

    if y_validation.isnull().any():
        raise ValueError("Validation target contains missing values.")

    if not y_validation.isin([0, 1]).all():
        raise ValueError("Validation target must contain only 0 and 1.")

    if y_validation.nunique() != 2:
        raise ValueError(
            "Validation target must contain both classes: 0 and 1."
        )

    if (
        isinstance(n_repeats, bool)
        or not isinstance(n_repeats, (int, np.integer))
        or n_repeats < 1
    ):
        raise ValueError("n_repeats must be a positive integer.")

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, (int, np.integer))
        or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer.")

    check_is_fitted(model)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support probability predictions.")

    if not hasattr(model, "classes_"):
        raise ValueError("Model must be fitted before importance analysis.")

    if not np.array_equal(model.classes_, np.array([0, 1])):
        raise ValueError(
            "Model classes must be ordered as [0, 1]."
        )

    validate_feature_data(X_validation)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Fitted model must contain original feature names."
        )

    if not X_validation.columns.equals(
        pd.Index(model.feature_names_in_)
    ):
        raise ValueError(
            "Validation columns must match the model training columns "
            "and their order."
        )

    importance = permutation_importance(
        estimator=model,
        X=X_validation,
        y=y_validation,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=int(random_state),
    )

    results = pd.DataFrame({
        "feature": X_validation.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    })

    return (
        results
        .sort_values(
            by="importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def explain_customer_prediction(
    model: Pipeline,
    customer_features: pd.DataFrame,
) -> dict:
    """
    Explain one customer's Logistic Regression prediction.

    Each transformed feature's contribution is calculated as:
        coefficient * transformed feature value

    The sum of all contributions and the intercept must equal
    the model's predicted log odds.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted preprocessing and Logistic Regression pipeline.

    customer_features : pandas.DataFrame
        Exactly one customer record using the original model input
        features and their training column order.

    Returns
    -------
    dict
        Base log odds, predicted log odds, predicted probability,
        and a DataFrame of feature-level contributions.

    Raises
    ------
    TypeError
        If input types are invalid.

    ValueError
        If the model or customer input is invalid.
    """

    if not isinstance(model, Pipeline):
        raise TypeError("Model must be a scikit-learn Pipeline.")

    if not isinstance(customer_features, pd.DataFrame):
        raise TypeError(
            "customer_features must be a pandas DataFrame."
        )

    if customer_features.empty:
        raise ValueError("Customer features cannot be empty.")

    if len(customer_features) != 1:
        raise ValueError(
            "customer_features must contain exactly one customer."
        )

    validate_feature_data(customer_features)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Fitted model must contain original feature names."
        )

    if not customer_features.columns.equals(
        pd.Index(model.feature_names_in_)
    ):
        raise ValueError(
            "Customer columns must match the model training columns "
            "and their order."
        )

    if "preprocessor" not in model.named_steps:
        raise ValueError("Pipeline must contain a preprocessor step.")

    if "classifier" not in model.named_steps:
        raise ValueError("Pipeline must contain a classifier step.")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    if not isinstance(classifier, LogisticRegression):
        raise ValueError("Classifier must be LogisticRegression.")

    check_is_fitted(model)

    if not np.array_equal(classifier.classes_, np.array([0, 1])):
        raise ValueError(
            "Classifier classes must be ordered as [0, 1]."
        )

    if not hasattr(classifier, "coef_") or not hasattr(
        classifier, "intercept_"
    ):
        raise ValueError(
            "Logistic Regression classifier must be fitted."
        )

    # Transform the original customer features.
    transformed = preprocessor.transform(customer_features)

    # Support both dense and sparse transformed feature matrices.
    if hasattr(transformed, "toarray"):
        transformed_values = transformed.toarray()[0]
    else:
        transformed_values = np.asarray(transformed)[0]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])

    if len(transformed_values) != len(feature_names):
        raise ValueError(
            "Transformed feature count does not match feature names."
        )

    if coefficients.shape != transformed_values.shape:
        raise ValueError(
            "Coefficient count does not match transformed features."
        )

    if not (
        np.isfinite(transformed_values).all()
        and np.isfinite(coefficients).all()
        and np.isfinite(intercept)
    ):
        raise ValueError(
            "Model explanation contains invalid numerical values."
        )

    # Calculate each feature's contribution to log odds.
    contributions = coefficients * transformed_values

    # Calculate the model's actual decision score.
    predicted_log_odds = float(
        classifier.decision_function(
            transformed.reshape(1, -1)
            if not hasattr(transformed, "toarray")
            else transformed
        )[0]
    )

    calculated_log_odds = intercept + contributions.sum()

    if not np.isclose(
        calculated_log_odds,
        predicted_log_odds,
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError(
            "Feature contributions do not reconstruct model log odds."
        )

    # Obtain the model's churn probability.
    predicted_probability = float(
        model.predict_proba(customer_features)[0, 1]
    )

    directions = np.where(
        contributions > 0,
        "Increases churn log odds",
        np.where(
            contributions < 0,
            "Decreases churn log odds",
            "No contribution",
        ),
    )

    contribution_results = pd.DataFrame({
        "feature": feature_names,
        "transformed_value": transformed_values,
        "coefficient": coefficients,
        "contribution": contributions,
        "direction": directions,
    })

    contribution_results = (
        contribution_results
        .assign(
            absolute_contribution=lambda df: df["contribution"].abs()
        )
        .sort_values(
            "absolute_contribution",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return {
        "base_log_odds": intercept,
        "predicted_log_odds": predicted_log_odds,
        "predicted_probability": predicted_probability,
        "contributions": contribution_results,
    }


def calculate_shap_values(
    model: Pipeline,
    X: pd.DataFrame,
    *,
    max_samples: int = 100,
    background_size: int = 50,
    random_state: int = 42,
    sample_indices: pd.Index | None = None,
    max_evals: int | None = None,
):
    """Calculate positive-class probability SHAP values on sampled rows."""
    if not isinstance(model, Pipeline):
        raise TypeError("Model must be a scikit-learn Pipeline.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X cannot be empty.")

    validate_feature_data(X)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Fitted model must contain original feature names.")

    feature_names = list(model.feature_names_in_)
    if list(X.columns) != feature_names:
        raise ValueError(
            "X columns must match the model training columns and order."
        )

    for parameter, value in (
        ("max_samples", max_samples),
        ("background_size", background_size),
        ("random_state", random_state),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError(f"{parameter} must be a non-negative integer.")
    if max_samples < 1 or background_size < 1:
        raise ValueError("max_samples and background_size must be positive.")
    minimum_evaluations = 2 * len(feature_names) + 1
    if max_evals is None:
        max_evals = minimum_evaluations
    if (
        isinstance(max_evals, bool)
        or not isinstance(max_evals, (int, np.integer))
        or max_evals < minimum_evaluations
    ):
        raise ValueError(
            "max_evals must be an integer of at least "
            f"{minimum_evaluations} for {len(feature_names)} features."
        )
    if sample_indices is not None:
        if not isinstance(sample_indices, pd.Index):
            raise TypeError("sample_indices must be a pandas Index.")
        if sample_indices.empty or sample_indices.has_duplicates:
            raise ValueError(
                "sample_indices must be non-empty and contain no duplicates."
            )
        if len(sample_indices) > max_samples:
            raise ValueError("sample_indices cannot exceed max_samples.")
        if not sample_indices.isin(X.index).all():
            raise ValueError("sample_indices must refer to rows in X.")

    check_is_fitted(model)

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for SHAP explanations. Install project "
            "dependencies from requirements.txt."
        ) from exc

    classes = np.asarray(model.classes_)
    positive_class_indices = np.flatnonzero(classes == 1)
    if len(positive_class_indices) != 1:
        raise ValueError("Model does not contain positive class 1.")
    positive_class_index = int(positive_class_indices[0])

    categorical_levels = {}
    shap_input = X.copy()
    for column in feature_names:
        if not pd.api.types.is_numeric_dtype(shap_input[column]):
            levels = list(pd.unique(shap_input[column]))
            categorical_levels[column] = {
                index: value for index, value in enumerate(levels)
            }
            encoded_values = {
                value: index for index, value in enumerate(levels)
            }
            shap_input[column] = (
                shap_input[column].map(encoded_values).astype(float)
            )

    background = shap_input.sample(
        n=min(background_size, len(X)),
        random_state=int(random_state),
    )
    if sample_indices is None:
        sample = shap_input.sample(
            n=min(max_samples, len(X)),
            random_state=int(random_state) + 1,
        )
    else:
        sample = shap_input.loc[sample_indices]

    def churn_probability(values):
        frame = pd.DataFrame(values, columns=feature_names)
        for column, levels in categorical_levels.items():
            codes = np.rint(frame[column]).astype(int)
            if not codes.isin(levels).all():
                raise ValueError(
                    f"SHAP generated an unknown category code for {column}."
                )
            frame[column] = codes.map(levels)
        return model.predict_proba(frame)[:, positive_class_index]

    explainer = shap.Explainer(
        churn_probability,
        background.to_numpy(),
        algorithm="permutation",
        feature_names=feature_names,
    )
    explanation = explainer(
        sample.to_numpy(),
        max_evals=int(max_evals),
    )
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        if values.shape[-1] <= positive_class_index:
            raise ValueError("SHAP output does not contain positive class 1.")
        values = values[..., positive_class_index]
    if values.shape != (len(sample), len(feature_names)):
        raise ValueError(
            "SHAP output shape does not align with sampled rows and features."
        )
    if not np.isfinite(values).all():
        raise ValueError("SHAP returned non-finite attribution values.")

    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    predictions = churn_probability(sample.to_numpy())
    if len(base_values) == 1 and len(sample) > 1:
        base_values = np.repeat(base_values, len(sample))
    if base_values.shape != (len(sample),):
        raise ValueError("SHAP base values do not align with explained rows.")
    if not np.isfinite(base_values).all():
        raise ValueError("SHAP returned non-finite base values.")
    additivity_residual = (
        base_values + values.sum(axis=1) - predictions
    )
    if not np.allclose(
        base_values + values.sum(axis=1),
        predictions,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            "SHAP base values and feature attributions do not reconstruct "
            "the model's positive-class probabilities."
        )

    global_importance = (
        pd.DataFrame({
            "feature": feature_names,
            "mean_absolute_shap": np.abs(values).mean(axis=0),
        })
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "feature_names": feature_names,
        "values": values,
        "data": X.loc[sample.index].reset_index(drop=True),
        "base_values": base_values,
        "predictions": predictions,
        "additivity_residual": additivity_residual,
        "max_evals": int(max_evals),
        "global_importance": global_importance,
        "explainer": explainer,
    }


def generate_partial_dependence(
    model: Pipeline,
    X: pd.DataFrame,
    feature: str,
    *,
    grid_size: int = 20,
    output_path: str | None = None,
):
    """Generate a partial dependence summary for a single feature."""
    if not isinstance(model, Pipeline):
        raise TypeError("Model must be a scikit-learn Pipeline.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' is not present in the dataset.")

    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be at least 2.")

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Fitted model must contain original feature names.")
    if list(X.columns) != list(model.feature_names_in_):
        raise ValueError(
            "X columns must match the model training columns and order."
        )

    check_is_fitted(model)
    pdp_data = X.copy()
    if pd.api.types.is_integer_dtype(pdp_data[feature]):
        pdp_data[feature] = pdp_data[feature].astype(float)

    display = PartialDependenceDisplay.from_estimator(
        estimator=model,
        X=pdp_data,
        features=[feature],
        grid_resolution=grid_size,
        response_method="predict_proba",
        target=1,
    )
    display.axes_[0, 0].set_title(f"Partial Dependence: {feature}")
    display.axes_[0, 0].set_xlabel(feature)
    display.axes_[0, 0].set_ylabel("Average predicted churn probability")
    display.figure_.tight_layout()

    if output_path is not None:
        from pathlib import Path

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        display.figure_.savefig(destination, dpi=300, bbox_inches="tight")

    return {
        "feature": feature,
        "figure": display.figure_,
        "axes": display.axes_,
        "result": display.pd_results[0],
    }
