"""Batch customer prediction page."""

import csv
import io

import pandas as pd
import streamlit as st

from dashboard.components import render_header
from dashboard.services import (
    build_prediction_output,
    get_prediction_service,
    prepare_probability_distribution,
)


REQUIRED_COLUMNS = {
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
}

PREDICTION_COLUMNS = {
    "churn_probability",
    "risk_score",
    "risk_category",
    "churn_prediction",
}

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_BATCH_ROWS = 50_000

RISK_CATEGORIES = [
    "Low",
    "Moderate",
    "High",
    "Critical",
]


def create_sample_template():
    """Create a sample CSV containing the required raw features."""

    sample = pd.DataFrame([
        {
            "CreditScore": 619,
            "Geography": "France",
            "Gender": "Female",
            "Age": 42,
            "Tenure": 2,
            "Balance": 125000.00,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 101348.88,
        }
    ])

    return sample.to_csv(index=False).encode("utf-8")


def read_uploaded_csv(uploaded_file):
    """Read and validate the uploaded CSV structure."""

    raw_bytes = uploaded_file.getvalue()

    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(raw_bytes) > MAX_UPLOAD_SIZE:
        raise ValueError("The CSV file exceeds the 10 MB upload limit.")

    try:
        content = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "The CSV must use UTF-8 encoding."
        ) from exc

    try:
        reader = csv.reader(io.StringIO(content))
        headers = next(reader, None)

        if not headers:
            raise ValueError("The CSV does not contain a header row.")

        headers = [header.strip() for header in headers]

        if any(not header for header in headers):
            raise ValueError("The CSV contains an empty column name.")

        if len(headers) != len(set(headers)):
            raise ValueError("The CSV contains duplicate column names.")

        dataframe = pd.read_csv(io.StringIO(content))

    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV does not contain readable data.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Invalid CSV structure: {exc}") from exc

    dataframe.columns = headers

    if dataframe.empty:
        raise ValueError("The CSV contains no customer records.")

    if len(dataframe) > MAX_BATCH_ROWS:
        raise ValueError(
            f"The CSV contains more than {MAX_BATCH_ROWS:,} records."
        )

    return dataframe


def validate_batch_input(customers):
    """Validate the uploaded customer DataFrame."""

    if "Exited" in customers.columns:
        raise ValueError(
            "Do not include the target column 'Exited'. "
            "Batch prediction does not require actual churn outcomes."
        )

    conflicting_columns = PREDICTION_COLUMNS.intersection(
        customers.columns
    )

    if conflicting_columns:
        raise ValueError(
            "The input already contains prediction output columns: "
            f"{sorted(conflicting_columns)}"
        )

    missing_columns = REQUIRED_COLUMNS - set(customers.columns)

    if missing_columns:
        raise ValueError(
            "Missing required customer columns: "
            f"{sorted(missing_columns)}"
        )

    required_data = customers[list(REQUIRED_COLUMNS)]

    if required_data.isnull().any().any():
        raise ValueError(
            "Required customer fields contain missing values."
        )

    string_data = required_data.select_dtypes(
        include=["object", "string"]
    )

    if string_data.apply(
        lambda column: column.str.strip().eq("").any()
    ).any():
        raise ValueError(
            "Required customer fields contain empty values."
        )


def render_batch_summary(results):
    """Display batch-level prediction metrics."""

    total_customers = len(results)

    predicted_churners = int(
        results["churn_prediction"].sum()
    )

    predicted_churn_rate = predicted_churners / total_customers

    st.header("Batch Prediction Summary")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}",
        )

    with metric_col2:
        st.metric(
            "Predicted Churners",
            f"{predicted_churners:,}",
        )

    with metric_col3:
        st.metric(
            "Predicted Churn Rate",
            f"{predicted_churn_rate:.1%}",
        )

    st.subheader("Risk Category Distribution")

    risk_distribution = (
        results["risk_category"]
        .value_counts()
        .reindex(RISK_CATEGORIES, fill_value=0)
        .rename_axis("Risk Category")
        .to_frame("Customers")
    )

    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        st.bar_chart(risk_distribution)

    with table_col:
        st.dataframe(
            risk_distribution,
            use_container_width=True,
        )

    st.subheader("Predicted Churn Probability Distribution")
    st.caption(
        "Distribution of model-predicted probabilities for this uploaded "
        "batch. Probabilities are not observed churn outcomes or the "
        "separate risk categories above."
    )
    probability_distribution = prepare_probability_distribution(
        results["churn_probability"].to_numpy()
    )
    st.bar_chart(
        probability_distribution,
        x="Probability range",
        y="Customers",
    )


def render_batch_prediction():
    """Render the batch customer prediction interface."""

    render_header(
        "Batch Customer Prediction",
        "Upload customer records and generate churn predictions "
        "for multiple customers.",
    )

    # 1. Downloadable input template
    st.header("CSV Input Template")

    st.write(
        "Download the sample template to understand the required "
        "customer columns and their format."
    )

    st.download_button(
        label="Download Sample CSV Template",
        data=create_sample_template(),
        file_name="customer_prediction_template.csv",
        mime="text/csv",
    )

    st.caption(
        "The template contains the 10 required raw customer features. "
        "Do not include the target column Exited."
    )

    st.divider()

    # 2. File upload
    st.header("Upload Customer Data")

    uploaded_file = st.file_uploader(
        "Choose a customer CSV file",
        type=["csv"],
        help="Maximum file size: 10 MB. Maximum records: 50,000.",
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin batch prediction.")
        return

    # 3. Load and validate the uploaded data
    try:
        customers = read_uploaded_csv(uploaded_file)
        validate_batch_input(customers)

    except (ValueError, TypeError) as exc:
        st.error(f"CSV validation failed: {exc}")
        return

    except Exception as exc:
        st.error(f"Unable to read the uploaded CSV: {exc}")
        return

    st.success(
        f"CSV loaded successfully: {len(customers):,} customer records."
    )

    if {"CustomerId", "Surname"}.intersection(customers.columns):
        st.warning(
            "CustomerId and Surname are ignored by the model and will be "
            "excluded from previews and downloaded prediction results."
        )

    extra_columns = (
        set(customers.columns) - REQUIRED_COLUMNS
    )

    if extra_columns:
        st.caption(
            "Additional columns will be ignored and excluded from "
            "prediction output: "
            f"{sorted(extra_columns)}"
        )

    with st.expander("Preview Uploaded Customer Data"):
        st.dataframe(
            customers[list(sorted(REQUIRED_COLUMNS))].head(10),
            use_container_width=True,
        )

    st.divider()

    # 4. Generate predictions
    if st.button(
        "Generate Batch Predictions",
        type="primary",
        use_container_width=True,
    ):

        try:
            prediction_service = get_prediction_service()

            with st.spinner(
                f"Generating predictions for {len(customers):,} customers..."
            ):
                predictions = prediction_service.predict_many(customers)

            if len(predictions) != len(customers):
                raise ValueError(
                    "Prediction result count does not match "
                    "the uploaded customer count."
                )

            results = build_prediction_output(customers, predictions)

        except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
            st.error(f"Batch prediction failed: {exc}")
            return

        except Exception as exc:
            st.error(f"Unexpected batch prediction error: {exc}")
            return

        # 5. Display results
        st.success(
            f"Successfully generated predictions for "
            f"{len(results):,} customers."
        )

        render_batch_summary(results)

        st.divider()

        st.header("Prediction Results")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

        # 6. Download results
        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="customer_churn_predictions.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

        st.info(
            "These are model predictions, not verified customer outcomes. "
            "The predicted churn rate is not the actual churn rate."
        )