"""Tests for batch customer prediction and CSV validation."""

import io

import pandas as pd
import pytest

from dashboard import batch_prediction as batch


class FakeUploadedFile:
    """Minimal uploaded-file object for testing CSV readers."""

    def __init__(self, content):
        self.content = content

    def getvalue(self):
        return self.content


@pytest.fixture
def valid_customers():
    """Return a valid customer DataFrame."""

    return pd.DataFrame([
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


def test_create_sample_template_contains_required_columns():
    content = batch.create_sample_template()

    dataframe = pd.read_csv(io.BytesIO(content))

    assert list(dataframe.columns) == [
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
    ]

    assert len(dataframe) == 1


def test_read_uploaded_csv_accepts_valid_csv(valid_customers):
    content = valid_customers.to_csv(index=False).encode("utf-8")

    result = batch.read_uploaded_csv(FakeUploadedFile(content))

    assert len(result) == 1
    assert list(result.columns) == list(valid_customers.columns)


def test_read_uploaded_csv_accepts_utf8_bom(valid_customers):
    content = valid_customers.to_csv(index=False).encode("utf-8-sig")

    result = batch.read_uploaded_csv(FakeUploadedFile(content))

    assert len(result) == 1
    assert "CreditScore" in result.columns


def test_read_uploaded_csv_rejects_empty_file():
    with pytest.raises(ValueError, match="uploaded file is empty"):
        batch.read_uploaded_csv(FakeUploadedFile(b""))


def test_read_uploaded_csv_rejects_invalid_encoding():
    with pytest.raises(ValueError, match="UTF-8"):
        batch.read_uploaded_csv(FakeUploadedFile(b"\xff\xfe\xfa"))


def test_read_uploaded_csv_rejects_missing_header():
    with pytest.raises(ValueError, match="header row"):
        batch.read_uploaded_csv(FakeUploadedFile(b"\xef\xbb\xbf"))


def test_read_uploaded_csv_rejects_duplicate_headers():
    content = (
        b"CreditScore,CreditScore\n"
        b"619,620\n"
    )

    with pytest.raises(ValueError, match="duplicate column names"):
        batch.read_uploaded_csv(FakeUploadedFile(content))


def test_read_uploaded_csv_rejects_blank_header():
    content = (
        b"CreditScore,,Age\n"
        b"619,France,42\n"
    )

    with pytest.raises(ValueError, match="empty column name"):
        batch.read_uploaded_csv(FakeUploadedFile(content))


def test_read_uploaded_csv_rejects_file_over_size_limit(monkeypatch):
    monkeypatch.setattr(batch, "MAX_UPLOAD_SIZE", 10)

    content = b"CreditScore,Age\n619,42\n"

    with pytest.raises(ValueError, match="10 MB upload limit"):
        batch.read_uploaded_csv(FakeUploadedFile(content))


def test_read_uploaded_csv_rejects_excessive_rows(monkeypatch):
    monkeypatch.setattr(batch, "MAX_BATCH_ROWS", 1)

    content = (
        b"CreditScore,Age\n"
        b"619,42\n"
        b"620,43\n"
    )

    with pytest.raises(ValueError, match="more than 1 records"):
        batch.read_uploaded_csv(FakeUploadedFile(content))


def test_read_uploaded_csv_rejects_header_only_file():
    content = b"CreditScore,Age\n"

    with pytest.raises(ValueError, match="no customer records"):
        batch.read_uploaded_csv(FakeUploadedFile(content))


def test_validate_batch_input_accepts_valid_customers(valid_customers):
    # Successful validation should not raise an exception.
    batch.validate_batch_input(valid_customers)


def test_validate_batch_input_accepts_extra_columns(valid_customers):
    valid_customers["CustomerId"] = 12345
    valid_customers["Surname"] = "Example"

    batch.validate_batch_input(valid_customers)


def test_validate_batch_input_rejects_target_column(valid_customers):
    valid_customers["Exited"] = 1

    with pytest.raises(ValueError, match="target column 'Exited'"):
        batch.validate_batch_input(valid_customers)


def test_validate_batch_input_rejects_prediction_columns(valid_customers):
    valid_customers["churn_probability"] = 0.75

    with pytest.raises(ValueError, match="prediction output columns"):
        batch.validate_batch_input(valid_customers)


def test_validate_batch_input_rejects_missing_columns(valid_customers):
    valid_customers.drop(columns=["Age"], inplace=True)

    with pytest.raises(ValueError, match="Missing required customer columns"):
        batch.validate_batch_input(valid_customers)


def test_validate_batch_input_rejects_missing_values(valid_customers):
    valid_customers.loc[0, "Age"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        batch.validate_batch_input(valid_customers)


def test_validate_batch_input_rejects_empty_strings(valid_customers):
    valid_customers.loc[0, "Geography"] = ""

    with pytest.raises(ValueError, match="empty values"):
        batch.validate_batch_input(valid_customers)


def test_validate_batch_input_rejects_whitespace_only_strings(valid_customers):
    valid_customers.loc[0, "Geography"] = "   "

    with pytest.raises(ValueError, match="empty values"):
        batch.validate_batch_input(valid_customers)