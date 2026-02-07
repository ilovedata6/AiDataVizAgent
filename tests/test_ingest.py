"""
Test suite for data ingestion and parsing.
Tests CSV/XLSX parsing, validation, and schema extraction.
"""

import pytest
import pandas as pd
from io import BytesIO

from services.ingest.parser import DatasetParser, DatasetSchema
from core.exceptions import FileUploadError, FileParsingError, SecurityError


@pytest.fixture
def parser():
    """Create DatasetParser instance."""
    return DatasetParser()


@pytest.fixture
def sample_csv():
    """Create sample CSV data."""
    csv_data = """name,age,city,salary
John,30,NYC,50000
Jane,25,LA,60000
Bob,35,Chicago,55000
Alice,28,SF,65000
"""
    return csv_data.encode("utf-8")


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame."""
    return pd.DataFrame(
        {
            "name": ["John", "Jane", "Bob"],
            "age": [30, 25, 35],
            "city": ["NYC", "LA", "Chicago"],
            "salary": [50000, 60000, 55000],
        }
    )


def test_parse_csv_success(parser, sample_csv):
    """Test successful CSV parsing."""
    df = parser.parse_csv(sample_csv, "test.csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert list(df.columns) == ["name", "age", "city", "salary"]
    assert df["age"].dtype in [int, "int64"]


def test_parse_csv_with_different_delimiters(parser):
    """Test CSV parsing with different delimiters."""
    # Tab-delimited
    tsv_data = b"name\tage\tscoreJohn\t30\t95\nJane\t25\t88"
    df = parser.parse_csv(tsv_data, "test.tsv")
    assert len(df.columns) >= 3

    # Semicolon-delimited
    csv_data = b"name;age;score\nJohn;30;95\nJane;25;88"
    df = parser.parse_csv(csv_data, "test.csv")
    assert len(df.columns) >= 3


def test_sanitize_filename_safe(parser):
    """Test filename sanitization with safe input."""
    safe_name = parser._sanitize_filename("data_2024.csv")
    assert safe_name == "data_2024.csv"


def test_sanitize_filename_dangerous(parser):
    """Test filename sanitization with dangerous characters."""
    dangerous = "../../../etc/passwd"
    sanitized = parser._sanitize_filename(dangerous)
    assert ".." not in sanitized
    assert "/" not in sanitized


def test_sanitize_filename_invalid_extension(parser):
    """Test rejection of invalid file extensions."""
    with pytest.raises(SecurityError):
        parser._sanitize_filename("malicious.exe")


def test_file_size_validation_success(parser, sample_csv):
    """Test file size validation passes for small files."""
    # Should not raise exception
    parser._validate_file_size(sample_csv, "test.csv")


def test_file_size_validation_failure(parser):
    """Test file size validation fails for large files."""
    # Create oversized data (100 MB)
    large_data = b"x" * (100 * 1024 * 1024)

    with pytest.raises(FileUploadError) as exc_info:
        parser._validate_file_size(large_data, "large.csv")

    assert "too large" in str(exc_info.value).lower()


def test_extract_schema(parser, sample_dataframe):
    """Test schema extraction from DataFrame."""
    schema = parser.extract_schema(sample_dataframe, sample_size=2)

    assert isinstance(schema, DatasetSchema)
    assert schema.row_count == 3
    assert schema.column_count == 4
    assert "name" in schema.columns
    assert "age" in schema.columns
    assert len(schema.sample_rows) == 2
    assert isinstance(schema.memory_usage_mb, float)


def test_extract_schema_missing_values(parser):
    """Test schema extraction handles missing values correctly."""
    df = pd.DataFrame({"col1": [1, 2, None, 4], "col2": [None, None, 3, 4]})

    schema = parser.extract_schema(df)

    assert schema.missing_values["col1"] == 1
    assert schema.missing_values["col2"] == 2


def test_parse_xlsx_format(parser):
    """Test XLSX parsing (mock test, actual requires xlsx file)."""
    # This would require creating actual XLSX bytes
    # For now, test that the method exists
    assert hasattr(parser, "parse_xlsx")


def test_parse_file_routing(parser, sample_csv):
    """Test parse_file routes to correct parser based on extension."""
    # CSV
    df = parser.parse_file(sample_csv, "test.csv")
    assert isinstance(df, pd.DataFrame)

    # Unknown extension should raise error
    with pytest.raises(FileUploadError):
        parser.parse_file(sample_csv, "test.unknown")


def test_delimiter_inference(parser):
    """Test CSV delimiter inference."""
    # Comma-delimited
    comma_csv = "a,b,c\n1,2,3"
    assert parser._infer_delimiter(comma_csv) == ","

    # Tab-delimited
    tab_csv = "a\tb\tc\n1\t2\t3"
    assert parser._infer_delimiter(tab_csv) == "\t"


def test_parse_csv_encoding_fallback(parser):
    """Test CSV parsing falls back to latin1 when UTF-8 fails."""
    # Create latin1-encoded data
    latin1_data = "name,value\nCafé,100\n".encode("latin1")

    # Should parse successfully with fallback
    df = parser.parse_csv(latin1_data, "test.csv")
    assert len(df) == 1


def test_parse_csv_malformed(parser):
    """Test error handling for malformed CSV."""
    malformed = b"name,age\nJohn,30,extra_column\nJane"

    with pytest.raises(FileParsingError):
        parser.parse_csv(malformed, "bad.csv")
