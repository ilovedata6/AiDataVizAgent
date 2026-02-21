"""
Safe CSV/XLSX parser with validation, type inference, and sanitization.
Never executes arbitrary code; performs controlled data ingestion only.
"""

import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from core.config import get_settings
from core.exceptions import FileParsingError, FileUploadError, SecurityError
from core.logging import get_logger

logger = get_logger(__name__)


class DatasetSchema(BaseModel):
    """Schema information for an uploaded dataset."""

    columns: list[str] = Field(description="List of column names")
    dtypes: dict[str, str] = Field(description="Mapping of column names to data types")
    row_count: int = Field(description="Number of rows in the dataset")
    sample_rows: list[dict[str, Any]] = Field(description="Sample rows for preview")
    missing_values: dict[str, int] = Field(description="Count of missing values per column")
    memory_usage_mb: float = Field(description="Memory usage in megabytes")


class DatasetParser:
    """
    Parse and validate uploaded CSV/XLSX files.
    Performs security checks, type inference, and schema extraction.
    """

    def __init__(self) -> None:
        """Initialize parser with configuration."""
        self.settings = get_settings()
        self.max_size_bytes = self.settings.max_upload_size_bytes
        self.allowed_extensions = self.settings.allowed_extensions_list

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal and injection attacks.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename

        Raises:
            SecurityError: If filename contains dangerous characters
        """
        if not self.settings.sanitize_filenames:
            return filename

        # Remove path components
        filename = Path(filename).name

        # Allow only alphanumeric, underscore, hyphen, and period
        sanitized = re.sub(r"[^\w\-.]", "_", filename)

        # Prevent hidden files
        if sanitized.startswith("."):
            sanitized = "_" + sanitized[1:]

        # Ensure extension is valid
        extension = Path(sanitized).suffix.lower().lstrip(".")
        if extension not in self.allowed_extensions:
            raise SecurityError(
                f"File extension '.{extension}' not allowed. "
                f"Allowed: {', '.join(self.allowed_extensions)}"
            )

        logger.info("Filename sanitized", original=filename, sanitized=sanitized)
        return sanitized

    def _validate_file_size(self, file_bytes: bytes, filename: str) -> None:
        """
        Validate file size is within limits.

        Args:
            file_bytes: Raw file bytes
            filename: Filename for logging

        Raises:
            FileUploadError: If file is too large
        """
        size_bytes = len(file_bytes)
        size_mb = size_bytes / (1024 * 1024)

        if size_bytes > self.max_size_bytes:
            raise FileUploadError(
                f"File '{filename}' is too large ({size_mb:.2f} MB). "
                f"Maximum allowed: {self.settings.max_upload_size_mb} MB",
                details={"size_mb": size_mb, "max_mb": self.settings.max_upload_size_mb},
            )

        logger.info("File size validated", filename=filename, size_mb=f"{size_mb:.2f}")

    def _infer_delimiter(self, content: str) -> str:
        """
        Infer CSV delimiter by sampling first few lines.

        Args:
            content: CSV content as string

        Returns:
            Detected delimiter (comma, tab, semicolon, or pipe)
        """
        # Sample first 5 lines
        lines = content.split("\n")[:5]
        sample = "\n".join(lines)

        # Count occurrences of common delimiters
        delimiters = [",", "\t", ";", "|"]
        counts = {d: sample.count(d) for d in delimiters}

        # Return delimiter with highest count
        delimiter = max(counts, key=counts.get)  # type: ignore
        logger.debug("Delimiter inferred", delimiter=repr(delimiter))
        return delimiter

    def parse_csv(self, file_bytes: bytes, filename: str) -> pd.DataFrame:
        """
        Parse CSV file with automatic delimiter detection and type inference.

        Args:
            file_bytes: Raw CSV bytes
            filename: Original filename

        Returns:
            Parsed DataFrame

        Raises:
            FileParsingError: If parsing fails
        """
        try:
            # Try UTF-8 first, fallback to latin1
            try:
                content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("UTF-8 decode failed, trying latin1", filename=filename)
                content = file_bytes.decode("latin1")

            # Infer delimiter
            delimiter = self._infer_delimiter(content)

            # Parse CSV
            df = pd.read_csv(
                BytesIO(content.encode("utf-8")),
                delimiter=delimiter,
                engine="python",
                encoding="utf-8",
                low_memory=False,
                # Prevent code execution via converters/eval
                converters=None,
            )

            logger.info(
                "CSV parsed successfully",
                filename=filename,
                rows=len(df),
                columns=len(df.columns),
            )
            return df

        except Exception as e:
            logger.error("CSV parsing failed", filename=filename, error=str(e))
            raise FileParsingError(
                f"Failed to parse CSV file '{filename}': {str(e)}", details={"filename": filename}
            ) from e

    def parse_xlsx(self, file_bytes: bytes, filename: str) -> pd.DataFrame:
        """
        Parse XLSX file (Excel format).

        Args:
            file_bytes: Raw XLSX bytes
            filename: Original filename

        Returns:
            Parsed DataFrame (first sheet only)

        Raises:
            FileParsingError: If parsing fails
        """
        try:
            # Read Excel file (first sheet)
            df = pd.read_excel(BytesIO(file_bytes), sheet_name=0, engine="openpyxl")

            logger.info(
                "XLSX parsed successfully",
                filename=filename,
                rows=len(df),
                columns=len(df.columns),
            )
            return df

        except Exception as e:
            logger.error("XLSX parsing failed", filename=filename, error=str(e))
            raise FileParsingError(
                f"Failed to parse XLSX file '{filename}': {str(e)}",
                details={"filename": filename},
            ) from e

    def parse_file(self, file_bytes: bytes, filename: str) -> pd.DataFrame:
        """
        Parse uploaded file based on extension.

        Args:
            file_bytes: Raw file bytes
            filename: Original filename

        Returns:
            Parsed DataFrame

        Raises:
            FileUploadError: If file is invalid
            FileParsingError: If parsing fails
        """
        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)

        # Validate file size
        self._validate_file_size(file_bytes, safe_filename)

        # Determine file type and parse
        extension = Path(safe_filename).suffix.lower().lstrip(".")

        if extension == "csv":
            return self.parse_csv(file_bytes, safe_filename)
        elif extension in ["xlsx", "xls"]:
            return self.parse_xlsx(file_bytes, safe_filename)
        else:
            raise FileUploadError(
                f"Unsupported file type: .{extension}. Allowed: {', '.join(self.allowed_extensions)}"
            )

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform automatic data cleaning before visualization.

        Cleaning steps:
        1. Strip leading/trailing whitespace from column names.
        2. Strip whitespace from string (object) columns.
        3. Drop rows that are entirely empty.
        4. Drop columns that are entirely empty.
        5. Attempt to coerce 'object' columns that look numeric to numeric types.
        6. Parse columns that look like dates.

        Args:
            df: Raw parsed DataFrame

        Returns:
            Cleaned DataFrame (copy — original is not mutated)
        """
        original_shape = df.shape
        df = df.copy()

        # 1. Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # 2. Strip whitespace from string columns
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

        # 3. Drop fully-empty rows
        df.dropna(how="all", inplace=True)

        # 4. Drop fully-empty columns
        df.dropna(axis=1, how="all", inplace=True)

        # 5. Coerce numeric-looking object columns
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                # Accept if ≥ 80 % of non-null values converted successfully
                non_null = df[col].notna().sum()
                if non_null > 0 and converted.notna().sum() / non_null >= 0.8:
                    df[col] = converted
            except Exception:
                pass

        # 6. Parse date-looking object columns
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().head(50)
            if sample.empty:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() / len(sample) >= 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

        logger.info(
            "Data cleaning complete",
            original_shape=original_shape,
            cleaned_shape=df.shape,
            rows_dropped=original_shape[0] - df.shape[0],
            cols_dropped=original_shape[1] - df.shape[1],
        )
        return df

    def extract_schema(self, df: pd.DataFrame, sample_size: int = 5) -> DatasetSchema:
        """
        Extract schema information from DataFrame.

        Args:
            df: Pandas DataFrame
            sample_size: Number of sample rows to include

        Returns:
            DatasetSchema with metadata
        """
        # Get column names and types
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Get sample rows
        sample_rows = df.head(sample_size).to_dict(orient="records")

        # Count missing values
        missing_values = df.isnull().sum().to_dict()

        # Calculate memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        schema = DatasetSchema(
            columns=columns,
            dtypes=dtypes,
            row_count=len(df),
            sample_rows=sample_rows,
            missing_values=missing_values,
            memory_usage_mb=round(memory_usage_mb, 2),
        )

        logger.info(
            "Schema extracted",
            columns=len(columns),
            rows=len(df),
            memory_mb=schema.memory_usage_mb,
        )

        return schema
