"""
Data profiling service for column statistics and automatic chart recommendations.
Generates summary statistics, detects column types, and suggests visualizations.
"""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger(__name__)


class ColumnProfile(BaseModel):
    """Profile information for a single column."""

    name: str = Field(description="Column name")
    dtype: str = Field(description="Data type")
    unique_count: int = Field(description="Number of unique values")
    missing_count: int = Field(description="Number of missing values")
    missing_percentage: float = Field(description="Percentage of missing values")
    sample_values: list[Any] = Field(description="Sample of unique values")

    # Numeric statistics
    mean: float | None = Field(default=None, description="Mean (numeric only)")
    median: float | None = Field(default=None, description="Median (numeric only)")
    std: float | None = Field(default=None, description="Standard deviation (numeric only)")
    min: float | None = Field(default=None, description="Minimum value (numeric only)")
    max: float | None = Field(default=None, description="Maximum value (numeric only)")

    # Categorical statistics
    most_common: list[tuple[Any, int]] | None = Field(
        default=None, description="Most common values (categorical only)"
    )

    # Column classification
    is_numeric: bool = Field(description="Whether column is numeric")
    is_categorical: bool = Field(description="Whether column is categorical")
    is_datetime: bool = Field(description="Whether column is datetime")
    is_boolean: bool = Field(description="Whether column is boolean")


class DatasetProfile(BaseModel):
    """Complete profile of a dataset."""

    row_count: int = Field(description="Total number of rows")
    column_count: int = Field(description="Total number of columns")
    columns: list[ColumnProfile] = Field(description="Profile for each column")
    numeric_columns: list[str] = Field(description="List of numeric column names")
    categorical_columns: list[str] = Field(description="List of categorical column names")
    datetime_columns: list[str] = Field(description="List of datetime column names")
    recommended_charts: list[dict[str, Any]] = Field(
        description="Recommended chart types based on data"
    )


class DataProfiler:
    """
    Profile datasets and generate column-level statistics.
    Provides automatic chart recommendations based on data characteristics.
    """

    def __init__(self, max_categorical_threshold: int = 20) -> None:
        """
        Initialize profiler.

        Args:
            max_categorical_threshold: Max unique values to treat as categorical
        """
        self.max_categorical_threshold = max_categorical_threshold

    def _classify_column(self, series: pd.Series) -> dict[str, bool]:
        """
        Classify column type (numeric, categorical, datetime, boolean).

        Args:
            series: Pandas Series to classify

        Returns:
            Dictionary with boolean flags for each type
        """
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        is_boolean = pd.api.types.is_bool_dtype(series) or set(series.dropna().unique()) <= {
            0,
            1,
            True,
            False,
        }

        # Consider categorical if:
        # - Not numeric/datetime
        # - Has limited unique values
        # - Or is object/string type
        unique_count = series.nunique()
        is_categorical = (
            not is_numeric
            and not is_datetime
            and (
                unique_count <= self.max_categorical_threshold
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_object_dtype(series)
            )
        )

        return {
            "is_numeric": is_numeric,
            "is_categorical": is_categorical,
            "is_datetime": is_datetime,
            "is_boolean": is_boolean,
        }

    def _profile_column(self, series: pd.Series) -> ColumnProfile:
        """
        Profile a single column.

        Args:
            series: Pandas Series to profile

        Returns:
            ColumnProfile with statistics
        """
        classification = self._classify_column(series)

        # Basic statistics
        unique_count = series.nunique()
        missing_count = int(series.isnull().sum())
        missing_percentage = round((missing_count / len(series)) * 100, 2)

        # Sample values (up to 5 unique)
        sample_values = series.dropna().unique()[:5].tolist()

        profile_data: dict[str, Any] = {
            "name": series.name,
            "dtype": str(series.dtype),
            "unique_count": unique_count,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "sample_values": sample_values,
            **classification,
        }

        # Numeric statistics
        if classification["is_numeric"]:
            try:
                profile_data["mean"] = float(series.mean())
                profile_data["median"] = float(series.median())
                profile_data["std"] = float(series.std())
                profile_data["min"] = float(series.min())
                profile_data["max"] = float(series.max())
            except Exception as e:
                logger.warning(f"Failed to compute numeric stats for {series.name}: {e}")

        # Categorical statistics
        if classification["is_categorical"]:
            try:
                value_counts = series.value_counts().head(5)
                profile_data["most_common"] = list(zip(value_counts.index.tolist(), value_counts.values.tolist()))
            except Exception as e:
                logger.warning(f"Failed to compute categorical stats for {series.name}: {e}")

        return ColumnProfile(**profile_data)

    def _recommend_charts(
        self, numeric_cols: list[str], categorical_cols: list[str], datetime_cols: list[str]
    ) -> list[dict[str, Any]]:
        """
        Recommend chart types based on column types.

        Args:
            numeric_cols: List of numeric columns
            categorical_cols: List of categorical columns
            datetime_cols: List of datetime columns

        Returns:
            List of recommended chart specifications
        """
        recommendations = []

        # Histogram for numeric columns
        if numeric_cols:
            recommendations.append(
                {
                    "chart_type": "histogram",
                    "suggested_x": numeric_cols[0],
                    "title": f"Distribution of {numeric_cols[0]}",
                    "description": "Shows the distribution of values",
                }
            )

        # Bar chart for categorical columns
        if categorical_cols:
            recommendations.append(
                {
                    "chart_type": "bar",
                    "suggested_x": categorical_cols[0],
                    "suggested_y": "count",
                    "title": f"Count by {categorical_cols[0]}",
                    "description": "Shows frequency of each category",
                }
            )

        # Time series if datetime available
        if datetime_cols and numeric_cols:
            recommendations.append(
                {
                    "chart_type": "line",
                    "suggested_x": datetime_cols[0],
                    "suggested_y": numeric_cols[0],
                    "title": f"{numeric_cols[0]} over time",
                    "description": "Shows trend over time",
                }
            )

        # Scatter plot if multiple numeric columns
        if len(numeric_cols) >= 2:
            recommendations.append(
                {
                    "chart_type": "scatter",
                    "suggested_x": numeric_cols[0],
                    "suggested_y": numeric_cols[1],
                    "title": f"{numeric_cols[1]} vs {numeric_cols[0]}",
                    "description": "Shows relationship between two variables",
                }
            )

        # Box plot for numeric with categorical grouping
        if numeric_cols and categorical_cols:
            recommendations.append(
                {
                    "chart_type": "box",
                    "suggested_x": categorical_cols[0],
                    "suggested_y": numeric_cols[0],
                    "title": f"{numeric_cols[0]} by {categorical_cols[0]}",
                    "description": "Shows distribution across categories",
                }
            )

        return recommendations

    def profile_dataset(self, df: pd.DataFrame) -> DatasetProfile:
        """
        Profile entire dataset and generate recommendations.

        Args:
            df: Pandas DataFrame to profile

        Returns:
            DatasetProfile with complete analysis
        """
        logger.info("Profiling dataset", rows=len(df), columns=len(df.columns))

        # Profile each column
        column_profiles = [self._profile_column(df[col]) for col in df.columns]

        # Categorize columns
        numeric_columns = [cp.name for cp in column_profiles if cp.is_numeric]
        categorical_columns = [cp.name for cp in column_profiles if cp.is_categorical]
        datetime_columns = [cp.name for cp in column_profiles if cp.is_datetime]

        # Generate recommendations
        recommended_charts = self._recommend_charts(
            numeric_columns, categorical_columns, datetime_columns
        )

        profile = DatasetProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns=column_profiles,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            recommended_charts=recommended_charts,
        )

        logger.info(
            "Dataset profiled",
            numeric_cols=len(numeric_columns),
            categorical_cols=len(categorical_columns),
            datetime_cols=len(datetime_columns),
            recommendations=len(recommended_charts),
        )

        return profile
