"""
Pydantic schemas for visualization specifications.
Defines strict JSON structure for LLM output - NO CODE EXECUTION.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChartType(str, Enum):
    """Supported chart types."""

    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    PIE = "pie"
    AREA = "area"


class FilterOperator(str, Enum):
    """Supported filter operators."""

    EQ = "=="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "in"
    NOT_IN = "not in"


class AggregateFunction(str, Enum):
    """Supported aggregation functions."""

    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"


class TransformOperation(str, Enum):
    """Safe transformation operations (no arbitrary code)."""

    LOG = "log"
    LOG10 = "log10"
    SQRT = "sqrt"
    ABS = "abs"
    DIFF = "diff"
    PCT_CHANGE = "pct_change"
    ROLLING_MEAN = "rolling_mean"


class FilterSpec(BaseModel):
    """Filter specification for data subsetting."""

    column: str = Field(description="Column name to filter")
    op: FilterOperator = Field(description="Filter operator")
    value: Any = Field(description="Value to compare against")

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: Any) -> Any:
        """Ensure value is a simple type (no code objects)."""
        if callable(v):
            raise ValueError("Filter value cannot be callable")
        return v


class TransformSpec(BaseModel):
    """Transformation specification for column operations."""

    op: TransformOperation = Field(description="Transformation operation")
    column: str = Field(description="Column to transform")
    output_column: str | None = Field(default=None, description="Output column name")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Operation parameters (e.g., window size)"
    )


class AggregateSpec(BaseModel):
    """Aggregation specification."""

    func: AggregateFunction = Field(description="Aggregation function")
    group_by: list[str] = Field(default_factory=list, description="Columns to group by")


class ChartOptions(BaseModel):
    """Additional chart customization options."""

    title: str | None = Field(default=None, description="Chart title")
    xlabel: str | None = Field(default=None, description="X-axis label")
    ylabel: str | None = Field(default=None, description="Y-axis label")
    color: str | None = Field(default=None, description="Column to use for color encoding")
    size: str | None = Field(default=None, description="Column to use for size encoding")
    height: int = Field(default=600, ge=200, le=2000, description="Chart height in pixels")
    width: int = Field(default=900, ge=300, le=3000, description="Chart width in pixels")
    show_legend: bool = Field(default=True, description="Whether to show legend")
    log_x: bool = Field(default=False, description="Use log scale for x-axis")
    log_y: bool = Field(default=False, description="Use log scale for y-axis")
    sort: str | None = Field(default=None, description="Sort order: ascending or descending")
    limit: int | None = Field(default=None, ge=1, le=10000, description="Limit number of data points shown")


class VisualizationSpec(BaseModel):
    """
    Complete visualization specification.
    This is the ONLY output format accepted from LLM.
    NO executable code is allowed.
    """

    chart_type: ChartType = Field(description="Type of chart to generate")
    x: str | None = Field(default=None, description="Column name for x-axis")
    y: str | None = Field(default=None, description="Column name for y-axis")

    aggregate: AggregateSpec | None = Field(
        default=None, description="Aggregation specification"
    )
    filters: list[FilterSpec] = Field(
        default_factory=list, description="Data filters to apply"
    )
    transformations: list[TransformSpec] = Field(
        default_factory=list, description="Column transformations"
    )
    options: ChartOptions = Field(
        default_factory=ChartOptions, description="Chart customization options"
    )

    explain: str = Field(description="One-line explanation of what the chart shows")

    @field_validator("x", "y")
    @classmethod
    def validate_column_names(cls, v: str | None) -> str | None:
        """Ensure column names don't contain code-injection patterns."""
        if v is None:
            return v

        # Only block actual code-injection patterns, NOT normal punctuation
        dangerous_patterns = ["eval(", "exec(", "import ", "__import__", "compile(",
                              "globals(", "locals(", "getattr(", "setattr("]
        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError(f"Column name cannot contain '{pattern}'")

        return v

    class Config:
        """Pydantic config."""

        use_enum_values = True


class SpecError(BaseModel):
    """Error response when LLM cannot generate a valid spec."""

    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    candidates: list[str] = Field(
        default_factory=list, description="Suggested column names or alternatives"
    )


class PlannerResponse(BaseModel):
    """Response from planner service."""

    spec: VisualizationSpec | None = Field(default=None, description="Generated specification")
    error: SpecError | None = Field(default=None, description="Error if spec generation failed")
    raw_llm_output: str = Field(description="Raw LLM response for debugging")
