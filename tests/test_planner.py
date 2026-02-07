"""
Test suite for planner service.
Tests LLM spec generation, JSON parsing, and validation.
"""

import json
import pytest
from unittest.mock import Mock, patch

from services.planner.planner import VisualizationPlanner, build_intent_prompt
from services.planner.spec_schema import (
    VisualizationSpec,
    ChartType,
    AggregateSpec,
    FilterSpec,
)
from services.ingest.parser import DatasetSchema
from services.llm.openai_client import UsageStats


@pytest.fixture
def sample_schema():
    """Create sample dataset schema."""
    return DatasetSchema(
        columns=["date", "region", "revenue", "units"],
        dtypes={"date": "object", "region": "object", "revenue": "float64", "units": "int64"},
        row_count=100,
        sample_rows=[
            {"date": "2024-01-01", "region": "North", "revenue": 10000.0, "units": 50},
            {"date": "2024-01-02", "region": "South", "revenue": 15000.0, "units": 75},
        ],
        missing_values={"date": 0, "region": 0, "revenue": 2, "units": 1},
        memory_usage_mb=0.01,
    )


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock()
    return client


@pytest.fixture
def planner(mock_llm_client):
    """Create planner with mock LLM client."""
    return VisualizationPlanner(mock_llm_client)


@pytest.fixture
def valid_spec_json():
    """Valid visualization spec as JSON string."""
    return json.dumps(
        {
            "chart_type": "line",
            "x": "date",
            "y": "revenue",
            "aggregate": None,
            "filters": [],
            "transformations": [],
            "options": {"title": "Revenue over time", "height": 600, "width": 900},
            "explain": "Shows revenue trend over time",
        }
    )


def test_build_intent_prompt(sample_schema):
    """Test intent prompt construction."""
    prompt = build_intent_prompt("Show revenue by region", sample_schema)

    assert "revenue" in prompt
    assert "region" in prompt
    assert "Show revenue by region" in prompt
    assert "Sample rows" in prompt


def test_build_intent_prompt_with_history(sample_schema):
    """Test intent prompt includes chat history."""
    history = [
        {"role": "user", "content": "Show me the data"},
        {"role": "assistant", "content": "Here's the data preview"},
    ]

    prompt = build_intent_prompt("Now show revenue", sample_schema, history)

    assert "Previous conversation" in prompt or "context" in prompt.lower()


def test_extract_json_from_plain_response(planner, valid_spec_json):
    """Test JSON extraction from plain JSON response."""
    extracted = planner._extract_json_from_response(valid_spec_json)

    assert isinstance(extracted, dict)
    assert extracted["chart_type"] == "line"
    assert extracted["x"] == "date"


def test_extract_json_from_markdown_code_block(planner):
    """Test JSON extraction from markdown code blocks."""
    response = '''```json
{
  "chart_type": "bar",
  "x": "region",
  "y": "revenue",
  "explain": "Revenue by region"
}
```'''

    extracted = planner._extract_json_from_response(response)

    assert extracted["chart_type"] == "bar"
    assert extracted["x"] == "region"


def test_extract_json_from_mixed_text(planner):
    """Test JSON extraction when LLM adds extra text."""
    response = '''Sure! Here's the visualization spec:

{"chart_type": "scatter", "x": "units", "y": "revenue", "explain": "Units vs revenue"}

Let me know if you need changes!'''

    extracted = planner._extract_json_from_response(response)

    assert extracted["chart_type"] == "scatter"


def test_extract_json_fails_gracefully(planner):
    """Test JSON extraction raises error for invalid response."""
    from core.exceptions import SpecValidationError

    invalid_response = "This is not JSON at all, just plain text."

    with pytest.raises(SpecValidationError):
        planner._extract_json_from_response(invalid_response)


def test_plan_success(planner, mock_llm_client, sample_schema, valid_spec_json):
    """Test successful planning with valid LLM response."""
    # Mock LLM response
    mock_usage = UsageStats(
        prompt_tokens=100, completion_tokens=50, total_tokens=150, model="gpt-4", duration_seconds=1.5
    )
    mock_llm_client.complete.return_value = (valid_spec_json, mock_usage)

    # Plan visualization
    response, usage = planner.plan("Show revenue over time", sample_schema)

    assert response.spec is not None
    assert response.error is None
    assert response.spec.chart_type == ChartType.LINE
    assert response.spec.x == "date"
    assert response.spec.y == "revenue"
    assert usage.total_tokens == 150


def test_plan_with_error_response(planner, mock_llm_client, sample_schema):
    """Test planning handles LLM error responses."""
    error_json = json.dumps(
        {"error": "column_not_found", "message": "Column 'sales' not found", "candidates": ["revenue"]}
    )

    mock_usage = UsageStats(
        prompt_tokens=50, completion_tokens=20, total_tokens=70, model="gpt-4", duration_seconds=0.8
    )
    mock_llm_client.complete.return_value = (error_json, mock_usage)

    response, usage = planner.plan("Show sales", sample_schema)

    assert response.spec is None
    assert response.error is not None
    assert response.error.error == "column_not_found"
    assert "revenue" in response.error.candidates


def test_visualization_spec_validation():
    """Test VisualizationSpec validation."""
    # Valid spec
    spec = VisualizationSpec(
        chart_type=ChartType.BAR,
        x="category",
        y="value",
        explain="Test chart",
    )

    assert spec.chart_type == ChartType.BAR
    assert spec.x == "category"


def test_visualization_spec_rejects_dangerous_columns():
    """Test VisualizationSpec rejects column names with dangerous characters."""
    with pytest.raises(ValueError):
        VisualizationSpec(
            chart_type=ChartType.LINE,
            x="date; DROP TABLE users;",
            y="value",
            explain="SQL injection attempt",
        )


def test_aggregate_spec_validation():
    """Test AggregateSpec validation."""
    agg = AggregateSpec(func="sum", group_by=["region", "product"])

    assert agg.func.value == "sum"
    assert len(agg.group_by) == 2


def test_filter_spec_validation():
    """Test FilterSpec validation."""
    filter_spec = FilterSpec(column="revenue", op=">=", value=1000)

    assert filter_spec.column == "revenue"
    assert filter_spec.op.value == ">="
    assert filter_spec.value == 1000


def test_filter_spec_rejects_callable_value():
    """Test FilterSpec rejects callable values."""
    with pytest.raises(ValueError):
        FilterSpec(column="col", op="==", value=lambda x: x)


def test_refine_spec(planner, mock_llm_client, sample_schema):
    """Test spec refinement."""
    current_spec = VisualizationSpec(
        chart_type=ChartType.LINE, x="date", y="revenue", explain="Revenue over time"
    )

    refined_json = json.dumps(
        {
            "chart_type": "bar",
            "x": "region",
            "y": "revenue",
            "aggregate": {"func": "sum", "group_by": ["region"]},
            "filters": [],
            "transformations": [],
            "options": {},
            "explain": "Total revenue by region",
        }
    )

    mock_usage = UsageStats(
        prompt_tokens=80, completion_tokens=40, total_tokens=120, model="gpt-4", duration_seconds=1.0
    )
    mock_llm_client.complete.return_value = (refined_json, mock_usage)

    response, usage = planner.refine("Group by region instead", current_spec, sample_schema)

    assert response.spec is not None
    assert response.spec.chart_type == ChartType.BAR
    assert response.spec.aggregate is not None
    assert response.spec.aggregate.func.value == "sum"
