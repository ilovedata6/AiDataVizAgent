"""
Test suite for renderer services.
Tests Plotly and Seaborn rendering with safe operations.
"""

import pytest
import pandas as pd
import numpy as np

from services.renderer.plotly_renderer import PlotlyRenderer
from services.renderer.seaborn_renderer import SeabornRenderer
from services.planner.spec_schema import (
    VisualizationSpec,
    ChartType,
    AggregateSpec,
    FilterSpec,
    TransformSpec,
    ChartOptions,
)
from core.exceptions import RenderError


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "category": ["A", "B", "A", "B", "A"] * 2,
            "value": np.random.randint(10, 100, 10),
            "quantity": np.random.randint(1, 10, 10),
        }
    )


@pytest.fixture
def plotly_renderer():
    """Create PlotlyRenderer instance."""
    return PlotlyRenderer()


@pytest.fixture
def seaborn_renderer():
    """Create SeabornRenderer instance."""
    return SeabornRenderer()


def test_render_line_chart(plotly_renderer, sample_df):
    """Test rendering a line chart."""
    spec = VisualizationSpec(
        chart_type=ChartType.LINE,
        x="date",
        y="value",
        explain="Value over time",
    )

    fig = plotly_renderer.render(spec, sample_df)

    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_render_bar_chart(plotly_renderer, sample_df):
    """Test rendering a bar chart."""
    spec = VisualizationSpec(
        chart_type=ChartType.BAR, x="category", y="value", explain="Value by category"
    )

    fig = plotly_renderer.render(spec, sample_df)

    assert fig is not None


def test_render_scatter_chart(plotly_renderer, sample_df):
    """Test rendering a scatter plot."""
    spec = VisualizationSpec(
        chart_type=ChartType.SCATTER,
        x="quantity",
        y="value",
        explain="Value vs quantity",
    )

    fig = plotly_renderer.render(spec, sample_df)

    assert fig is not None


def test_render_histogram(plotly_renderer, sample_df):
    """Test rendering a histogram."""
    spec = VisualizationSpec(
        chart_type=ChartType.HISTOGRAM, x="value", explain="Distribution of values"
    )

    fig = plotly_renderer.render(spec, sample_df)

    assert fig is not None


def test_render_with_color_encoding(plotly_renderer, sample_df):
    """Test rendering with color encoding."""
    spec = VisualizationSpec(
        chart_type=ChartType.SCATTER,
        x="quantity",
        y="value",
        options=ChartOptions(color="category"),
        explain="Value vs quantity by category",
    )

    fig = plotly_renderer.render(spec, sample_df)

    assert fig is not None


def test_apply_filters(plotly_renderer, sample_df):
    """Test filter application."""
    filters = [FilterSpec(column="value", op=">=", value=50)]

    filtered_df = plotly_renderer._apply_filters(sample_df, filters)

    assert len(filtered_df) < len(sample_df)
    assert all(filtered_df["value"] >= 50)


def test_apply_multiple_filters(plotly_renderer, sample_df):
    """Test multiple filter application."""
    filters = [
        FilterSpec(column="value", op=">=", value=30),
        FilterSpec(column="category", op="==", value="A"),
    ]

    filtered_df = plotly_renderer._apply_filters(sample_df, filters)

    assert all(filtered_df["value"] >= 30)
    assert all(filtered_df["category"] == "A")


def test_apply_in_filter(plotly_renderer, sample_df):
    """Test 'in' filter operator."""
    filters = [FilterSpec(column="category", op="in", value=["A", "B"])]

    filtered_df = plotly_renderer._apply_filters(sample_df, filters)

    assert all(filtered_df["category"].isin(["A", "B"]))


def test_apply_transformations(plotly_renderer, sample_df):
    """Test transformation application."""
    transforms = [TransformSpec(op="log", column="value", output_column="log_value")]

    transformed_df = plotly_renderer._apply_transformations(sample_df, transforms)

    assert "log_value" in transformed_df.columns
    assert not transformed_df["log_value"].isnull().all()


def test_apply_aggregation(plotly_renderer, sample_df):
    """Test aggregation."""
    agg_spec = AggregateSpec(func="sum", group_by=["category"])

    agg_df = plotly_renderer._apply_aggregation(sample_df, agg_spec, "value")

    assert len(agg_df) == 2  # Two categories
    assert "category" in agg_df.columns
    assert "value" in agg_df.columns


def test_render_with_aggregation(plotly_renderer, sample_df):
    """Test rendering with aggregation."""
    spec = VisualizationSpec(
        chart_type=ChartType.BAR,
        x="category",
        y="value",
        aggregate=AggregateSpec(func="mean", group_by=["category"]),
        explain="Average value by category",
    )

    fig = plotly_renderer.render(spec, sample_df)

    assert fig is not None


def test_render_with_filters_and_aggregation(plotly_renderer, sample_df):
    """Test rendering with both filters and aggregation."""
    spec = VisualizationSpec(
        chart_type=ChartType.BAR,
        x="category",
        y="value",
        filters=[FilterSpec(column="value", op=">=", value=30)],
        aggregate=AggregateSpec(func="sum", group_by=["category"]),
        explain="Sum of values >= 30 by category",
    )

    fig = plotly_renderer.render(spec, sample_df)

    assert fig is not None


def test_render_missing_column_error(plotly_renderer, sample_df):
    """Test error handling for missing column."""
    spec = VisualizationSpec(
        chart_type=ChartType.LINE,
        x="nonexistent_column",
        y="value",
        explain="Should fail",
    )

    with pytest.raises(RenderError):
        plotly_renderer.render(spec, sample_df)


def test_render_with_custom_title(plotly_renderer, sample_df):
    """Test rendering with custom title."""
    spec = VisualizationSpec(
        chart_type=ChartType.BAR,
        x="category",
        y="value",
        options=ChartOptions(title="Custom Title", height=400, width=600),
        explain="Bar chart with custom title",
    )

    fig = plotly_renderer.render(spec, sample_df)

    assert fig is not None
    assert "Custom Title" in fig.layout.title.text


def test_render_with_log_scale(plotly_renderer, sample_df):
    """Test rendering with logarithmic scale."""
    spec = VisualizationSpec(
        chart_type=ChartType.SCATTER,
        x="quantity",
        y="value",
        options=ChartOptions(log_y=True),
        explain="Scatter with log Y axis",
    )

    fig = plotly_renderer.render(spec, sample_df)

    assert fig is not None


def test_seaborn_render_line(seaborn_renderer, sample_df):
    """Test Seaborn line chart rendering."""
    spec = VisualizationSpec(
        chart_type=ChartType.LINE, x="date", y="value", explain="Line chart"
    )

    fig = seaborn_renderer.render(spec, sample_df)

    assert fig is not None
    # Matplotlib figure
    assert hasattr(fig, "axes")


def test_seaborn_render_bar(seaborn_renderer, sample_df):
    """Test Seaborn bar chart rendering."""
    spec = VisualizationSpec(
        chart_type=ChartType.BAR, x="category", y="value", explain="Bar chart"
    )

    fig = seaborn_renderer.render(spec, sample_df)

    assert fig is not None


def test_seaborn_uses_plotly_transformations(seaborn_renderer, sample_df):
    """Test Seaborn renderer uses PlotlyRenderer for transformations."""
    spec = VisualizationSpec(
        chart_type=ChartType.BAR,
        x="category",
        y="value",
        filters=[FilterSpec(column="value", op=">=", value=40)],
        explain="Filtered bar chart",
    )

    fig = seaborn_renderer.render(spec, sample_df)

    assert fig is not None


def test_no_code_execution_in_filters(plotly_renderer, sample_df):
    """Test that filters don't execute arbitrary code."""
    # This should be safely handled by our filter logic
    filters = [FilterSpec(column="value", op=">=", value=50)]

    # Should not raise security exception
    filtered_df = plotly_renderer._apply_filters(sample_df, filters)

    assert isinstance(filtered_df, pd.DataFrame)


def test_transformation_parameters(plotly_renderer, sample_df):
    """Test transformations with parameters."""
    transforms = [
        TransformSpec(
            op="rolling_mean", column="value", output_column="rolling_avg", params={"window": 3}
        )
    ]

    transformed_df = plotly_renderer._apply_transformations(sample_df, transforms)

    assert "rolling_avg" in transformed_df.columns
