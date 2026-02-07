"""
Plotly renderer: convert VisualizationSpec to interactive Plotly figures.
Uses ONLY safe Pandas operations - no eval, exec, or arbitrary code execution.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Figure

from core.exceptions import RenderError
from core.logging import get_logger
from services.planner.spec_schema import (
    AggregateSpec,
    FilterSpec,
    TransformSpec,
    VisualizationSpec,
)

logger = get_logger(__name__)


class PlotlyRenderer:
    """
    Render VisualizationSpec to Plotly figures using safe operations only.
    All data transformations use controlled Pandas methods.
    """

    def __init__(self) -> None:
        """Initialize Plotly renderer."""
        logger.info("Plotly renderer initialized")

    def _apply_filters(self, df: pd.DataFrame, filters: list[FilterSpec]) -> pd.DataFrame:
        """
        Apply filters to DataFrame using safe query operations.

        Args:
            df: Input DataFrame
            filters: List of FilterSpec

        Returns:
            Filtered DataFrame
        """
        result = df.copy()

        for f in filters:
            if f.column not in result.columns:
                logger.warning(f"Filter column '{f.column}' not found, skipping")
                continue

            try:
                if f.op.value == "==":
                    result = result[result[f.column] == f.value]
                elif f.op.value == "!=":
                    result = result[result[f.column] != f.value]
                elif f.op.value == ">":
                    result = result[result[f.column] > f.value]
                elif f.op.value == ">=":
                    result = result[result[f.column] >= f.value]
                elif f.op.value == "<":
                    result = result[result[f.column] < f.value]
                elif f.op.value == "<=":
                    result = result[result[f.column] <= f.value]
                elif f.op.value == "in":
                    result = result[result[f.column].isin(f.value)]
                elif f.op.value == "not in":
                    result = result[~result[f.column].isin(f.value)]

                logger.debug(f"Filter applied: {f.column} {f.op.value} {f.value}")

            except Exception as e:
                logger.error(f"Filter failed: {e}", column=f.column, op=f.op.value)
                raise RenderError(f"Failed to apply filter on '{f.column}': {str(e)}")

        return result

    def _apply_transformations(
        self, df: pd.DataFrame, transforms: list[TransformSpec]
    ) -> pd.DataFrame:
        """
        Apply safe column transformations.

        Args:
            df: Input DataFrame
            transforms: List of TransformSpec

        Returns:
            Transformed DataFrame
        """
        result = df.copy()

        for t in transforms:
            if t.column not in result.columns:
                logger.warning(f"Transform column '{t.column}' not found, skipping")
                continue

            output_col = t.output_column or f"{t.column}_{t.op.value}"

            try:
                if t.op.value == "log":
                    result[output_col] = np.log(result[t.column])
                elif t.op.value == "log10":
                    result[output_col] = np.log10(result[t.column])
                elif t.op.value == "sqrt":
                    result[output_col] = np.sqrt(result[t.column])
                elif t.op.value == "abs":
                    result[output_col] = np.abs(result[t.column])
                elif t.op.value == "diff":
                    result[output_col] = result[t.column].diff()
                elif t.op.value == "pct_change":
                    result[output_col] = result[t.column].pct_change()
                elif t.op.value == "rolling_mean":
                    window = t.params.get("window", 3)
                    result[output_col] = result[t.column].rolling(window=window).mean()

                logger.debug(f"Transformation applied: {t.op.value} on {t.column}")

            except Exception as e:
                logger.error(f"Transformation failed: {e}", column=t.column, op=t.op.value)
                raise RenderError(f"Failed to transform '{t.column}': {str(e)}")

        return result

    def _apply_aggregation(self, df: pd.DataFrame, agg: AggregateSpec, y_col: str) -> pd.DataFrame:
        """
        Apply aggregation using safe groupby operations.

        Args:
            df: Input DataFrame
            agg: AggregateSpec
            y_col: Column to aggregate

        Returns:
            Aggregated DataFrame
        """
        if not agg.group_by:
            # No grouping, just aggregate entire column
            if agg.func.value == "count":
                return pd.DataFrame({y_col: [len(df)]})
            else:
                agg_func = getattr(df[y_col], agg.func.value)
                return pd.DataFrame({y_col: [agg_func()]})

        # Group by specified columns
        try:
            grouped = df.groupby(agg.group_by, as_index=False)

            if agg.func.value == "count":
                result = grouped.size().rename(columns={"size": y_col})
            else:
                result = grouped[y_col].agg(agg.func.value).reset_index()

            logger.debug(f"Aggregation applied: {agg.func.value} grouped by {agg.group_by}")
            return result

        except Exception as e:
            logger.error(f"Aggregation failed: {e}", func=agg.func.value, group_by=agg.group_by)
            raise RenderError(f"Failed to aggregate data: {str(e)}")

    def render(self, spec: VisualizationSpec, df: pd.DataFrame) -> Figure:
        """
        Render VisualizationSpec to Plotly Figure.

        Args:
            spec: Visualization specification
            df: Source DataFrame

        Returns:
            Plotly Figure

        Raises:
            RenderError: If rendering fails
        """
        logger.info(f"Rendering {spec.chart_type} chart", x=spec.x, y=spec.y)

        try:
            # Apply filters
            if spec.filters:
                df = self._apply_filters(df, spec.filters)
                logger.debug(f"After filters: {len(df)} rows")

            # Apply transformations
            if spec.transformations:
                df = self._apply_transformations(df, spec.transformations)

            # Apply aggregation
            if spec.aggregate:
                df = self._apply_aggregation(df, spec.aggregate, spec.y or "count")

            # Validate columns exist
            if spec.x and spec.x not in df.columns:
                raise RenderError(f"Column '{spec.x}' not found in data")
            if spec.y and spec.y not in df.columns:
                raise RenderError(f"Column '{spec.y}' not found in data")

            # Create chart based on type
            fig = self._create_chart(spec, df)

            logger.info("Chart rendered successfully", chart_type=spec.chart_type)
            return fig

        except RenderError:
            raise
        except Exception as e:
            logger.error(f"Rendering failed: {e}", chart_type=spec.chart_type)
            raise RenderError(f"Failed to render {spec.chart_type} chart: {str(e)}")

    def _create_chart(self, spec: VisualizationSpec, df: pd.DataFrame) -> Figure:
        """
        Create Plotly chart based on chart type.

        Args:
            spec: Visualization specification
            df: Prepared DataFrame

        Returns:
            Plotly Figure
        """
        opts = spec.options

        # Common parameters
        common_params = {
            "title": opts.title or spec.explain,
            "height": opts.height,
            "width": opts.width,
            "color": opts.color if opts.color and opts.color in df.columns else None,
        }

        # Chart-specific rendering
        if spec.chart_type.value == "line":
            fig = px.line(df, x=spec.x, y=spec.y, **common_params)

        elif spec.chart_type.value == "bar":
            fig = px.bar(df, x=spec.x, y=spec.y, **common_params)

        elif spec.chart_type.value == "scatter":
            size = opts.size if opts.size and opts.size in df.columns else None
            fig = px.scatter(df, x=spec.x, y=spec.y, size=size, **common_params)

        elif spec.chart_type.value == "histogram":
            fig = px.histogram(df, x=spec.x, **common_params)

        elif spec.chart_type.value == "box":
            fig = px.box(df, x=spec.x, y=spec.y, **common_params)

        elif spec.chart_type.value == "heatmap":
            # Pivot data for heatmap
            if spec.x and spec.y:
                pivot = df.pivot_table(values=spec.y, index=spec.x, aggfunc="mean")
                fig = px.imshow(pivot, **{k: v for k, v in common_params.items() if k != "color"})
            else:
                raise RenderError("Heatmap requires both x and y columns")

        elif spec.chart_type.value == "pie":
            fig = px.pie(df, names=spec.x, values=spec.y, **common_params)

        elif spec.chart_type.value == "area":
            fig = px.area(df, x=spec.x, y=spec.y, **common_params)

        else:
            raise RenderError(f"Unsupported chart type: {spec.chart_type}")

        # Apply axis labels and scales
        fig.update_xaxes(title=opts.xlabel or spec.x, type="log" if opts.log_x else None)
        fig.update_yaxes(title=opts.ylabel or spec.y, type="log" if opts.log_y else None)

        # Legend
        fig.update_layout(showlegend=opts.show_legend)

        return fig
