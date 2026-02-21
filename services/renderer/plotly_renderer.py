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
                op = f.op.value if hasattr(f.op, 'value') else str(f.op)
                if op == "==":
                    result = result[result[f.column] == f.value]
                elif op == "!=":
                    result = result[result[f.column] != f.value]
                elif op == ">":
                    result = result[result[f.column] > f.value]
                elif op == ">=":
                    result = result[result[f.column] >= f.value]
                elif op == "<":
                    result = result[result[f.column] < f.value]
                elif op == "<=":
                    result = result[result[f.column] <= f.value]
                elif op == "in":
                    result = result[result[f.column].isin(f.value)]
                elif op == "not in":
                    result = result[~result[f.column].isin(f.value)]

                logger.debug(f"Filter applied: {f.column} {op} {f.value}")

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

            output_col = t.output_column or f"{t.column}_{t.op if isinstance(t.op, str) else t.op.value}"

            try:
                top = t.op.value if hasattr(t.op, 'value') else str(t.op)
                if top == "log":
                    result[output_col] = np.log(result[t.column])
                elif top == "log10":
                    result[output_col] = np.log10(result[t.column])
                elif top == "sqrt":
                    result[output_col] = np.sqrt(result[t.column])
                elif top == "abs":
                    result[output_col] = np.abs(result[t.column])
                elif top == "diff":
                    result[output_col] = result[t.column].diff()
                elif top == "pct_change":
                    result[output_col] = result[t.column].pct_change()
                elif top == "rolling_mean":
                    window = t.params.get("window", 3)
                    result[output_col] = result[t.column].rolling(window=window).mean()

                logger.debug(f"Transformation applied: {top} on {t.column}")

            except Exception as e:
                logger.error(f"Transformation failed: {e}", column=t.column, op=top)
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
            func = agg.func.value if hasattr(agg.func, 'value') else str(agg.func)
            if func == "count":
                return pd.DataFrame({y_col: [len(df)]})
            else:
                agg_fn = getattr(df[y_col], func)
                return pd.DataFrame({y_col: [agg_fn()]})

        # Group by specified columns
        try:
            func = agg.func.value if hasattr(agg.func, 'value') else str(agg.func)
            grouped = df.groupby(agg.group_by, as_index=False)

            if func == "count":
                result = grouped.size().rename(columns={"size": y_col})
            else:
                result = grouped[y_col].agg(func).reset_index()

            logger.debug(f"Aggregation applied: {func} grouped by {agg.group_by}")
            return result

        except Exception as e:
            logger.error(f"Aggregation failed: {e}", func=func, group_by=agg.group_by)
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
        chart_type_str = self._chart_type_str(spec)
        logger.info(f"Rendering {chart_type_str} chart", x=spec.x, y=spec.y)

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

            # Fuzzy-match column names (handle case mismatches from LLM)
            if spec.x and spec.x not in df.columns:
                spec = self._fix_column_name(spec, "x", df)
            if spec.y and spec.y not in df.columns:
                spec = self._fix_column_name(spec, "y", df)

            # Apply sorting (after aggregation, before charting)
            opts = spec.options
            if opts.sort and spec.y and spec.y in df.columns:
                ascending = opts.sort.lower() in ("ascending", "asc")
                df = df.sort_values(by=spec.y, ascending=ascending)

            # Apply row limit
            if opts.limit and opts.limit > 0:
                df = df.head(opts.limit)

            # Create chart based on type
            fig = self._create_chart(spec, df)

            logger.info("Chart rendered successfully", chart_type=chart_type_str)
            return fig

        except RenderError:
            raise
        except Exception as e:
            logger.error(f"Rendering failed: {e}", chart_type=chart_type_str)
            raise RenderError(f"Failed to render {chart_type_str} chart: {str(e)}")

    @staticmethod
    def _chart_type_str(spec: VisualizationSpec) -> str:
        """Safely get chart type as a plain string."""
        ct = spec.chart_type
        return ct.value if hasattr(ct, "value") else str(ct)

    @staticmethod
    def _fix_column_name(spec: VisualizationSpec, axis: str, df: pd.DataFrame) -> VisualizationSpec:
        """Try case-insensitive match for a missing column."""
        col = getattr(spec, axis)
        col_lower = col.lower()
        for real_col in df.columns:
            if real_col.lower() == col_lower:
                spec = spec.model_copy(update={axis: real_col})
                logger.info(f"Column '{col}' matched to '{real_col}' (case-insensitive)")
                return spec
        raise RenderError(f"Column '{col}' not found in data. Available: {', '.join(df.columns[:15])}")

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
        ct = self._chart_type_str(spec)

        # Common parameters
        common_params = {
            "title": opts.title or spec.explain,
            "height": opts.height,
            "width": opts.width,
            "color": opts.color if opts.color and opts.color in df.columns else None,
            "template": "plotly_white",
        }

        # Chart-specific rendering
        if ct == "line":
            fig = px.line(df, x=spec.x, y=spec.y, **common_params, markers=True)

        elif ct == "bar":
            fig = px.bar(df, x=spec.x, y=spec.y, **common_params)

        elif ct == "scatter":
            size = opts.size if opts.size and opts.size in df.columns else None
            fig = px.scatter(df, x=spec.x, y=spec.y, size=size, **common_params)

        elif ct == "histogram":
            # Remove 'color' from common_params if not applicable
            hist_params = {k: v for k, v in common_params.items()}
            fig = px.histogram(df, x=spec.x, **hist_params)

        elif ct == "box":
            fig = px.box(df, x=spec.x, y=spec.y, **common_params)

        elif ct == "heatmap":
            if spec.x and spec.y:
                pivot = df.pivot_table(values=spec.y, index=spec.x, aggfunc="mean")
                hm_params = {k: v for k, v in common_params.items() if k != "color"}
                fig = px.imshow(pivot, **hm_params, color_continuous_scale="Viridis")
            else:
                raise RenderError("Heatmap requires both x and y columns")

        elif ct == "pie":
            pie_params = {k: v for k, v in common_params.items() if k != "color"}
            fig = px.pie(df, names=spec.x, values=spec.y, **pie_params, hole=0.35)

        elif ct == "area":
            fig = px.area(df, x=spec.x, y=spec.y, **common_params)

        else:
            raise RenderError(f"Unsupported chart type: {ct}")

        # Apply axis labels and scales
        fig.update_xaxes(title=opts.xlabel or spec.x, type="log" if opts.log_x else None)
        fig.update_yaxes(title=opts.ylabel or spec.y, type="log" if opts.log_y else None)

        # Legend
        fig.update_layout(
            showlegend=opts.show_legend,
            font=dict(family="Inter, system-ui, sans-serif", size=13),
            margin=dict(l=60, r=30, t=60, b=50),
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=px.colors.qualitative.Set2,
        )

        return fig
