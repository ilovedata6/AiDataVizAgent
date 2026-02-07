"""
Seaborn renderer: fallback static chart renderer.
Used when Plotly fails or for quick static visualizations.
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Use non-interactive backend

from matplotlib.figure import Figure

from core.exceptions import RenderError
from core.logging import get_logger
from services.planner.spec_schema import VisualizationSpec
from services.renderer.plotly_renderer import PlotlyRenderer

logger = get_logger(__name__)


class SeabornRenderer:
    """
    Fallback renderer using Seaborn/Matplotlib for static charts.
    Uses same safe transformation logic as PlotlyRenderer.
    """

    def __init__(self) -> None:
        """Initialize Seaborn renderer."""
        # Use PlotlyRenderer for data transformations
        self.plotly_renderer = PlotlyRenderer()

        # Set Seaborn style
        sns.set_theme(style="whitegrid")

        logger.info("Seaborn renderer initialized")

    def render(self, spec: VisualizationSpec, df: pd.DataFrame) -> Figure:
        """
        Render VisualizationSpec to Matplotlib Figure.

        Args:
            spec: Visualization specification
            df: Source DataFrame

        Returns:
            Matplotlib Figure

        Raises:
            RenderError: If rendering fails
        """
        logger.info(f"Rendering {spec.chart_type} chart with Seaborn", x=spec.x, y=spec.y)

        try:
            # Apply filters and transformations using PlotlyRenderer logic
            if spec.filters:
                df = self.plotly_renderer._apply_filters(df, spec.filters)

            if spec.transformations:
                df = self.plotly_renderer._apply_transformations(df, spec.transformations)

            if spec.aggregate:
                df = self.plotly_renderer._apply_aggregation(df, spec.aggregate, spec.y or "count")

            # Validate columns
            if spec.x and spec.x not in df.columns:
                raise RenderError(f"Column '{spec.x}' not found in data")
            if spec.y and spec.y not in df.columns:
                raise RenderError(f"Column '{spec.y}' not found in data")

            # Create figure
            fig = self._create_chart(spec, df)

            logger.info("Chart rendered successfully with Seaborn", chart_type=spec.chart_type)
            return fig

        except RenderError:
            raise
        except Exception as e:
            logger.error(f"Seaborn rendering failed: {e}", chart_type=spec.chart_type)
            raise RenderError(f"Failed to render {spec.chart_type} chart: {str(e)}")

    def _create_chart(self, spec: VisualizationSpec, df: pd.DataFrame) -> Figure:
        """
        Create Matplotlib chart based on chart type.

        Args:
            spec: Visualization specification
            df: Prepared DataFrame

        Returns:
            Matplotlib Figure
        """
        opts = spec.options

        # Create figure
        fig, ax = plt.subplots(figsize=(opts.width / 100, opts.height / 100))

        # Chart-specific rendering
        if spec.chart_type.value == "line":
            if opts.color and opts.color in df.columns:
                for group in df[opts.color].unique():
                    subset = df[df[opts.color] == group]
                    ax.plot(subset[spec.x], subset[spec.y], label=group, marker="o")
                ax.legend()
            else:
                ax.plot(df[spec.x], df[spec.y], marker="o")

        elif spec.chart_type.value == "bar":
            if opts.color and opts.color in df.columns:
                sns.barplot(data=df, x=spec.x, y=spec.y, hue=opts.color, ax=ax)
            else:
                sns.barplot(data=df, x=spec.x, y=spec.y, ax=ax)

        elif spec.chart_type.value == "scatter":
            if opts.color and opts.color in df.columns:
                sns.scatterplot(data=df, x=spec.x, y=spec.y, hue=opts.color, ax=ax)
            else:
                sns.scatterplot(data=df, x=spec.x, y=spec.y, ax=ax)

        elif spec.chart_type.value == "histogram":
            if opts.color and opts.color in df.columns:
                for group in df[opts.color].unique():
                    subset = df[df[opts.color] == group]
                    ax.hist(subset[spec.x], alpha=0.6, label=group, bins=30)
                ax.legend()
            else:
                ax.hist(df[spec.x], bins=30)

        elif spec.chart_type.value == "box":
            if opts.color and opts.color in df.columns:
                sns.boxplot(data=df, x=spec.x, y=spec.y, hue=opts.color, ax=ax)
            else:
                sns.boxplot(data=df, x=spec.x, y=spec.y, ax=ax)

        elif spec.chart_type.value == "heatmap":
            if spec.x and spec.y:
                pivot = df.pivot_table(values=spec.y, index=spec.x, aggfunc="mean")
                sns.heatmap(pivot, annot=True, fmt=".2f", ax=ax)
            else:
                raise RenderError("Heatmap requires both x and y columns")

        elif spec.chart_type.value == "area":
            if opts.color and opts.color in df.columns:
                for group in df[opts.color].unique():
                    subset = df[df[opts.color] == group]
                    ax.fill_between(subset[spec.x], subset[spec.y], alpha=0.5, label=group)
                ax.legend()
            else:
                ax.fill_between(df[spec.x], df[spec.y], alpha=0.5)

        else:
            raise RenderError(f"Unsupported chart type for Seaborn: {spec.chart_type}")

        # Apply labels and title
        ax.set_xlabel(opts.xlabel or spec.x or "")
        ax.set_ylabel(opts.ylabel or spec.y or "")
        ax.set_title(opts.title or spec.explain)

        # Apply log scales
        if opts.log_x:
            ax.set_xscale("log")
        if opts.log_y:
            ax.set_yscale("log")

        plt.tight_layout()
        return fig
