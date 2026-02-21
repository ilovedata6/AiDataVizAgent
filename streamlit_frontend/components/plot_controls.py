"""
Plot controls component for manual visualization editing.
Allows users to modify specs and regenerate charts.
"""

import streamlit as st
import json
from typing import Any

from services.planner.spec_schema import VisualizationSpec, ChartType


def render_plot_controls(
    current_spec: VisualizationSpec | None, available_columns: list[str]
) -> VisualizationSpec | None:
    """
    Render controls for editing visualization spec.

    Args:
        current_spec: Current VisualizationSpec (if any)
        available_columns: List of available column names

    Returns:
        Modified VisualizationSpec if user makes changes, else None
    """
    if current_spec is None:
        return None

    st.sidebar.header("🎨 Plot Controls")

    with st.sidebar.expander("Edit Visualization", expanded=False):
        # Get current chart type as string
        current_chart_type = (
            current_spec.chart_type.value 
            if isinstance(current_spec.chart_type, ChartType) 
            else current_spec.chart_type
        )
        
        # Chart type selector
        chart_type = st.selectbox(
            "Chart Type",
            options=[ct.value for ct in ChartType],
            index=[ct.value for ct in ChartType].index(current_chart_type),
        )

        # X and Y axis selectors
        x_col = st.selectbox(
            "X Axis",
            options=["None"] + available_columns,
            index=(
                available_columns.index(current_spec.x) + 1
                if current_spec.x and current_spec.x in available_columns
                else 0
            ),
        )

        y_col = st.selectbox(
            "Y Axis",
            options=["None"] + available_columns,
            index=(
                available_columns.index(current_spec.y) + 1
                if current_spec.y and current_spec.y in available_columns
                else 0
            ),
        )

        # Color encoding
        color_col = st.selectbox(
            "Color By",
            options=["None"] + available_columns,
            index=(
                available_columns.index(current_spec.options.color) + 1
                if current_spec.options.color
                and current_spec.options.color in available_columns
                else 0
            ),
        )

        # Chart title
        title = st.text_input("Title", value=current_spec.options.title or "")

        # Apply button
        if st.button("Apply Changes", type="primary"):
            # Create modified spec
            modified_spec = current_spec.model_copy(deep=True)
            modified_spec.chart_type = ChartType(chart_type)
            modified_spec.x = x_col if x_col != "None" else None
            modified_spec.y = y_col if y_col != "None" else None
            modified_spec.options.color = color_col if color_col != "None" else None
            modified_spec.options.title = title if title else None

            st.success("✅ Changes applied!")
            return modified_spec

    return None


def render_spec_json_editor(current_spec: VisualizationSpec | None) -> VisualizationSpec | None:
    """
    Render JSON editor for advanced spec editing.

    Args:
        current_spec: Current VisualizationSpec

    Returns:
        Modified spec if valid JSON provided, else None
    """
    if current_spec is None:
        return None

    st.sidebar.header("🔧 Advanced")

    with st.sidebar.expander("JSON Editor", expanded=False):
        spec_json = st.text_area(
            "Edit Spec JSON",
            value=current_spec.model_dump_json(indent=2),
            height=300,
            help="Advanced: edit the raw visualization specification",
        )

        if st.button("Validate & Apply"):
            try:
                spec_dict = json.loads(spec_json)
                modified_spec = VisualizationSpec(**spec_dict)
                st.success("✅ Spec validated and applied!")
                return modified_spec
            except Exception as e:
                st.error(f"❌ Invalid spec: {str(e)}")

    return None


def render_export_controls(fig: Any, spec: VisualizationSpec) -> None:
    """
    Render export controls for saving charts.

    Args:
        fig: Plotly or Matplotlib figure
        spec: Current VisualizationSpec
    """
    st.sidebar.header("💾 Export")

    with st.sidebar.expander("Download Chart"):
        # Determine figure type
        try:
            import plotly.graph_objects as go

            ct_str = spec.chart_type if isinstance(spec.chart_type, str) else spec.chart_type.value

            if isinstance(fig, go.Figure):
                # Plotly figure - can export as HTML or PNG
                col1, col2 = st.columns(2)

                with col1:
                    html_buffer = fig.to_html(include_plotlyjs="cdn")
                    st.download_button(
                        "HTML",
                        data=html_buffer,
                        file_name=f"chart_{ct_str}.html",
                        mime="text/html",
                    )

                with col2:
                    # PNG export requires kaleido
                    try:
                        img_bytes = fig.to_image(format="png")
                        st.download_button(
                            "PNG",
                            data=img_bytes,
                            file_name=f"chart_{ct_str}.png",
                            mime="image/png",
                        )
                    except Exception:
                        st.caption("PNG export requires kaleido package")

        except Exception as e:
            st.caption("Export not available for this chart type")


def render_recommended_charts(recommendations: list[dict[str, Any]]) -> str | None:
    """
    Render recommended chart types based on data profiling.

    Args:
        recommendations: List of recommended chart specs

    Returns:
        Selected recommendation description if user clicks one
    """
    if not recommendations:
        return None

    st.subheader("💡 Suggested Visualizations")

    cols = st.columns(min(3, len(recommendations)))

    for i, rec in enumerate(recommendations[:3]):
        with cols[i]:
            if st.button(
                f"{rec['chart_type'].title()}\n\n{rec['title']}",
                key=f"rec_{i}",
                use_container_width=True,
            ):
                return rec["description"] + f" using {rec['chart_type']} chart"

    return None
