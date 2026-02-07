"""
Main Streamlit application entry point.
Enterprise-grade AI Data Visualization Agent.
"""

import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from core.config import get_settings
from core.exceptions import get_user_friendly_message
from core.logging import get_logger, setup_logging
from services.ingest.parser import DatasetSchema
from services.llm.openai_client import OpenAIClient
from services.memory.sql_memory import ChatMessage, SQLMemory
from services.planner.planner import VisualizationPlanner
from services.planner.spec_schema import VisualizationSpec
from services.profile.profiler import DataProfiler
from services.renderer.plotly_renderer import PlotlyRenderer
from services.renderer.seaborn_renderer import SeabornRenderer
from streamlit_frontend.components.chat import (
    chat_input_form,
    render_chat_history,
)
from streamlit_frontend.components.plot_controls import (
    render_export_controls,
    render_plot_controls,
    render_recommended_charts,
    render_spec_json_editor,
)
from streamlit_frontend.components.uploader import (
    render_file_uploader,
    render_schema_preview,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="AI Data Viz Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info("New session started", session_id=st.session_state.session_id)

    if "memory" not in st.session_state:
        st.session_state.memory = SQLMemory()

    if "llm_client" not in st.session_state:
        st.session_state.llm_client = OpenAIClient()

    if "planner" not in st.session_state:
        st.session_state.planner = VisualizationPlanner(st.session_state.llm_client)

    if "renderer" not in st.session_state:
        st.session_state.renderer = PlotlyRenderer()

    if "fallback_renderer" not in st.session_state:
        st.session_state.fallback_renderer = SeabornRenderer()

    if "profiler" not in st.session_state:
        st.session_state.profiler = DataProfiler()

    if "current_df" not in st.session_state:
        st.session_state.current_df = None

    if "current_schema" not in st.session_state:
        st.session_state.current_schema = None

    if "current_profile" not in st.session_state:
        st.session_state.current_profile = None

    if "current_spec" not in st.session_state:
        st.session_state.current_spec = None

    if "current_figure" not in st.session_state:
        st.session_state.current_figure = None


def add_message_to_history(role: str, content: str, metadata: dict | None = None) -> None:
    """
    Add message to persistent history.

    Args:
        role: 'user' or 'assistant'
        content: Message content
        metadata: Optional metadata dictionary
    """
    message = ChatMessage(
        session_id=st.session_state.session_id,
        role=role,
        content=content,
        metadata=metadata or {},
        timestamp=datetime.utcnow(),
    )

    st.session_state.memory.add_message(message)


def get_chat_history() -> list[ChatMessage]:
    """Retrieve chat history for current session."""
    return st.session_state.memory.get_session_history(
        st.session_state.session_id, limit=get_settings().max_chat_history
    )


def process_user_query(query: str) -> None:
    """
    Process user query and generate visualization.

    Args:
        query: User's natural language query
    """
    if st.session_state.current_df is None or st.session_state.current_schema is None:
        st.error("❌ Please upload a dataset first!")
        return

    # Add user message
    add_message_to_history("user", query)

    # Get chat history for context
    history = get_chat_history()
    history_for_llm = [{"role": msg.role, "content": msg.content} for msg in history[-5:]]

    try:
        # Generate visualization spec
        with st.spinner("🤔 Planning visualization..."):
            response, usage = st.session_state.planner.plan(
                user_query=query,
                schema=st.session_state.current_schema,
                chat_history=history_for_llm,
            )

        # Check for errors
        if response.error:
            error_msg = f"I couldn't generate a visualization: {response.error.message}"
            if response.error.candidates:
                error_msg += f"\n\nDid you mean: {', '.join(response.error.candidates[:3])}?"
            add_message_to_history("assistant", error_msg)
            st.error(error_msg)
            return

        if response.spec is None:
            add_message_to_history("assistant", "I couldn't understand that request.")
            st.error("Failed to generate specification")
            return

        # Store spec
        st.session_state.current_spec = response.spec

        # Render chart
        with st.spinner("📊 Rendering chart..."):
            try:
                fig = st.session_state.renderer.render(
                    response.spec, st.session_state.current_df
                )
                st.session_state.current_figure = fig
                render_success = True
            except Exception as e:
                logger.warning(f"Plotly rendering failed, using fallback: {e}")
                try:
                    fig = st.session_state.fallback_renderer.render(
                        response.spec, st.session_state.current_df
                    )
                    st.session_state.current_figure = fig
                    render_success = True
                except Exception as fallback_error:
                    logger.error(f"Fallback rendering also failed: {fallback_error}")
                    render_success = False

        # Add assistant response
        if render_success:
            assistant_msg = f"✅ {response.spec.explain}"
            metadata = {
                "chart_type": response.spec.chart_type.value,
                "tokens_used": usage.total_tokens,
                "duration_seconds": usage.duration_seconds,
            }
            add_message_to_history("assistant", assistant_msg, metadata)
        else:
            add_message_to_history("assistant", "❌ Failed to render chart. Please try again.")

    except Exception as e:
        error_msg = get_user_friendly_message(e)
        add_message_to_history("assistant", f"❌ Error: {error_msg}")
        st.error(f"Error: {error_msg}")
        logger.error("Query processing failed", error=str(e))


def main() -> None:
    """Main application logic."""
    initialize_session_state()

    # Header
    st.title("📊 AI Data Visualization Agent")
    st.caption("Upload data, ask questions in natural language, and get interactive charts!")

    # Sidebar: File upload and controls
    df, schema = render_file_uploader()

    if df is not None and schema is not None:
        st.session_state.current_df = df
        st.session_state.current_schema = schema

        # Profile dataset if not already done
        if st.session_state.current_profile is None:
            with st.spinner("Analyzing dataset..."):
                st.session_state.current_profile = st.session_state.profiler.profile_dataset(df)

    # Main area layout
    if st.session_state.current_schema is not None:
        # Two column layout
        col1, col2 = st.columns([1, 1])

        with col1:
            # Chat interface
            st.subheader("💬 Chat")

            # Display chat history
            history = get_chat_history()
            render_chat_history(history)

            # Chat input
            user_input = chat_input_form()
            if user_input:
                process_user_query(user_input)
                st.rerun()

        with col2:
            # Visualization area
            st.subheader("📈 Visualization")

            if st.session_state.current_figure is not None:
                # Display chart
                try:
                    import plotly.graph_objects as go

                    if isinstance(st.session_state.current_figure, go.Figure):
                        st.plotly_chart(
                            st.session_state.current_figure, use_container_width=True
                        )
                    else:
                        st.pyplot(st.session_state.current_figure)
                except Exception as e:
                    st.error(f"Error displaying chart: {str(e)}")

                # Export controls
                if st.session_state.current_spec:
                    render_export_controls(
                        st.session_state.current_figure, st.session_state.current_spec
                    )

            else:
                # Show recommended charts
                if st.session_state.current_profile:
                    selected_rec = render_recommended_charts(
                        st.session_state.current_profile.recommended_charts
                    )
                    if selected_rec:
                        process_user_query(selected_rec)
                        st.rerun()
                else:
                    render_schema_preview(st.session_state.current_schema)

        # Plot controls in sidebar
        if st.session_state.current_spec and st.session_state.current_schema:
            # Manual controls
            modified_spec = render_plot_controls(
                st.session_state.current_spec, st.session_state.current_schema.columns
            )

            if modified_spec:
                st.session_state.current_spec = modified_spec
                # Re-render
                try:
                    fig = st.session_state.renderer.render(
                        modified_spec, st.session_state.current_df
                    )
                    st.session_state.current_figure = fig
                    st.rerun()
                except Exception as e:
                    st.error(f"Re-render failed: {str(e)}")

            # JSON editor
            json_modified = render_spec_json_editor(st.session_state.current_spec)
            if json_modified:
                st.session_state.current_spec = json_modified
                st.rerun()

    else:
        # No dataset loaded - show welcome
        st.info("👈 Start by uploading a CSV or XLSX file from the sidebar")

        # Show features
        st.subheader("✨ Features")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🤖 AI-Powered**")
            st.caption("Natural language to visualization")

        with col2:
            st.markdown("**📊 Interactive Charts**")
            st.caption("Plotly-based interactive plots")

        with col3:
            st.markdown("**💾 Memory**")
            st.caption("Context-aware across sessions")

    # Footer
    st.sidebar.divider()
    st.sidebar.caption("AI Data Viz Agent v0.1.0")

    # Usage stats
    if hasattr(st.session_state, "llm_client"):
        usage = st.session_state.llm_client.get_usage_summary()
        st.sidebar.caption(f"LLM Calls: {usage['total_calls']} | Tokens: {usage['total_tokens']}")


if __name__ == "__main__":
    main()
