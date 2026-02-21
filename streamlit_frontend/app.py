"""
Main Streamlit application entry point.
Enterprise-grade AI Data Visualization Agent.
"""

import uuid
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import get_settings
from core.exceptions import get_user_friendly_message
from core.logging import get_logger, setup_logging
from services.ingest.parser import DatasetParser, DatasetSchema
from services.llm.openai_client import OpenAIClient
from services.memory.sql_memory import ChatMessage, SQLMemory
from services.planner.planner import VisualizationPlanner
from services.planner.spec_schema import VisualizationSpec
from services.profile.profiler import DataProfiler
from services.renderer.plotly_renderer import PlotlyRenderer
from services.renderer.seaborn_renderer import SeabornRenderer

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

# ── Custom CSS (theme-safe — no hardcoded text/bg colors) ──────────
st.markdown("""
<style>
    .block-container { padding-top: 3rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state ───────────────────────────────────────────────────

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "session_id": str(uuid.uuid4()),
        "memory": None,
        "llm_client": None,
        "planner": None,
        "renderer": None,
        "fallback_renderer": None,
        "profiler": None,
        "current_df": None,
        "current_schema": None,
        "current_profile": None,
        "current_spec": None,
        "current_figure": None,
        "suggestions": None,
        "chat_messages": [],  # lightweight in-session list
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Lazy-init heavy objects
    if st.session_state.memory is None:
        st.session_state.memory = SQLMemory()
    if st.session_state.llm_client is None:
        st.session_state.llm_client = OpenAIClient()
    if st.session_state.planner is None:
        st.session_state.planner = VisualizationPlanner(st.session_state.llm_client)
    if st.session_state.renderer is None:
        st.session_state.renderer = PlotlyRenderer()
    if st.session_state.fallback_renderer is None:
        st.session_state.fallback_renderer = SeabornRenderer()
    if st.session_state.profiler is None:
        st.session_state.profiler = DataProfiler()


# ── Helpers ─────────────────────────────────────────────────────────

def add_chat(role: str, content: str, metadata: dict | None = None) -> None:
    """Append a message to the in-session chat list and SQL memory."""
    msg = {
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow(),
    }
    st.session_state.chat_messages.append(msg)

    # Also persist
    try:
        st.session_state.memory.add_message(
            ChatMessage(
                session_id=st.session_state.session_id,
                role=role,
                content=content,
                metadata=metadata or {},
                timestamp=msg["timestamp"],
            )
        )
    except Exception:
        pass  # non-critical


def render_chat() -> None:
    """Display all chat messages."""
    for msg in st.session_state.chat_messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("metadata"):
                meta = msg["metadata"]
                parts = []
                if "chart_type" in meta:
                    parts.append(f"📊 {meta['chart_type']}")
                if "tokens_used" in meta:
                    parts.append(f"🔤 {meta['tokens_used']} tokens")
                if "duration_seconds" in meta:
                    parts.append(f"⏱ {meta['duration_seconds']:.1f}s")
                if parts:
                    st.caption(" · ".join(parts))


def process_user_query(query: str) -> None:
    """Process a user question → plan → render → display."""
    if st.session_state.current_df is None or st.session_state.current_schema is None:
        st.error("Please upload a dataset first!")
        return

    add_chat("user", query)

    # Gather recent history for LLM context
    recent = st.session_state.chat_messages[-6:]
    history_for_llm = [{"role": m["role"], "content": m["content"]} for m in recent]

    try:
        with st.spinner("🤔 Thinking…"):
            response, usage = st.session_state.planner.plan(
                user_query=query,
                schema=st.session_state.current_schema,
                chat_history=history_for_llm,
            )

        # Error from planner / LLM
        if response.error:
            error_msg = f"Sorry, I couldn't create that chart: {response.error.message}"
            if response.error.candidates:
                error_msg += f"\n\nDid you mean: **{', '.join(response.error.candidates[:5])}**?"
            add_chat("assistant", error_msg)
            return

        if response.spec is None:
            add_chat("assistant", "I couldn't understand that request. Could you rephrase?")
            return

        st.session_state.current_spec = response.spec

        # Render chart
        with st.spinner("📊 Rendering chart…"):
            fig = _render_spec(response.spec)

        if fig is not None:
            st.session_state.current_figure = fig
            add_chat(
                "assistant",
                f"✅ **{response.spec.explain}**",
                {
                    "chart_type": response.spec.chart_type if isinstance(response.spec.chart_type, str) else response.spec.chart_type.value,
                    "tokens_used": usage.total_tokens,
                    "duration_seconds": usage.duration_seconds,
                },
            )
        else:
            add_chat("assistant", "❌ Failed to render the chart. Please try a different question.")

    except Exception as e:
        error_msg = get_user_friendly_message(e)
        add_chat("assistant", f"❌ Error: {error_msg}")
        logger.error("Query processing failed", error=str(e))


def _render_spec(spec: VisualizationSpec):
    """Try Plotly, fall back to Seaborn."""
    try:
        return st.session_state.renderer.render(spec, st.session_state.current_df)
    except Exception as e:
        logger.warning(f"Plotly failed ({e}), trying Seaborn fallback")
        try:
            return st.session_state.fallback_renderer.render(spec, st.session_state.current_df)
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return None


# ── Main layout ─────────────────────────────────────────────────────

def main() -> None:
    initialize_session_state()

    # ── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📊 AI Data Viz Agent")
        st.caption("Upload data → ask questions → get charts")
        st.divider()

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx"],
            help="Max 50 MB. Supported: .csv, .xlsx",
        )

        if uploaded_file is not None:
            # Parse only once per file
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("_loaded_file_key") != file_key:
                with st.spinner("Parsing file…"):
                    parser = DatasetParser()
                    file_bytes = uploaded_file.read()
                    try:
                        df = parser.parse_file(file_bytes, uploaded_file.name)
                        df = parser.clean_dataframe(df)
                        schema = parser.extract_schema(df)
                        profile = st.session_state.profiler.profile_dataset(df)

                        st.session_state.current_df = df
                        st.session_state.current_schema = schema
                        st.session_state.current_profile = profile
                        st.session_state.current_figure = None
                        st.session_state.current_spec = None
                        st.session_state.suggestions = None
                        st.session_state.chat_messages = []
                        st.session_state._loaded_file_key = file_key
                    except Exception as e:
                        st.error(f"❌ {get_user_friendly_message(e)}")

            # Show file info
            if st.session_state.current_schema is not None:
                schema = st.session_state.current_schema
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", f"{schema.row_count:,}")
                c2.metric("Cols", len(schema.columns))
                c3.metric("Size", f"{schema.memory_usage_mb:.1f} MB")

                with st.expander("📋 Columns"):
                    for col in schema.columns:
                        dtype = schema.dtypes[col]
                        miss = schema.missing_values.get(col, 0)
                        st.caption(f"**{col}** — {dtype}" + (f"  ({miss} missing)" if miss else ""))

                with st.expander("🔍 Sample data"):
                    st.dataframe(
                        st.session_state.current_df.head(5),
                        use_container_width=True,
                        hide_index=True,
                    )

        st.divider()
        # Usage stats
        usage = st.session_state.llm_client.get_usage_summary()
        st.caption(f"LLM calls: {usage['total_calls']}  ·  Tokens: {usage['total_tokens']:,}")
        st.caption("v0.1.0")

    # ── Main area ───────────────────────────────────────────────────
    if st.session_state.current_df is None:
        # Welcome screen
        st.markdown("# 📊 AI Data Visualization Agent")
        st.markdown("Upload a **CSV** or **Excel** file in the sidebar to get started.")
        st.divider()

        cols = st.columns(3)
        with cols[0]:
            st.markdown("#### 🤖 AI-Powered")
            st.caption("Ask questions in plain English and get interactive charts automatically.")
        with cols[1]:
            st.markdown("#### 📈 Beautiful Charts")
            st.caption("Plotly-powered interactive visualizations you can download instantly.")
        with cols[2]:
            st.markdown("#### 💡 Smart Suggestions")
            st.caption("Get data-specific visualization ideas before you even ask.")
        return

    # ── Smart suggestions ───────────────────────────────────────────
    if st.session_state.suggestions is None and st.session_state.current_schema is not None:
        with st.spinner("🔍 Analyzing your data for visualization ideas…"):
            try:
                st.session_state.suggestions = st.session_state.planner.generate_suggestions(
                    st.session_state.current_schema
                )
            except Exception:
                st.session_state.suggestions = []

    # Show suggestions if no chart has been created yet
    if st.session_state.current_figure is None and st.session_state.suggestions:
        st.markdown("#### 💡 Try one of these visualizations")
        cols = st.columns(len(st.session_state.suggestions))
        for i, suggestion in enumerate(st.session_state.suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    process_user_query(suggestion)
                    st.rerun()

    # ── Two-column layout: Chat + Chart ─────────────────────────────
    chat_col, chart_col = st.columns([2, 3])

    with chat_col:
        st.markdown("#### 💬 Chat")
        render_chat()

        user_input = st.chat_input("Ask me to visualize your data…")
        if user_input:
            process_user_query(user_input)
            st.rerun()

    with chart_col:
        st.markdown("#### 📈 Visualization")

        if st.session_state.current_figure is not None:
            fig = st.session_state.current_figure

            # Display chart
            if isinstance(fig, go.Figure):
                st.plotly_chart(fig, use_container_width=True, key="main_chart")
            else:
                st.pyplot(fig, use_container_width=True)

            # ── Download buttons (prominent, right below chart) ─────
            st.markdown("---")
            dl1, dl2, dl3 = st.columns(3)

            with dl1:
                if isinstance(fig, go.Figure):
                    html_data = fig.to_html(include_plotlyjs="cdn")
                    st.download_button(
                        "⬇️  Download HTML",
                        data=html_data,
                        file_name="chart.html",
                        mime="text/html",
                        use_container_width=True,
                    )

            with dl2:
                if isinstance(fig, go.Figure):
                    try:
                        img_bytes = fig.to_image(format="png", scale=2)
                        st.download_button(
                            "⬇️  Download PNG",
                            data=img_bytes,
                            file_name="chart.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                    except Exception:
                        st.caption("PNG needs `kaleido` package")

            with dl3:
                if st.session_state.current_spec:
                    spec_json = st.session_state.current_spec.model_dump_json(indent=2)
                    st.download_button(
                        "⬇️  Download JSON Spec",
                        data=spec_json,
                        file_name="chart_spec.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            # Explanation
            if st.session_state.current_spec:
                st.info(f"**What this shows:** {st.session_state.current_spec.explain}")

        else:
            # No chart yet — show data preview
            st.dataframe(
                st.session_state.current_df.head(8),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Ask a question or click a suggestion above to create a chart.")


if __name__ == "__main__":
    main()
