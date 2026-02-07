"""
Chat UI component with message bubbles and append-only history.
Displays user and assistant messages with formatting and metadata.
"""

import streamlit as st
from datetime import datetime

from services.memory.sql_memory import ChatMessage


def render_chat_message(message: ChatMessage) -> None:
    """
    Render a single chat message with styling.

    Args:
        message: ChatMessage to display
    """
    # Determine avatar based on role
    avatar = "🧑" if message.role == "user" else "🤖"

    with st.chat_message(message.role, avatar=avatar):
        st.markdown(message.content)

        # Show metadata in expander if available
        if message.metadata:
            with st.expander("Details"):
                if "chart_type" in message.metadata:
                    st.caption(f"Chart Type: {message.metadata['chart_type']}")
                if "tokens_used" in message.metadata:
                    st.caption(f"Tokens: {message.metadata['tokens_used']}")
                if "duration_seconds" in message.metadata:
                    st.caption(f"Duration: {message.metadata['duration_seconds']:.2f}s")
                st.caption(f"Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


def render_chat_history(messages: list[ChatMessage]) -> None:
    """
    Render complete chat history.

    Args:
        messages: List of ChatMessage in chronological order
    """
    if not messages:
        st.info("👋 Welcome! Upload a dataset and ask me to create visualizations.")
        return

    for message in messages:
        render_chat_message(message)


def chat_input_form() -> str | None:
    """
    Render chat input form.

    Returns:
        User input text, or None if no input
    """
    user_input = st.chat_input("Ask me to visualize your data...")
    return user_input


def display_typing_indicator() -> None:
    """Show a typing indicator while processing."""
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("_Thinking..._")
