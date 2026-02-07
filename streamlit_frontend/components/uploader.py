"""
File uploader component with validation and preview.
Handles CSV/XLSX uploads with security checks.
"""

import streamlit as st
import pandas as pd
from io import BytesIO

from services.ingest.parser import DatasetParser, DatasetSchema
from core.exceptions import FileUploadError, FileParsingError, get_user_friendly_message


def render_file_uploader() -> tuple[pd.DataFrame, DatasetSchema] | tuple[None, None]:
    """
    Render file uploader and return parsed data.

    Returns:
        Tuple of (DataFrame, DatasetSchema) if file uploaded, else (None, None)
    """
    st.sidebar.header("📁 Upload Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or XLSX file",
        type=["csv", "xlsx"],
        help="Upload your dataset to start creating visualizations",
    )

    if uploaded_file is None:
        return None, None

    try:
        # Parse file
        parser = DatasetParser()
        file_bytes = uploaded_file.read()
        df = parser.parse_file(file_bytes, uploaded_file.name)

        # Extract schema
        schema = parser.extract_schema(df)

        # Display schema info
        st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
        st.sidebar.metric("Rows", schema.row_count)
        st.sidebar.metric("Columns", len(schema.columns))
        st.sidebar.metric("Memory", f"{schema.memory_usage_mb:.2f} MB")

        return df, schema

    except (FileUploadError, FileParsingError) as e:
        st.sidebar.error(f"❌ {get_user_friendly_message(e)}")
        return None, None

    except Exception as e:
        st.sidebar.error(f"❌ Unexpected error: {str(e)}")
        return None, None


def render_schema_preview(schema: DatasetSchema) -> None:
    """
    Render dataset schema preview in main area.

    Args:
        schema: DatasetSchema to display
    """
    st.subheader("📊 Dataset Preview")

    # Column types
    with st.expander("Column Information", expanded=True):
        col_data = []
        for col in schema.columns:
            dtype = schema.dtypes[col]
            missing = schema.missing_values.get(col, 0)
            missing_pct = (missing / schema.row_count * 100) if schema.row_count > 0 else 0

            col_data.append(
                {"Column": col, "Type": dtype, "Missing": f"{missing} ({missing_pct:.1f}%)"}
            )

        st.dataframe(col_data, use_container_width=True, hide_index=True)

    # Sample rows
    with st.expander("Sample Data"):
        st.dataframe(schema.sample_rows, use_container_width=True)
