#!/usr/bin/env python
"""
Development server runner script.
Quick way to start the Streamlit app with proper configuration.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Run the Streamlit development server."""
    import streamlit.web.cli as stcli

    # Check for .env file
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  Warning: .env file not found!")
        print("📝 Copy .env.example to .env and add your OPENAI_API_KEY")
        print()
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            sys.exit(1)

    # Set up Streamlit app path
    app_path = project_root / "streamlit_frontend" / "app.py"

    # Run Streamlit
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "localhost",
        "--server.port",
        "8501",
    ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
