# Quick Setup Guide

Get up and running with the AI Data Visualization Agent in minutes!

## Prerequisites

- Python 3.12
- [uv](https://astral.sh/uv/) package manager
- OpenAI API key

## Step-by-Step Setup

### 1. Install uv (if not already installed)

**Windows (PowerShell)**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup

```bash
# Navigate to the project directory
cd "e:\RAGs\AI Data Viz Agent"

# Create virtual environment with Python 3.12
uv venv --python 3.12

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Open .env and add your OpenAI API key
# OPENAI_API_KEY=sk-proj-your-key-here
```

### 4. Run the Application

```bash
# Option 1: Using uv
uv run streamlit run streamlit_frontend/app.py

# Option 2: Using the dev script
python scripts/run_dev.py

# Option 3: Direct streamlit command
streamlit run streamlit_frontend/app.py
```

The app will open in your browser at http://localhost:8501

## Verification

To verify everything is working:

```bash
# Run tests
uv run pytest -v

# Check linting
uv run ruff check .

# Format code
uv run black .
```

## Docker Setup (Alternative)

If you prefer Docker:

```bash
# Build image
docker build -t ai-viz-agent:latest .

# Create .env file with your API key
# Then run:
docker-compose up
```

## Troubleshooting

### "Module not found" errors
```bash
# Reinstall dependencies
uv pip install -e .
```

### "OpenAI API key not found"
```bash
# Make sure .env file exists and contains:
# OPENAI_API_KEY=sk-proj-...
```

### Port 8501 already in use
```bash
# Use a different port
streamlit run streamlit_frontend/app.py --server.port 8502
```

## Next Steps

- Read [README.md](../README.md) for full documentation
- Check [ARCHITECTURE.md](../ARCHITECTURE.md) to understand the design
- See [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute

## Quick Test

Once running, try this:

1. Upload a CSV file (or create a simple one with sales data)
2. Ask: "Show me a histogram of values"
3. Watch as the AI generates an interactive chart!

---

**Need help? Open an issue on GitHub!**
