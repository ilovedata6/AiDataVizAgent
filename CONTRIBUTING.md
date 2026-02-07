# Contributing to AI Data Visualization Agent

Thank you for your interest in contributing! I'm excited to collaborate with you on making this project better.

---

## 🎯 How I Approach Contributions

I welcome contributions of all kinds:
- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🧪 Test additions
- 🎨 UI enhancements
- 🔧 Refactoring

---

## 🚀 Getting Started

### 1. Set Up Your Development Environment

I use `uv` for dependency management, so you'll need that installed:

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/ai-data-viz-agent.git
cd ai-data-viz-agent

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (including dev dependencies)
uv pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 2. Create a Branch

I follow a simple branching strategy:

```bash
# Feature branches
git checkout -b feature/your-feature-name

# Bug fix branches
git checkout -b fix/issue-description

# Documentation branches
git checkout -b docs/what-you-are-documenting
```

### 3. Make Your Changes

I have some guidelines I'd like you to follow:

#### Code Style

I use **Ruff** and **Black** for consistent formatting:

```bash
# Format code
uv run black .

# Check linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .
```

#### Type Hints

I use type hints everywhere. Please add them to all new functions:

```python
def process_data(df: pd.DataFrame, column: str) -> pd.Series:
    """Process data with type safety."""
    return df[column]
```

#### Documentation

I document all public functions with docstrings:

```python
def calculate_statistics(data: list[float]) -> dict[str, float]:
    """
    Calculate summary statistics for a list of numbers.

    Args:
        data: List of numeric values

    Returns:
        Dictionary with mean, median, and std

    Raises:
        ValueError: If data is empty
    """
    ...
```

### 4. Write Tests

I require tests for all new features. Here's my testing philosophy:

- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test component interactions
- **Mock external services**: Don't call real APIs in tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_planner.py

# Run specific test
uv run pytest tests/test_planner.py::test_extract_json_from_plain_response
```

### 5. Commit Your Changes

I prefer clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "feat: Add histogram chart type support"
git commit -m "fix: Handle missing columns in filter application"
git commit -m "docs: Update README with Docker deployment instructions"
git commit -m "test: Add tests for CSV delimiter detection"

# Follow conventional commits format
# <type>: <description>
# Types: feat, fix, docs, test, refactor, style, chore
```

### 6. Push and Create a Pull Request

```bash
git push origin your-branch-name
```

Then open a pull request on GitHub. I'll review it as soon as possible!

---

## 📋 Pull Request Guidelines

When I review PRs, I look for:

### Required

- ✅ **Tests**: New features must include tests
- ✅ **Documentation**: Update relevant docs (README, docstrings)
- ✅ **Linting**: Code must pass `ruff` and `black` checks
- ✅ **Type Hints**: All functions must have type annotations
- ✅ **Description**: Clear PR description explaining what and why

### Good to Have

- 📝 Update CHANGELOG.md
- 🧪 Include integration tests if applicable
- 📊 Add screenshots for UI changes
- 🔍 Reference related issues

---

## 🧪 Testing Checklist

Before submitting, I'd like you to verify:

```bash
# 1. All tests pass
uv run pytest -v

# 2. Code is formatted
uv run black .

# 3. No linting errors
uv run ruff check .

# 4. Type checking passes (optional but appreciated)
uv run mypy .

# 5. Application runs
uv run streamlit run streamlit_frontend/app.py
```

---

## 🏗️ Development Workflow

Here's how I typically develop new features:

### For New Chart Types

1. Add chart type to `ChartType` enum in `services/planner/spec_schema.py`
2. Implement rendering logic in `services/renderer/plotly_renderer.py`
3. Add fallback in `services/renderer/seaborn_renderer.py` (if applicable)
4. Write tests in `tests/test_renderer.py`
5. Update LLM prompt in `services/planner/planner.py`
6. Update documentation

### For New Data Sources

1. Create parser in `services/ingest/`
2. Add file type validation
3. Implement schema extraction
4. Write tests in `tests/test_ingest.py`
5. Update file uploader component
6. Update documentation

### For UI Changes

1. Modify components in `streamlit_frontend/components/`
2. Test manually in browser
3. Check responsive behavior
4. Update screenshots in documentation

---

## 🐛 Reporting Bugs

I track bugs through GitHub Issues. When reporting a bug, please include:

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Upload file '...'
2. Ask question '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g., Windows 11, macOS 14, Ubuntu 22.04]
 - Python version: [e.g., 3.12.1]
 - uv version: [e.g., 0.1.0]

**Additional context**
Any other relevant information.
```

---

## 💡 Requesting Features

I love hearing ideas! To request a feature:

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other information, mockups, or examples.
```

---

## 📚 Resources

Here are resources I found helpful:

- **Streamlit Docs**: https://docs.streamlit.io/
- **Plotly Docs**: https://plotly.com/python/
- **OpenAI API Docs**: https://platform.openai.com/docs/
- **uv Docs**: https://docs.astral.sh/uv/
- **Pandas Docs**: https://pandas.pydata.org/docs/
- **Pydantic Docs**: https://docs.pydantic.dev/

---

## 🎖️ Recognition

I maintain a list of contributors in the README. Your contributions will be recognized!

---

## ❓ Questions?

If you have questions:

1. Check existing issues and discussions
2. Read the [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions
3. Open a GitHub Discussion
4. Reach out via GitHub Issues

---

## 🤝 Code of Conduct

I maintain a welcoming and inclusive environment. Please read and follow our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! Together, we're building something amazing. 🚀**
