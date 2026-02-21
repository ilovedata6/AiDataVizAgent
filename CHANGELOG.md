# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Smart file-based suggestions** — LLM generates 4 dataset-specific visualization ideas on upload, with profiler-based fallback
- **Download buttons** — Prominent HTML, PNG, and JSON Spec download buttons directly below every chart
- **Auto data cleaning** — `DatasetParser.clean_dataframe()` automatically strips whitespace, drops empty rows/columns, coerces numeric-looking strings, and parses date columns before visualization
- **Sort & limit support** — `ChartOptions` schema now accepts `sort` and `limit`, enabling "top N / bottom N" queries
- **Case-insensitive column matching** — `PlotlyRenderer._fix_column_name()` resolves mismatched casing from LLM output
- **LLM spec sanitizer** — `VisualizationPlanner._sanitize_spec()` strips hallucinated/invalid transformations and filters before Pydantic validation
- **Robust JSON extraction** — brace-counting scanner (`_find_json_objects`) handles arbitrarily nested JSON; trailing-comma and single-quote cleanup for malformed LLM output
- Initial project structure with uv configuration
- Centralized configuration with Python 3.12 support

### Changed
- **UI/UX overhaul** — Sidebar file upload with metrics and column info; suggestion cards above chat; two-column chat + chart layout; theme-safe CSS (no hardcoded colors)
- **CSS padding** — Increased `.block-container padding-top` from `1.5rem` to `3rem` to prevent header text cropping
- **System prompt improvements** — Explicit rule to always return empty `transformations` array; clearer JSON-only output instructions; better aggregation guidance
- **Column validation relaxed** — Switched from character blocklist to injection-pattern blocklist in `spec_schema.py` so column names with `()`, `[]`, `$` are accepted
- **Retry logic reworked** — `openai_client.py` now lets `RateLimitError`/`TimeoutError` propagate to tenacity for actual retries instead of wrapping them prematurely

### Fixed
- **`chart_type.value` AttributeError** — Safe `hasattr` checks across Plotly and Seaborn renderers when Pydantic v2 stores enums as plain strings
- **White/invisible suggestion card text** — Removed hardcoded light-theme CSS colors that were invisible in dark Streamlit themes
- **`.env` inline comment** — Removed inline comment on `MEMORY_BACKEND` that could break dotenv parsing
- **JSON extraction failures on complex queries** — Replaced single-level nesting regex with a proper brace-counting parser, fixing "Could not extract valid JSON" errors on aggregated/computed-column queries
- **Invalid transformation crash** — LLM-hallucinated operations like `drop_duplicates` are now stripped before validation

### Removed

## [0.1.0] - 2026-02-07

### Added
- Structured logging with structlog and sensitive data censorship
- Custom exception hierarchy for error handling
- Data ingestion service with CSV/XLSX parsing and security validation
- Data profiling service with auto-recommendations
- OpenAI client with retry logic and rate limiting
- Visualization spec planner with LLM-based intent extraction
- Plotly and Seaborn renderers with safe transformation operations
- SQL memory service with append-only chat history
- Vector memory service with ChromaDB adapter
- Streamlit frontend with interactive chat and visualization components
- Comprehensive test suite with pytest coverage
- Docker configuration with multi-stage builds
- GitHub Actions CI/CD pipeline with security scanning
- Development scripts for running app and generating sample data
- Complete documentation in first-person (README, ARCHITECTURE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, SETUP)
- Git configuration with standard Python ignores

### Fixed
- Hatchling package configuration to enable wheel building
- SQLAlchemy reserved name conflict in Message model
- OpenAI API parameter compatibility for gpt-5 and gpt-4o models

