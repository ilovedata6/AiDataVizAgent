"""
Core configuration module using Pydantic Settings.
Loads configuration from environment variables and .env files.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All sensitive data must be provided via .env or host secrets.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key (required)")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    openai_max_tokens: int = Field(default=2000, description="Max tokens per request")
    openai_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # Application Configuration
    app_env: Literal["development", "production", "test"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    debug: bool = Field(default=False)

    # File Upload Configuration
    max_upload_size_mb: int = Field(default=50, ge=1, le=500)
    allowed_extensions: str = Field(default="csv,xlsx")

    # Memory Configuration
    memory_backend: Literal["sqlite", "vector"] = Field(default="sqlite")
    sqlite_db_path: str = Field(default="./data/memory.db")
    vector_store_path: str = Field(default="./data/vector_store")

    # Session Configuration
    session_timeout_minutes: int = Field(default=60, ge=5, le=1440)
    max_chat_history: int = Field(default=50, ge=10, le=500)

    # Rate Limiting
    max_llm_calls_per_minute: int = Field(default=30, ge=1, le=100)
    llm_retry_attempts: int = Field(default=3, ge=1, le=10)
    llm_retry_delay_seconds: float = Field(default=2.0, ge=0.5, le=60.0)

    # Observability (Optional)
    sentry_dsn: str = Field(default="")
    enable_metrics: bool = Field(default=False)
    metrics_port: int = Field(default=9090, ge=1024, le=65535)

    # Security
    cors_origins: str = Field(default="*")
    sanitize_filenames: bool = Field(default=True)

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is not a placeholder."""
        if not v or v.startswith("your_") or v == "":
            raise ValueError("OpenAI API key must be set in .env file")
        return v

    @property
    def allowed_extensions_list(self) -> list[str]:
        """Return allowed extensions as a list."""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        """Return max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_data_directories(self) -> None:
        """Create necessary data directories if they don't exist."""
        Path(self.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    Creates and caches the instance on first call.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_data_directories()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload settings from environment.
    Useful for testing.
    """
    global _settings
    _settings = None
    return get_settings()
