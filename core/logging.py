"""
Structured logging configuration using structlog.
Provides JSON-formatted logs with correlation IDs and context.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from core.config import get_settings


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log entries."""
    event_dict["app"] = "ai-data-viz-agent"
    event_dict["env"] = get_settings().app_env
    return event_dict


def censor_sensitive_data(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Censor sensitive data from logs.
    Never log API keys, passwords, or raw file contents.
    """
    sensitive_keys = ["api_key", "password", "token", "secret", "apikey", "openai_api_key"]

    def _censor_dict(d: dict[str, Any]) -> dict[str, Any]:
        """Recursively censor sensitive keys in dictionaries."""
        censored = {}
        for key, value in d.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                censored[key] = "***REDACTED***"
            elif isinstance(value, dict):
                censored[key] = _censor_dict(value)
            elif isinstance(value, list):
                censored[key] = [
                    _censor_dict(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                censored[key] = value
        return censored

    return _censor_dict(event_dict)


def setup_logging() -> None:
    """
    Configure structured logging for the application.
    Uses JSON format for production, console format for development.
    """
    settings = get_settings()

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )

    # Processors for structlog
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_app_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        censor_sensitive_data,
    ]

    # Add exception formatting
    if settings.app_env == "development":
        processors.extend(
            [
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(),
            ]
        )
    else:
        processors.extend(
            [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ]
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)
