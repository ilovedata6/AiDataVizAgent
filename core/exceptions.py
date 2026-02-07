"""
Custom exception classes for the AI Data Visualization Agent.
All exceptions include context and are designed for user-friendly error messages.
"""


class AIVizException(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: dict[str, any] | None = None) -> None:
        """
        Initialize exception with message and optional details.

        Args:
            message: Human-readable error message
            details: Additional context for logging/debugging
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error message."""
        return self.message


class ConfigurationError(AIVizException):
    """Raised when configuration is invalid or missing."""

    pass


class FileUploadError(AIVizException):
    """Raised when file upload fails validation."""

    pass


class FileParsingError(AIVizException):
    """Raised when file parsing fails."""

    pass


class DataValidationError(AIVizException):
    """Raised when data validation fails."""

    pass


class LLMError(AIVizException):
    """Raised when LLM API call fails."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""

    pass


class SpecValidationError(AIVizException):
    """Raised when visualization spec validation fails."""

    pass


class RenderError(AIVizException):
    """Raised when chart rendering fails."""

    pass


class MemoryError(AIVizException):
    """Raised when memory operations fail."""

    pass


class SecurityError(AIVizException):
    """Raised when security validation fails."""

    pass


def get_user_friendly_message(exception: Exception) -> str:
    """
    Convert exception to user-friendly message.

    Args:
        exception: The exception to convert

    Returns:
        User-friendly error message
    """
    if isinstance(exception, AIVizException):
        return exception.message

    # Map common exceptions to friendly messages
    error_messages = {
        FileNotFoundError: "The file could not be found. Please check the path and try again.",
        PermissionError: "I don't have permission to access this file.",
        ValueError: "The data contains invalid values. Please check your input.",
        KeyError: "A required field is missing from the data.",
        TypeError: "The data type is incompatible with this operation.",
    }

    for exc_type, message in error_messages.items():
        if isinstance(exception, exc_type):
            return message

    return "An unexpected error occurred. Please try again or contact support."
