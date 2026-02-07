"""
OpenAI API client wrapper with retry logic, rate limiting, and usage tracking.
Never logs API keys; implements backoff and defensive error handling.
"""

import time
from typing import Any

from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import get_settings
from core.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from core.logging import get_logger

logger = get_logger(__name__)


class UsageStats(BaseModel):
    """Token usage statistics for an LLM call."""

    prompt_tokens: int = Field(description="Tokens in prompt")
    completion_tokens: int = Field(description="Tokens in completion")
    total_tokens: int = Field(description="Total tokens used")
    model: str = Field(description="Model name")
    duration_seconds: float = Field(description="Request duration")


class OpenAIClient:
    """
    Wrapper for OpenAI API with retry logic and usage tracking.
    Implements defensive error handling and rate limiting.
    """

    def __init__(self) -> None:
        """Initialize OpenAI client with configuration."""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_model
        self.max_tokens = self.settings.openai_max_tokens
        self.temperature = self.settings.openai_temperature

        # Usage tracking
        self.total_calls = 0
        self.total_tokens = 0
        self.last_call_time: float | None = None

        logger.info(
            "OpenAI client initialized",
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _check_rate_limit(self) -> None:
        """
        Basic rate limiting check.
        Prevents excessive API calls.
        """
        if self.last_call_time is not None:
            elapsed = time.time() - self.last_call_time
            # Enforce minimum delay between calls (1 second)
            if elapsed < 1.0:
                sleep_time = 1.0 - elapsed
                logger.debug("Rate limit delay", sleep_seconds=sleep_time)
                time.sleep(sleep_time)

    @retry(
        retry=retry_if_exception_type((RateLimitError, TimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_with_retry(
        self, messages: list[dict[str, str]], temperature: float | None = None
    ) -> dict[str, Any]:
        """
        Call OpenAI API with retry logic.

        Args:
            messages: List of message dictionaries
            temperature: Optional temperature override

        Returns:
            API response dictionary

        Raises:
            LLMError: If API call fails after retries
        """
        start_time = time.time()

        try:
            # Build kwargs based on model capabilities
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            
            # gpt-5 models:
            # - Only support temperature=1 (default), so don't send it
            # - Require max_completion_tokens instead of max_tokens
            if "gpt-5" in self.model:
                kwargs["max_completion_tokens"] = self.max_tokens
                # Don't set temperature for gpt-5 (only default value of 1 is supported)
                logger.debug("Using gpt-5 with default temperature=1")
            
            # gpt-4o models require max_completion_tokens but support temperature
            elif "gpt-4o" in self.model:
                kwargs["max_completion_tokens"] = self.max_tokens
                kwargs["temperature"] = temperature or self.temperature
            
            # Older models (gpt-4, gpt-3.5-turbo, etc.) use max_tokens
            else:
                kwargs["max_tokens"] = self.max_tokens
                kwargs["temperature"] = temperature or self.temperature
            
            response = self.client.chat.completions.create(**kwargs)  # type: ignore

            duration = time.time() - start_time

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            usage = response.usage

            # Track usage
            if usage:
                self.total_tokens += usage.total_tokens
                self.total_calls += 1

                logger.info(
                    "LLM call succeeded",
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    duration_seconds=round(duration, 2),
                )

            return {
                "content": content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
                "duration_seconds": duration,
            }

        except RateLimitError as e:
            logger.warning("Rate limit exceeded, retrying", error=str(e))
            raise LLMRateLimitError("OpenAI rate limit exceeded. Retrying...") from e

        except TimeoutError as e:
            logger.warning("Request timeout, retrying", error=str(e))
            raise LLMTimeoutError("OpenAI request timed out. Retrying...") from e

        except Exception as e:
            logger.error("LLM call failed", error=str(e), error_type=type(e).__name__)
            raise LLMError(f"OpenAI API call failed: {str(e)}") from e

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> tuple[str, UsageStats]:
        """
        Generate completion for a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for instructions
            temperature: Optional temperature override

        Returns:
            Tuple of (completion_text, usage_stats)

        Raises:
            LLMError: If API call fails
        """
        self._check_rate_limit()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call API
        response = self._call_with_retry(messages, temperature)
        self.last_call_time = time.time()

        # Build usage stats
        usage_stats = UsageStats(
            prompt_tokens=response["usage"]["prompt_tokens"],
            completion_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
            model=self.model,
            duration_seconds=response["duration_seconds"],
        )

        return response["content"], usage_stats

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> tuple[str, UsageStats]:
        """
        Multi-turn chat with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override

        Returns:
            Tuple of (completion_text, usage_stats)

        Raises:
            LLMError: If API call fails
        """
        self._check_rate_limit()

        # Call API
        response = self._call_with_retry(messages, temperature)
        self.last_call_time = time.time()

        # Build usage stats
        usage_stats = UsageStats(
            prompt_tokens=response["usage"]["prompt_tokens"],
            completion_tokens=response["usage"]["completion_tokens"],
            total_tokens=response["usage"]["total_tokens"],
            model=self.model,
            duration_seconds=response["duration_seconds"],
        )

        return response["content"], usage_stats

    def get_usage_summary(self) -> dict[str, Any]:
        """
        Get summary of API usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "model": self.model,
        }
