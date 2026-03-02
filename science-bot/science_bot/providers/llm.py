"""Thin OpenAI wrapper for structured LLM responses."""

import os
from typing import TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
T = TypeVar("T", bound=BaseModel)


class LLMProviderError(Exception):
    """Base exception for provider-layer failures."""


class LLMConfigurationError(LLMProviderError):
    """Raised when the LLM client is not properly configured."""


class LLMResponseFormatError(LLMProviderError):
    """Raised when a structured LLM response is missing or invalid."""


def parse_structured(
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> T:
    """Request and parse a structured response into a Pydantic model.

    Args:
        system_prompt: Instruction text for the model.
        user_prompt: User input text to process.
        response_model: Pydantic model class to parse into.
        model: Optional model override.
        timeout_seconds: Request timeout in seconds.
        max_retries: SDK retry count for transient failures.

    Returns:
        T: Parsed structured response.

    Raises:
        LLMProviderError: If the provider call fails.
        LLMResponseFormatError: If the parsed response is missing or invalid.
    """
    resolved_model = model if model is not None else DEFAULT_OPENAI_MODEL
    if not resolved_model.strip():
        raise LLMConfigurationError("OpenAI model name must be non-empty.")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None or not api_key.strip():
        raise LLMConfigurationError("OPENAI_API_KEY is required.")
    if timeout_seconds <= 0:
        raise LLMConfigurationError("timeout_seconds must be greater than zero.")
    if max_retries < 0:
        raise LLMConfigurationError("max_retries must be zero or greater.")

    client = OpenAI(
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )

    try:
        response = client.responses.parse(
            model=resolved_model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=response_model,
        )
    except OpenAIError as exc:
        raise LLMProviderError(
            f"OpenAI structured response request failed: {exc}"
        ) from exc

    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        raise LLMResponseFormatError(
            "Structured response did not include parsed output."
        )
    if not isinstance(parsed, response_model):
        raise LLMResponseFormatError(
            "Structured response did not match the requested response model."
        )
    return parsed
