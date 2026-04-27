"""Anthropic Messages API backend for the memory compression study."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import httpx

from src.common.contracts import ModelResponse

logger = logging.getLogger("memory_compression.api_backend")

_MAX_RETRIES = 1
_RETRY_BACKOFF_SECONDS = 2.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}


@dataclass
class ClaudeAPIBackend:
    """Anthropic Messages API backend via httpx.

    Args:
        api_base_url: Base URL for the API (e.g. "https://cc.580ai.net").
        auth_token: API authentication token.
        model: Model identifier (e.g. "claude-sonnet-4-20250514").
    """

    api_base_url: str
    auth_token: str
    model: str = "claude-sonnet-4-20250514"
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelResponse:
        """Generate a response via the Anthropic Messages API.

        Args:
            system_prompt: System-level instruction.
            user_prompt: User message content.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Populated ModelResponse.

        Raises:
            RuntimeError: On non-retryable API errors or exhausted retries.
        """
        client = self._get_client()
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        response = self._request_with_retry(client, payload)
        body = response.json()

        text = body["content"][0]["text"]
        usage = body.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_output=text,
            metadata={
                "backend": "anthropic_api",
                "model": self.model,
                "stop_reason": body.get("stop_reason"),
            },
        )

    def clear_memory(self) -> None:
        """No-op — API backend is stateless."""

    def _get_client(self) -> httpx.Client:
        """Lazily create the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.api_base_url,
                headers={
                    "x-api-key": self.auth_token,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=httpx.Timeout(timeout=120.0),
            )
        return self._client

    def _request_with_retry(
        self, client: httpx.Client, payload: Dict[str, Any]
    ) -> httpx.Response:
        """POST with basic retry on transient errors."""
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = client.post("/v1/messages", json=payload)
                if response.status_code == 200:
                    return response
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise RuntimeError(
                        f"API error {response.status_code}: {response.text[:500]}"
                    )
                last_error = RuntimeError(
                    f"API error {response.status_code}: {response.text[:500]}"
                )
            except httpx.TransportError as exc:
                last_error = exc

            if attempt < _MAX_RETRIES:
                logger.warning(
                    "Retrying API request (attempt %d/%d): %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    last_error,
                )
                time.sleep(_RETRY_BACKOFF_SECONDS)

        raise RuntimeError(
            f"API request failed after {_MAX_RETRIES + 1} attempts: {last_error}"
        )
