"""OpenAI-compatible API backend for the memory compression study.

Works with OpenRouter, Together AI, vLLM local servers, and any provider
that implements the OpenAI ``/v1/chat/completions`` endpoint.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict
from urllib.parse import urlparse

import httpx

from src.common.contracts import ModelResponse

logger = logging.getLogger("memory_compression.openai_backend")

_MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 2.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove thinking blocks (e.g. Gemma/Qwen ``<think>`` tags) from output."""
    text = _THINK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


def _chat_completions_path(api_base_url: str) -> str:
    """Return the correct chat completions path for an OpenAI-compatible base."""
    parsed = urlparse(api_base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        return "chat/completions"
    return "/v1/chat/completions"


@dataclass
class OpenAICompatibleBackend:
    """OpenAI-compatible chat completions backend via httpx.

    Args:
        api_base_url: Base URL (e.g. "https://openrouter.ai/api",
                      "http://localhost:8000" for vLLM).
        auth_token: Bearer token for Authorization header.
        model: Model identifier (e.g. "Qwen/Qwen3-8B").
        strip_thinking: Whether to strip ``<think>`` tags from output.
    """

    api_base_url: str
    auth_token: str
    model: str
    strip_thinking: bool = True
    chat_completions_path: str | None = None
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelResponse:
        """Generate a response via OpenAI-compatible chat completions API.

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
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        response = self._request_with_retry(client, payload)
        body = response.json()

        raw_text = body["choices"][0]["message"]["content"] or ""
        text = _strip_think_tags(raw_text) if self.strip_thinking else raw_text
        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_output=raw_text,
            metadata={
                "backend": "openai_compatible",
                "model": self.model,
                "finish_reason": body["choices"][0].get("finish_reason"),
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
                    "Authorization": f"Bearer {self.auth_token}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(timeout=120.0),
            )
        return self._client

    def _request_with_retry(
        self, client: httpx.Client, payload: Dict[str, Any]
    ) -> httpx.Response:
        """POST to /v1/chat/completions with retry on transient errors."""
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = client.post(
                    self.chat_completions_path
                    or _chat_completions_path(self.api_base_url),
                    json=payload,
                )
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
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))

        raise RuntimeError(
            f"API request failed after {_MAX_RETRIES + 1} attempts: {last_error}"
        )
