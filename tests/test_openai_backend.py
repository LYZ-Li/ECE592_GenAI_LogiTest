"""Tests for src.runner.openai_backend and factory integration."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.runner.config import ModelConfig
from src.runner.engine import build_model_backend
from src.runner.openai_backend import (
    OpenAICompatibleBackend,
    _chat_completions_path,
    _strip_think_tags,
)


def _mock_response(status_code: int = 200, body: dict | None = None) -> MagicMock:
    """Create a mock httpx.Response with OpenAI-format body."""
    resp = MagicMock()
    resp.status_code = status_code
    if body is None:
        body = {
            "choices": [
                {
                    "message": {
                        "content": '{"action_name": "move", "arguments": ["robot_1", "room_1", "room_2"]}'
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25},
        }
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


class TestOpenAICompatibleBackend:
    def test_generate_returns_model_response(self) -> None:
        backend = OpenAICompatibleBackend(
            api_base_url="https://openrouter.ai/api",
            auth_token="test-token",
            model="Qwen/Qwen3-8B",
        )
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response()
        backend._client = mock_client

        result = backend.generate(
            system_prompt="You are a planner.",
            user_prompt="Pick up pkg_1.",
            max_new_tokens=256,
            temperature=0.0,
        )

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 25
        assert "move" in result.text
        assert result.metadata["backend"] == "openai_compatible"
        assert result.metadata["finish_reason"] == "stop"

        # Verify the request payload uses OpenAI format
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/v1/chat/completions"
        payload = call_args[1]["json"]
        assert payload["model"] == "Qwen/Qwen3-8B"
        assert payload["max_tokens"] == 256
        assert payload["temperature"] == 0.0
        assert payload["messages"] == [
            {"role": "system", "content": "You are a planner."},
            {"role": "user", "content": "Pick up pkg_1."},
        ]

    def test_generate_raises_on_non_retryable_error(self) -> None:
        backend = OpenAICompatibleBackend(
            api_base_url="https://openrouter.ai/api",
            auth_token="test-token",
            model="test-model",
        )
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            status_code=400,
            body={"error": {"message": "bad request"}},
        )
        backend._client = mock_client

        with pytest.raises(RuntimeError, match="API error 400"):
            backend.generate("sys", "user", 256, 0.0)

        assert mock_client.post.call_count == 1

    def test_generate_retries_on_429(self) -> None:
        backend = OpenAICompatibleBackend(
            api_base_url="https://openrouter.ai/api",
            auth_token="test-token",
            model="test-model",
        )
        mock_client = MagicMock()
        rate_limit_resp = _mock_response(
            status_code=429,
            body={"error": {"message": "rate limited"}},
        )
        ok_resp = _mock_response()
        mock_client.post.side_effect = [rate_limit_resp, ok_resp]
        backend._client = mock_client

        with patch("src.runner.openai_backend.time.sleep"):
            result = backend.generate("sys", "user", 256, 0.0)

        assert result.prompt_tokens == 100
        assert mock_client.post.call_count == 2

    def test_generate_strips_think_tags(self) -> None:
        """Thinking tokens should be stripped from text but preserved in raw_output."""
        backend = OpenAICompatibleBackend(
            api_base_url="http://localhost:8000",
            auth_token="test-token",
            model="gemma-4-26B-A4B-it",
        )
        thinking_output = (
            '<think>The robot needs to move from room_1 to room_2.</think>'
            '{"action_name": "move", "arguments": ["robot_1", "room_1", "room_2"]}'
        )
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            body={
                "choices": [{"message": {"content": thinking_output}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
        )
        backend._client = mock_client

        result = backend.generate("sys", "user", 2048, 0.0)
        assert result.text == '{"action_name": "move", "arguments": ["robot_1", "room_1", "room_2"]}'
        assert "<think>" in result.raw_output

    def test_generate_handles_unclosed_think_tag(self) -> None:
        """Truncated thinking output (token limit hit mid-thought) should yield empty text."""
        backend = OpenAICompatibleBackend(
            api_base_url="http://localhost:8000",
            auth_token="test-token",
            model="gemma-4-26B-A4B-it",
        )
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            body={
                "choices": [{"message": {"content": "<think>Still thinking about this..."}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 512},
            }
        )
        backend._client = mock_client

        result = backend.generate("sys", "user", 512, 0.0)
        assert result.text == ""
        assert "<think>" in result.raw_output

    def test_generate_handles_none_content(self) -> None:
        """Some APIs return null content — should not crash."""
        backend = OpenAICompatibleBackend(
            api_base_url="http://localhost:8000",
            auth_token="test-token",
            model="test-model",
        )
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_response(
            body={
                "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 0},
            }
        )
        backend._client = mock_client

        result = backend.generate("sys", "user", 256, 0.0)
        assert result.text == ""

    def test_clear_memory_is_noop(self) -> None:
        backend = OpenAICompatibleBackend(
            api_base_url="https://openrouter.ai/api",
            auth_token="test-token",
            model="test-model",
        )
        backend.clear_memory()  # should not raise


class TestStripThinkTags:
    def test_closed_think_tag(self) -> None:
        assert _strip_think_tags("<think>reasoning</think>answer") == "answer"

    def test_unclosed_think_tag(self) -> None:
        assert _strip_think_tags("<think>still going...") == ""

    def test_no_think_tags(self) -> None:
        assert _strip_think_tags("plain text") == "plain text"

    def test_multiple_think_blocks(self) -> None:
        text = "<think>a</think>first<think>b</think>second"
        assert _strip_think_tags(text) == "firstsecond"


class TestChatCompletionsPath:
    def test_base_without_v1_uses_absolute_v1_path(self) -> None:
        assert _chat_completions_path("https://openrouter.ai/api") == "/v1/chat/completions"

    def test_base_with_v1_uses_relative_path(self) -> None:
        assert (
            _chat_completions_path(
                "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            )
            == "chat/completions"
        )


class TestBuildModelBackendOpenAI:
    def test_openai_compatible_with_env_var(self) -> None:
        config = ModelConfig(
            backend_type="openai_compatible",
            api_base_url="http://localhost:8000",
            api_model="Qwen/Qwen3-8B",
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = build_model_backend(config)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.auth_token == "sk-test"
        assert backend.api_base_url == "http://localhost:8000"
        assert backend.model == "Qwen/Qwen3-8B"

    def test_openai_compatible_missing_env_var_raises(self) -> None:
        config = ModelConfig(backend_type="openai_compatible")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                build_model_backend(config)

    @pytest.mark.parametrize(
        "api_base_url",
        [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://[::1]:8000",
        ],
    )
    def test_openai_compatible_local_server_without_env_var(
        self, api_base_url: str
    ) -> None:
        config = ModelConfig(
            backend_type="openai_compatible",
            api_base_url=api_base_url,
            api_model="local-model",
        )
        with patch.dict(os.environ, {}, clear=True):
            backend = build_model_backend(config)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.auth_token == "local-openai-compatible"
        assert backend.api_base_url == api_base_url
        assert backend.model == "local-model"

    def test_openai_compatible_defaults(self) -> None:
        config = ModelConfig(backend_type="openai_compatible")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            backend = build_model_backend(config)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.api_base_url == "https://openrouter.ai/api"
        assert backend.model == "Qwen/Qwen3-8B"

    def test_openai_compatible_custom_api_key_env(self) -> None:
        config = ModelConfig(
            backend_type="openai_compatible",
            api_base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
            api_model="qwen3.6-plus",
            api_key_env="ALIYUN_API_KEY",
        )
        with patch.dict(os.environ, {"ALIYUN_API_KEY": "aliyun-test"}, clear=True):
            backend = build_model_backend(config)

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.auth_token == "aliyun-test"
        assert backend.api_base_url.endswith("/compatible-mode/v1")
        assert backend.model == "qwen3.6-plus"

    def test_openai_compatible_missing_custom_env_var_raises(self) -> None:
        config = ModelConfig(
            backend_type="openai_compatible",
            api_key_env="ALIYUN_API_KEY",
        )
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="ALIYUN_API_KEY"):
                build_model_backend(config)
