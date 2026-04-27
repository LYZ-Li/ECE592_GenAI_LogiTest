"""Model backend Protocol and local HF engine implementation."""

from __future__ import annotations

import gc
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Protocol
from urllib.parse import urlparse

from src.common.contracts import ModelResponse

if TYPE_CHECKING:
    from src.runner.config import ModelConfig

logger = logging.getLogger("memory_compression.engine")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_LOCAL_API_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _strip_think_tags(text: str) -> str:
    """Remove Qwen3-style <think>…</think> reasoning blocks from output.

    Handles both closed tags and truncated (unclosed) blocks where the
    model ran out of tokens mid-thought.
    """
    text = _THINK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text


def _is_local_api_url(api_base_url: str | None) -> bool:
    """Return whether an API URL points at a local OpenAI-compatible server."""
    if not api_base_url:
        return False
    hostname = urlparse(api_base_url).hostname
    return hostname in _LOCAL_API_HOSTS


class ModelBackend(Protocol):
    """Interface for model inference backends."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelResponse: ...

    def clear_memory(self) -> None: ...


@dataclass
class TransformersQwenBackend:
    """Local Hugging Face backend with configurable quantization.

    Supports 4-bit (BitsAndBytes), 8-bit, or float16 inference.

    Args:
        model_name_or_path: HF model identifier or local path.
        tokenizer_name_or_path: Optional separate tokenizer path.
        quantization: One of "4bit", "8bit", or "none".
        device_map: Device placement strategy.
        trust_remote_code: Whether to allow remote code execution.
        enable_thinking: Whether to enable Qwen3 thinking mode in chat template.
    """

    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    quantization: str = "none"
    device_map: str = "auto"
    trust_remote_code: bool = False
    enable_thinking: bool = False
    _pipeline: object | None = field(default=None, init=False, repr=False)
    _tokenizer: object | None = field(default=None, init=False, repr=False)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> ModelResponse:
        """Generate a model response given system and user prompts."""
        pipeline, tokenizer = self._load_pipeline()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        outputs = pipeline(
            prompt_text,
            max_new_tokens=max_new_tokens,
            max_length=None,
            temperature=temperature,
            do_sample=temperature > 0.0,
            return_full_text=False,
        )
        raw_output = outputs[0]["generated_text"]
        generated_text = _strip_think_tags(raw_output).strip()
        prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        completion_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
        return ModelResponse(
            text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_output=raw_output,
            metadata={
                "backend": "transformers_qwen",
                "model_name_or_path": self.model_name_or_path,
            },
        )

    def clear_memory(self) -> None:
        """Release GPU memory between trials."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

    def _load_pipeline(self):
        if self._pipeline is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "TransformersQwenBackend requires `transformers` and `torch`. "
                    "Install dependencies before running model inference."
                ) from exc

            tokenizer_name = self.tokenizer_name_or_path or self.model_name_or_path
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=self.trust_remote_code,
            )

            # Fix Q3: apply quantization from config instead of hardcoding float16
            quant_config = None
            load_kwargs: dict = {
                "trust_remote_code": self.trust_remote_code,
                "device_map": self.device_map,
            }

            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = quant_config
            elif self.quantization == "8bit":
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["quantization_config"] = quant_config
            else:
                load_kwargs["torch_dtype"] = (
                    torch.float16 if torch.cuda.is_available() else None
                )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, **load_kwargs
            )
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
            )

            logger.info(
                "Model loaded: %s (quantization=%s)",
                self.model_name_or_path,
                self.quantization,
            )
        return self._pipeline, self._tokenizer


def build_model_backend(config: ModelConfig) -> ModelBackend:
    """Create a model backend from config.

    Args:
        config: Model configuration with backend_type selection.

    Returns:
        A configured ModelBackend instance.

    Raises:
        RuntimeError: If required environment variables are missing.
        ValueError: If backend_type is unsupported.
    """
    if config.backend_type == "anthropic_api":
        from src.runner.api_backend import ClaudeAPIBackend

        token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if not token:
            raise RuntimeError(
                "ANTHROPIC_AUTH_TOKEN environment variable is required "
                "for the anthropic_api backend."
            )
        return ClaudeAPIBackend(
            api_base_url=config.api_base_url or "https://cc.580ai.net",
            auth_token=token,
            model=config.api_model or "claude-sonnet-4-20250514",
        )

    if config.backend_type == "openai_compatible":
        from src.runner.openai_backend import OpenAICompatibleBackend

        api_base_url = config.api_base_url or "https://openrouter.ai/api"
        api_key_env = config.api_key_env or "OPENAI_API_KEY"
        token = os.environ.get(api_key_env)
        if not token and _is_local_api_url(api_base_url):
            token = "local-openai-compatible"
        if not token:
            raise RuntimeError(
                f"{api_key_env} environment variable is required "
                "for the openai_compatible backend. Set model.api_key_env "
                "to use a provider-specific key variable."
            )
        return OpenAICompatibleBackend(
            api_base_url=api_base_url,
            auth_token=token,
            model=config.api_model or "Qwen/Qwen3-8B",
        )

    if config.backend_type == "transformers":
        return TransformersQwenBackend(
            model_name_or_path=config.name_or_path,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
            quantization=config.quantization,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
            enable_thinking=config.enable_thinking,
        )

    raise ValueError(
        f"Unsupported backend_type '{config.backend_type}'. "
        "Supported: transformers, anthropic_api, openai_compatible."
    )
