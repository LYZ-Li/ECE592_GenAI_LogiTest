"""Memory policy abstractions for the memory compression study.

Implements three conditions:
  1. FullContextPolicy — no compression
  2. RecentWindowPolicy — tail truncation (Condition 2)
  3. SummarizationPolicy — compressed summary + recent window (Condition 3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Protocol, Tuple

from src.common.contracts import MemoryContext, MemoryRecord

logger = logging.getLogger("memory_compression.memory")


class TokenCounter(Protocol):
    """Count prompt tokens with the same tokenizer family as the model backend."""

    def count_text(self, text: str) -> int: ...


@dataclass
class TransformersTokenCounter:
    """Lazy `transformers` tokenizer wrapper used by memory policies."""

    model_name_or_path: str
    _tokenizer: object | None = field(default=None, init=False, repr=False)

    def count_text(self, text: str) -> int:
        tokenizer = self._load_tokenizer()
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _load_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "TransformersTokenCounter requires `transformers`."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        return self._tokenizer


class MemoryPolicy(Protocol):
    name: str

    def prepare_context(
        self,
        history: List[MemoryRecord],
        query: str,
    ) -> MemoryContext: ...


@dataclass
class FullContextPolicy:
    """Condition 1: pass all history records uncompressed."""

    name: str = "full_context"

    def prepare_context(
        self, history: List[MemoryRecord], query: str
    ) -> MemoryContext:
        return MemoryContext(
            policy_name=self.name,
            prompt_context="\n\n".join(record.text for record in history),
            source_records=list(history),
            metadata={
                "query": query,
                "record_count": len(history),
                "compressed": False,
            },
        )


@dataclass
class RecentWindowPolicy:
    """Condition 2: keep only the most recent records within a token budget."""

    max_context_tokens: int
    token_counter: TokenCounter
    name: str = "recent_window"
    _token_cache: dict = field(default_factory=dict, init=False, repr=False)

    def prepare_context(
        self, history: List[MemoryRecord], query: str
    ) -> MemoryContext:
        selected: List[MemoryRecord] = []
        running_tokens = 0
        for record in reversed(history):
            token_count = self._count(record)
            if selected and running_tokens + token_count > self.max_context_tokens:
                break
            selected.append(
                MemoryRecord(
                    step_index=record.step_index,
                    text=record.text,
                    tags=list(record.tags),
                    token_count=token_count,
                )
            )
            running_tokens += token_count
            if running_tokens >= self.max_context_tokens:
                break

        selected.reverse()
        return MemoryContext(
            policy_name=self.name,
            prompt_context="\n\n".join(record.text for record in selected),
            source_records=selected,
            metadata={
                "query": query,
                "max_context_tokens": self.max_context_tokens,
                "token_count": running_tokens,
                "record_count": len(selected),
                "compressed": len(selected) < len(history),
            },
        )

    def _count(self, record: MemoryRecord) -> int:
        key = id(record)
        if key not in self._token_cache:
            self._token_cache[key] = (
                record.token_count
                if record.token_count > 0
                else self.token_counter.count_text(record.text)
            )
        return self._token_cache[key]


@dataclass
class SummarizationPolicy:
    """Condition 3: compressed summary of old context + verbatim recent window.

    Two modes controlled by summary_model config:
      - Extractive (summary_model=None): uniform subsampling of old records.
        Keeps first record (initial state context) + evenly spaced records
        from the middle, up to summary_budget tokens.
      - Abstractive (summary_model set): calls a separate HF model to compress
        old records into a summary string. Falls back to extractive on failure.
    """

    max_context_tokens: int
    recent_window_tokens: int
    token_counter: TokenCounter
    summary_model: str | None = None
    name: str = "summarization"
    _summarizer: object | None = field(default=None, init=False, repr=False)
    _token_cache: dict = field(default_factory=dict, init=False, repr=False)

    def prepare_context(
        self, history: List[MemoryRecord], query: str
    ) -> MemoryContext:
        if not history:
            return MemoryContext(
                policy_name=self.name,
                prompt_context="",
                source_records=[],
                metadata={"compressed": False},
            )

        # 1. Split into recent window and older records
        recent, old = self._split_window(history)

        # 2. Compress the old section
        summary_budget = self.max_context_tokens - sum(
            self._count(r) for r in recent
        )
        if summary_budget <= 0:
            compressed_text = ""
            old_selected: List[MemoryRecord] = []
        elif self.summary_model is not None:
            compressed_text, old_selected = self._abstractive(old, summary_budget)
        else:
            compressed_text, old_selected = self._extractive(old, summary_budget)

        # 3. Assemble final context
        parts = []
        if compressed_text:
            parts.append("[Summary of earlier steps]\n" + compressed_text)
        if recent:
            parts.append("\n\n".join(r.text for r in recent))
        prompt_context = "\n\n".join(parts)

        return MemoryContext(
            policy_name=self.name,
            prompt_context=prompt_context,
            source_records=old_selected + recent,
            metadata={
                "compressed": len(old) > len(old_selected),
                "summary_budget": summary_budget,
                "recent_count": len(recent),
                "old_count": len(old),
                "old_selected_count": len(old_selected),
                "method": "abstractive" if self.summary_model else "extractive",
                "query": query,
            },
        )

    def _split_window(
        self, history: List[MemoryRecord]
    ) -> Tuple[List[MemoryRecord], List[MemoryRecord]]:
        """Split history into (recent_tail, older_prefix)."""
        recent: List[MemoryRecord] = []
        budget = self.recent_window_tokens
        for record in reversed(history):
            cost = self._count(record)
            if recent and budget - cost < 0:
                break
            recent.append(record)
            budget -= cost
        recent.reverse()
        cutoff = len(history) - len(recent)
        return recent, history[:cutoff]

    def _extractive(
        self, old: List[MemoryRecord], budget: int
    ) -> Tuple[str, List[MemoryRecord]]:
        """Keep first record + uniformly spaced sample within budget."""
        if not old:
            return "", []
        selected = [old[0]]  # always keep first (initial state context)
        remaining_budget = budget - self._count(old[0])

        if len(old) > 1 and remaining_budget > 0:
            candidates = old[1:]
            total_cost = sum(self._count(r) for r in candidates)
            avg_cost = total_cost / len(candidates) if candidates else 1
            n_to_keep = max(1, int(remaining_budget / avg_cost))
            if n_to_keep >= len(candidates):
                selected.extend(candidates)
            else:
                step = len(candidates) / n_to_keep
                indices = [int(i * step) for i in range(n_to_keep)]
                selected.extend(candidates[i] for i in indices)

        text = "\n".join(
            f"* [Step {r.step_index}] {r.text}" for r in selected
        )
        return text, selected

    def _abstractive(
        self, old: List[MemoryRecord], budget: int
    ) -> Tuple[str, List[MemoryRecord]]:
        """Summarize old records via a separate model. Fallback to extractive."""
        try:
            summarizer = self._load_summarizer()
            old_text = "\n\n".join(r.text for r in old)
            prompt = (
                f"Summarize the following planning trace in under {budget} tokens. "
                f"Preserve all precondition information and goal progress:\n\n{old_text}"
            )
            result = summarizer(prompt, max_new_tokens=budget)
            summary_text = result[0]["generated_text"] if isinstance(result, list) else str(result)
            return summary_text, list(old)
        except Exception:
            logger.warning(
                "Abstractive summarization failed; falling back to extractive"
            )
            return self._extractive(old, budget)

    def _count(self, record: MemoryRecord) -> int:
        key = id(record)
        if key not in self._token_cache:
            self._token_cache[key] = (
                record.token_count
                if record.token_count > 0
                else self.token_counter.count_text(record.text)
            )
        return self._token_cache[key]

    def _load_summarizer(self):
        if self._summarizer is None:
            from transformers import BitsAndBytesConfig, pipeline
            import torch

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self._summarizer = pipeline(
                "text-generation",
                model=self.summary_model,
                quantization_config=quant,
                device_map="auto",
            )
        return self._summarizer
