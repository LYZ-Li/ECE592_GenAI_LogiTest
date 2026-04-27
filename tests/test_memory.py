"""Tests for src.runner.memory."""

from src.common.contracts import MemoryRecord
from src.runner.memory import (
    FullContextPolicy,
    RecentWindowPolicy,
    SummarizationPolicy,
)


class FakeTokenCounter:
    """Counts tokens as number of whitespace-separated words (for testing)."""

    def count_text(self, text: str) -> int:
        return len(text.split())


def _make_records(n: int, words_per_record: int = 10) -> list[MemoryRecord]:
    """Create n MemoryRecords with token_count=0 (forces counter fallback)."""
    return [
        MemoryRecord(
            step_index=i,
            text=" ".join(f"word{j}" for j in range(words_per_record)),
            token_count=0,
        )
        for i in range(n)
    ]


class TestFullContextPolicy:
    def test_returns_all_records(self) -> None:
        records = _make_records(5)
        policy = FullContextPolicy()
        ctx = policy.prepare_context(records, "goal")
        assert len(ctx.source_records) == 5
        assert ctx.metadata["compressed"] is False

    def test_empty_history(self) -> None:
        policy = FullContextPolicy()
        ctx = policy.prepare_context([], "goal")
        assert ctx.prompt_context == ""
        assert ctx.source_records == []


class TestRecentWindowPolicy:
    def test_truncates_to_budget(self) -> None:
        counter = FakeTokenCounter()
        records = _make_records(10, words_per_record=20)  # 20 tokens each
        policy = RecentWindowPolicy(
            max_context_tokens=50, token_counter=counter
        )
        ctx = policy.prepare_context(records, "goal")
        # Budget of 50 tokens, 20 tokens each -> should keep 2 records
        assert len(ctx.source_records) <= 3
        assert ctx.metadata["compressed"] is True

    def test_keeps_all_when_under_budget(self) -> None:
        counter = FakeTokenCounter()
        records = _make_records(3, words_per_record=5)  # 5 tokens each
        policy = RecentWindowPolicy(
            max_context_tokens=100, token_counter=counter
        )
        ctx = policy.prepare_context(records, "goal")
        assert len(ctx.source_records) == 3
        assert ctx.metadata["compressed"] is False

    def test_keeps_most_recent(self) -> None:
        counter = FakeTokenCounter()
        records = _make_records(5, words_per_record=10)
        policy = RecentWindowPolicy(
            max_context_tokens=25, token_counter=counter
        )
        ctx = policy.prepare_context(records, "goal")
        # Should keep the last 2 records (20 tokens), not the first
        indices = [r.step_index for r in ctx.source_records]
        assert indices[-1] == 4  # last record

    def test_empty_history(self) -> None:
        counter = FakeTokenCounter()
        policy = RecentWindowPolicy(
            max_context_tokens=100, token_counter=counter
        )
        ctx = policy.prepare_context([], "goal")
        assert ctx.source_records == []


class TestSummarizationPolicy:
    def test_extractive_keeps_first_record(self) -> None:
        counter = FakeTokenCounter()
        records = _make_records(10, words_per_record=10)
        policy = SummarizationPolicy(
            max_context_tokens=100,
            recent_window_tokens=30,
            token_counter=counter,
        )
        ctx = policy.prepare_context(records, "goal")
        # First record should always be in old_selected
        selected_indices = [r.step_index for r in ctx.source_records]
        assert 0 in selected_indices

    def test_extractive_compressed_flag(self) -> None:
        counter = FakeTokenCounter()
        records = _make_records(20, words_per_record=10)
        policy = SummarizationPolicy(
            max_context_tokens=80,
            recent_window_tokens=30,
            token_counter=counter,
        )
        ctx = policy.prepare_context(records, "goal")
        assert ctx.metadata["compressed"] is True
        assert ctx.metadata["method"] == "extractive"

    def test_recent_window_preserved(self) -> None:
        counter = FakeTokenCounter()
        records = _make_records(10, words_per_record=10)
        policy = SummarizationPolicy(
            max_context_tokens=200,
            recent_window_tokens=30,
            token_counter=counter,
        )
        ctx = policy.prepare_context(records, "goal")
        # Recent window should include the last ~3 records (30 tokens / 10 each)
        assert ctx.metadata["recent_count"] >= 2

    def test_empty_history(self) -> None:
        counter = FakeTokenCounter()
        policy = SummarizationPolicy(
            max_context_tokens=100,
            recent_window_tokens=30,
            token_counter=counter,
        )
        ctx = policy.prepare_context([], "goal")
        assert ctx.prompt_context == ""
        assert ctx.metadata["compressed"] is False

    def test_short_history_no_compression(self) -> None:
        counter = FakeTokenCounter()
        records = _make_records(2, words_per_record=5)
        policy = SummarizationPolicy(
            max_context_tokens=200,
            recent_window_tokens=50,
            token_counter=counter,
        )
        ctx = policy.prepare_context(records, "goal")
        # Both records fit in recent window, no old records to compress
        assert ctx.metadata["old_count"] == 0

    def test_summary_in_prompt_context(self) -> None:
        counter = FakeTokenCounter()
        records = _make_records(15, words_per_record=10)
        policy = SummarizationPolicy(
            max_context_tokens=80,
            recent_window_tokens=20,
            token_counter=counter,
        )
        ctx = policy.prepare_context(records, "goal")
        if ctx.metadata["old_selected_count"] > 0:
            assert "[Summary of earlier steps]" in ctx.prompt_context
