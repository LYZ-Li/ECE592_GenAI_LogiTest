"""Tests for src.common.contracts."""

from src.common.contracts import (
    EvaluationBreakdown,
    MemoryRecord,
    PlanStep,
    PlanTrace,
    PlanningTaskInstance,
    TraceEpisode,
    build_memory_records,
)


class TestPlanStep:
    def test_signature(self) -> None:
        step = PlanStep(index=0, action_name="move", arguments=["robot_1", "room_1", "room_2"])
        assert step.signature == "move robot_1 room_1 room_2"

    def test_signature_no_args(self) -> None:
        step = PlanStep(index=0, action_name="wait", arguments=[])
        assert step.signature == "wait"


class TestPlanTrace:
    def test_signatures(self) -> None:
        steps = [
            PlanStep(index=0, action_name="move", arguments=["r", "a", "b"]),
            PlanStep(index=1, action_name="pickup", arguments=["r", "p", "b"]),
        ]
        trace = PlanTrace(steps=steps)
        assert trace.signatures() == ["move r a b", "pickup r p b"]

    def test_empty_trace(self) -> None:
        trace = PlanTrace(steps=[])
        assert trace.signatures() == []


class TestEvaluationBreakdown:
    def test_total_dependency_failures(self) -> None:
        breakdown = EvaluationBreakdown(
            plan_accuracy=0.5,
            exact_match=False,
            precondition_violations=2,
            ordering_violations=1,
            grounding_errors=3,
            valid_plan=False,
            goal_satisfied=False,
        )
        assert breakdown.total_dependency_failures == 6

    def test_zero_failures(self) -> None:
        breakdown = EvaluationBreakdown(
            plan_accuracy=1.0,
            exact_match=True,
            precondition_violations=0,
            ordering_violations=0,
            grounding_errors=0,
            valid_plan=True,
            goal_satisfied=True,
        )
        assert breakdown.total_dependency_failures == 0


class TestBuildMemoryRecords:
    def test_sets_token_count_to_zero(self) -> None:
        """Q1 fix: token_count must be 0 to force BPE fallback in policies."""
        episodes = [
            TraceEpisode(
                step_index=0,
                state_before=["at(r, a)"],
                action_signature="move r a b",
                state_after=["at(r, b)"],
                text="Step 0: moved robot from a to b",
            ),
            TraceEpisode(
                step_index=1,
                state_before=["at(r, b)"],
                action_signature="pickup r p b",
                state_after=["holding(r, p)"],
                text="Step 1: picked up package p in room b",
            ),
        ]
        records = build_memory_records(episodes)
        assert len(records) == 2
        for record in records:
            assert record.token_count == 0, "token_count must be 0 (Q1 fix)"

    def test_preserves_step_indices(self) -> None:
        episodes = [
            TraceEpisode(
                step_index=5,
                state_before=[],
                action_signature="drop r p c",
                state_after=[],
                text="Step 5",
            ),
        ]
        records = build_memory_records(episodes)
        assert records[0].step_index == 5

    def test_preserves_text(self) -> None:
        text = "Step 0\nState before:\nat(r, a)\nAction: move r a b"
        episodes = [
            TraceEpisode(
                step_index=0,
                state_before=["at(r, a)"],
                action_signature="move r a b",
                state_after=["at(r, b)"],
                text=text,
            ),
        ]
        records = build_memory_records(episodes)
        assert records[0].text == text

    def test_empty_episodes(self) -> None:
        assert build_memory_records([]) == []
