"""Tests for src.runner.inference."""

import pytest

from src.common.contracts import PlanningTaskInstance
from src.runner.inference import (
    PromptBuilder,
    StrictActionParser,
    build_incident_history_records,
)


def _make_task() -> PlanningTaskInstance:
    return PlanningTaskInstance(
        task_id="test-1",
        domain_name="household_logistics",
        problem_name="test",
        goal_text="Deliver pkg_1 to room_3",
        valid_objects=["robot_1", "pkg_1", "room_1", "room_2", "room_3"],
        typed_objects={
            "robot": ["robot_1"],
            "package": ["pkg_1"],
            "room": ["room_1", "room_2", "room_3"],
        },
        initial_facts=["at(robot_1, room_1)", "at(pkg_1, room_1)", "handempty(robot_1)"],
        goal_facts=["at(pkg_1, room_3)"],
        metadata={
            "action_schemas": {
                "move": ["robot", "room", "room"],
                "pickup": ["robot", "package", "room"],
                "drop": ["robot", "package", "room"],
            }
        },
    )


class TestPromptBuilder:
    def test_system_prompt_direct(self) -> None:
        builder = PromptBuilder()
        prompt = builder.system_prompt("direct_action")
        assert "bare JSON object" in prompt
        assert "nothing else" in prompt

    def test_system_prompt_cot(self) -> None:
        builder = PromptBuilder()
        prompt = builder.system_prompt("cot")
        assert "Think step by step" in prompt
        assert "action_name" in prompt

    def test_system_prompts_differ(self) -> None:
        builder = PromptBuilder()
        assert builder.system_prompt("direct_action") != builder.system_prompt("cot")

    def test_user_prompt_direct(self) -> None:
        builder = PromptBuilder()
        task = _make_task()
        prompt = builder.user_prompt(
            task=task,
            memory_context="",
            prompt_mode="direct_action",
            current_state_facts=["at(robot_1, room_1)"],
        )
        assert "bare JSON only" in prompt
        assert "at(robot_1, room_1)" in prompt

    def test_user_prompt_cot(self) -> None:
        builder = PromptBuilder()
        task = _make_task()
        prompt = builder.user_prompt(
            task=task,
            memory_context="",
            prompt_mode="cot",
            current_state_facts=["at(robot_1, room_1)"],
        )
        assert "Reason step by step" in prompt

    def test_user_prompt_includes_action_schemas(self) -> None:
        builder = PromptBuilder()
        task = _make_task()
        prompt = builder.user_prompt(
            task=task,
            memory_context="",
            prompt_mode="direct_action",
            current_state_facts=[],
        )
        assert "move(" in prompt
        assert "pickup(" in prompt
        assert "drop(" in prompt
        assert "requires:" in prompt
        assert "effects:" in prompt
        assert "Choose only actions whose listed requirements are satisfied" in prompt

    def test_empty_memory_shows_placeholder(self) -> None:
        builder = PromptBuilder()
        task = _make_task()
        prompt = builder.user_prompt(
            task=task,
            memory_context="",
            prompt_mode="direct_action",
            current_state_facts=[],
        )
        assert "[no prior executed steps]" in prompt

    def test_incidents_are_separate_from_executed_history(self) -> None:
        builder = PromptBuilder()
        task = _make_task()
        prompt = builder.user_prompt(
            task=task,
            memory_context="Step 0\nAction: inspect_gripper robot_1 room_1",
            prompt_mode="direct_action",
            current_state_facts=[],
            incident_context="Prior incident: A package slipped during verification.",
        )
        assert "[Background Incident Records]" in prompt
        assert "[Executed History]" in prompt
        assert prompt.index("[Background Incident Records]") < prompt.index("[Executed History]")
        assert "Prior incident: A package slipped during verification." in prompt
        assert "Step 0\nAction: inspect_gripper robot_1 room_1" in prompt

    def test_repair_feedback_is_in_prompt(self) -> None:
        builder = PromptBuilder()
        task = _make_task()
        prompt = builder.user_prompt(
            task=task,
            memory_context="",
            prompt_mode="direct_action",
            current_state_facts=["at(robot_1, room_1)"],
            repair_context="Missing preconditions: package_inspected(pkg_1)",
        )
        assert "[Symbolic Feedback]" in prompt
        assert "Missing preconditions: package_inspected(pkg_1)" in prompt
        assert "legal replacement action" in prompt

    def test_incident_history_records_are_deterministic(self) -> None:
        task = _make_task()
        records1 = build_incident_history_records(task, seed=42, count=10)
        records2 = build_incident_history_records(task, seed=42, count=10)
        assert [r.text for r in records1] == [r.text for r in records2]
        assert len(records1) == 10
        assert all("incident_history" in r.tags for r in records1)


class TestStrictActionParser:
    def setup_method(self) -> None:
        self.parser = StrictActionParser(valid_action_names=["move", "pickup", "drop"])

    def test_parse_valid_direct(self) -> None:
        raw = '{"action_name": "move", "arguments": ["robot_1", "room_1", "room_2"]}'
        step = self.parser.parse(raw, mode="direct_action")
        assert step.action_name == "move"
        assert step.arguments == ["robot_1", "room_1", "room_2"]

    def test_parse_strips_whitespace_q2(self) -> None:
        """Q2 fix: trailing whitespace should not cause failure."""
        raw = '  {"action_name": "move", "arguments": ["robot_1", "room_1", "room_2"]}  \n'
        step = self.parser.parse(raw, mode="direct_action")
        assert step.action_name == "move"

    def test_parse_direct_with_preamble(self) -> None:
        """Direct parser should extract JSON even when the model prepends text."""
        raw = (
            "Sure, here is the next action:\n"
            '{"action_name": "move", "arguments": ["robot_1", "room_1", "room_2"]}'
        )
        step = self.parser.parse(raw, mode="direct_action")
        assert step.action_name == "move"
        assert step.arguments == ["robot_1", "room_1", "room_2"]

    def test_parse_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="No valid JSON action found"):
            self.parser.parse("not json", mode="direct_action")

    def test_parse_unknown_action(self) -> None:
        raw = '{"action_name": "fly", "arguments": ["robot_1"]}'
        with pytest.raises(ValueError, match="Unknown action"):
            self.parser.parse(raw, mode="direct_action")

    def test_parse_extra_keys(self) -> None:
        raw = '{"action_name": "move", "arguments": ["a", "b", "c"], "extra": true}'
        with pytest.raises(ValueError, match="only `action_name` and `arguments`"):
            self.parser.parse(raw, mode="direct_action")

    def test_parse_cot_mode(self) -> None:
        raw = (
            "Let me think about this...\n"
            "The robot is in room_1 and needs to go to room_2.\n"
            '{"action_name": "move", "arguments": ["robot_1", "room_1", "room_2"]}'
        )
        step = self.parser.parse(raw, mode="cot")
        assert step.action_name == "move"
        assert step.arguments == ["robot_1", "room_1", "room_2"]

    def test_parse_cot_finds_last_json(self) -> None:
        """CoT parser should use the last valid JSON line, not the first."""
        raw = (
            'I considered {"action_name": "pickup"} but that is wrong.\n'
            '{"action_name": "move", "arguments": ["robot_1", "room_1", "room_2"]}'
        )
        step = self.parser.parse(raw, mode="cot")
        assert step.action_name == "move"

    def test_parse_cot_no_json_found(self) -> None:
        raw = "I thought about it but could not decide."
        with pytest.raises(ValueError, match="No valid JSON action found"):
            self.parser.parse(raw, mode="cot")

    def test_parse_empty_arguments(self) -> None:
        raw = '{"action_name": "move", "arguments": []}'
        step = self.parser.parse(raw, mode="direct_action")
        assert step.arguments == []
