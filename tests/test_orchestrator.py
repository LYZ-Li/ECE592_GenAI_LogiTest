"""Tests for src.runner.orchestrator."""

import json
from pathlib import Path
from typing import List

import pytest

from src.common.contracts import ModelResponse, PlanStep, PlanningTaskInstance
from src.eval.check import HardConstraintEvaluator
from src.generator.generate import ACTION_SCHEMAS
from src.runner.inference import PromptBuilder, StrictActionParser
from src.runner.memory import FullContextPolicy
from src.runner.orchestrator import (
    ProposalExperimentRunner,
    _append_jsonl,
    _load_completed_keys,
    _serialize_result,
    build_memory_policy,
    model_output_dir,
)
from src.runner.config import MemoryPolicyConfig, ProposalConfig
from src.runner.planner import (
    OracleTraceBuilder,
    SymbolicPlanExecutor,
)


def _make_task() -> PlanningTaskInstance:
    return PlanningTaskInstance(
        task_id="test-1",
        domain_name="household_logistics",
        problem_name="test",
        goal_text="Deliver pkg_1 to room_2",
        valid_objects=["robot_1", "pkg_1", "room_1", "room_2"],
        typed_objects={
            "robot": ["robot_1"],
            "package": ["pkg_1"],
            "room": ["room_1", "room_2"],
        },
        initial_facts=[
            "at(pkg_1, room_1)",
            "at(robot_1, room_1)",
            "accessible(pkg_1)",
            "connected(room_1, room_2)",
            "connected(room_2, room_1)",
            "gripper_calibrated(robot_1)",
            "gripper_clean(robot_1)",
            "gripper_inspected(robot_1)",
            "handempty(robot_1)",
            "mobility_ready(robot_1)",
            "stable(pkg_1)",
        ],
        goal_facts=["delivery_verified(pkg_1, room_2)"],
        metadata={
            "action_schemas": {
                name: list(args) for name, args in ACTION_SCHEMAS.items()
            }
        },
    )


def _gold_steps() -> List[PlanStep]:
    return [
        PlanStep(index=0, action_name="inspect_package", arguments=["robot_1", "pkg_1", "room_1"]),
        PlanStep(index=1, action_name="pickup", arguments=["robot_1", "pkg_1", "room_1"]),
        PlanStep(index=2, action_name="verify_grasp", arguments=["robot_1", "pkg_1", "room_1"]),
        PlanStep(index=3, action_name="move", arguments=["robot_1", "room_1", "room_2"]),
        PlanStep(index=4, action_name="drop", arguments=["robot_1", "pkg_1", "room_2"]),
        PlanStep(index=5, action_name="verify_delivery", arguments=["robot_1", "pkg_1", "room_2"]),
    ]


def _direct_responses() -> List[str]:
    return [
        '{"action_name": "inspect_package", "arguments": ["robot_1", "pkg_1", "room_1"]}',
        '{"action_name": "pickup", "arguments": ["robot_1", "pkg_1", "room_1"]}',
        '{"action_name": "verify_grasp", "arguments": ["robot_1", "pkg_1", "room_1"]}',
        '{"action_name": "move", "arguments": ["robot_1", "room_1", "room_2"]}',
        '{"action_name": "drop", "arguments": ["robot_1", "pkg_1", "room_2"]}',
        '{"action_name": "verify_delivery", "arguments": ["robot_1", "pkg_1", "room_2"]}',
    ]


class FakePlannerBackend:
    """Returns a fixed plan without needing Fast Downward."""

    def __init__(self, steps: List[PlanStep]) -> None:
        self._steps = steps

    def solve(self, task: PlanningTaskInstance):
        from src.common.contracts import PlanTrace
        return PlanTrace(steps=self._steps)

    def validate(self, task, candidate_steps, gold_steps):
        pass


class FakeModelBackend:
    """Returns a sequence of pre-defined responses."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.user_prompts: List[str] = []

    def generate(self, system_prompt, user_prompt, max_new_tokens, temperature):
        self.user_prompts.append(user_prompt)
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return ModelResponse(
            text=text, prompt_tokens=10, completion_tokens=5, raw_output=text,
        )

    def clear_memory(self):
        pass


class TestProposalExperimentRunner:
    def test_run_once_perfect_plan(self) -> None:
        task = _make_task()
        gold_steps = _gold_steps()
        model_responses = _direct_responses()
        executor = SymbolicPlanExecutor()
        runner = ProposalExperimentRunner(
            planner=FakePlannerBackend(gold_steps),
            memory_policy=FullContextPolicy(),
            model=FakeModelBackend(model_responses),
            prompt_builder=PromptBuilder(),
            parser=StrictActionParser(valid_action_names=list(ACTION_SCHEMAS)),
            evaluator=HardConstraintEvaluator(executor=executor),
            trace_builder=OracleTraceBuilder(executor=executor),
            executor=executor,
        )
        result = runner.run_once(
            task=task, prompt_mode="direct_action",
            max_new_tokens=128, temperature=0.0, max_rollout_steps=10,
        )
        assert result["evaluation"].goal_satisfied is True
        assert result["evaluation"].valid_plan is True
        assert len(result["predicted_steps"]) == 6

    def test_run_once_parse_error(self) -> None:
        task = _make_task()
        gold_steps = _gold_steps()
        executor = SymbolicPlanExecutor()
        runner = ProposalExperimentRunner(
            planner=FakePlannerBackend(gold_steps),
            memory_policy=FullContextPolicy(),
            model=FakeModelBackend(["not valid json"]),
            prompt_builder=PromptBuilder(),
            parser=StrictActionParser(valid_action_names=list(ACTION_SCHEMAS)),
            evaluator=HardConstraintEvaluator(executor=executor),
            trace_builder=OracleTraceBuilder(executor=executor),
            executor=executor,
        )
        result = runner.run_once(
            task=task, prompt_mode="direct_action",
            max_new_tokens=128, temperature=0.0, max_rollout_steps=10,
        )
        assert result["evaluation"].failure_reason == "parse_error"
        assert result["evaluation"].valid_plan is False

    def test_run_once_cot_mode(self) -> None:
        task = _make_task()
        gold_steps = _gold_steps()
        cot_responses = [
            f"Next.\n{response}"
            for response in _direct_responses()
        ]
        executor = SymbolicPlanExecutor()
        runner = ProposalExperimentRunner(
            planner=FakePlannerBackend(gold_steps),
            memory_policy=FullContextPolicy(),
            model=FakeModelBackend(cot_responses),
            prompt_builder=PromptBuilder(),
            parser=StrictActionParser(valid_action_names=list(ACTION_SCHEMAS)),
            evaluator=HardConstraintEvaluator(executor=executor),
            trace_builder=OracleTraceBuilder(executor=executor),
            executor=executor,
        )
        result = runner.run_once(
            task=task, prompt_mode="cot",
            max_new_tokens=256, temperature=0.0, max_rollout_steps=10,
        )
        assert result["evaluation"].goal_satisfied is True
        assert len(result["predicted_steps"]) == 6

    def test_run_once_records_trace_and_parse_recovery(self) -> None:
        task = _make_task()
        gold_steps = _gold_steps()
        executor = SymbolicPlanExecutor()
        runner = ProposalExperimentRunner(
            planner=FakePlannerBackend(gold_steps),
            memory_policy=FullContextPolicy(),
            model=FakeModelBackend([
                "I am still thinking and forgot the JSON",
                *_direct_responses(),
            ]),
            prompt_builder=PromptBuilder(),
            parser=StrictActionParser(valid_action_names=list(ACTION_SCHEMAS)),
            evaluator=HardConstraintEvaluator(executor=executor),
            trace_builder=OracleTraceBuilder(executor=executor),
            executor=executor,
        )
        result = runner.run_once(
            task=task, prompt_mode="cot",
            max_new_tokens=256, temperature=0.0, max_rollout_steps=10,
        )
        assert result["evaluation"].goal_satisfied is True
        assert result["parse_error_count"] == 1
        assert result["repair_attempt_count"] == 1
        assert len(result["trace_records"]) == 7
        assert result["trace_records"][0]["parse_error"]
        assert result["trace_records"][1]["repair_attempted"] is True

    def test_trace_records_annotated_parsed_action(self) -> None:
        task = _make_task()
        gold_steps = _gold_steps()
        executor = SymbolicPlanExecutor()
        runner = ProposalExperimentRunner(
            planner=FakePlannerBackend(gold_steps),
            memory_policy=FullContextPolicy(),
            model=FakeModelBackend(_direct_responses()),
            prompt_builder=PromptBuilder(),
            parser=StrictActionParser(valid_action_names=list(ACTION_SCHEMAS)),
            evaluator=HardConstraintEvaluator(executor=executor),
            trace_builder=OracleTraceBuilder(executor=executor),
            executor=executor,
        )
        result = runner.run_once(
            task=task, prompt_mode="direct_action",
            max_new_tokens=128, temperature=0.0, max_rollout_steps=10,
        )
        parsed = result["trace_records"][0]["parsed_action"]
        assert parsed["preconditions"] == ["at(robot_1, room_1)", "at(pkg_1, room_1)"]
        assert parsed["effects"] == ["package_inspected(pkg_1)"]

    @pytest.mark.parametrize("prompt_mode", ["direct_action", "cot"])
    def test_symbolic_repair_success_continues_rollout(self, prompt_mode: str) -> None:
        task = _make_task()
        gold_steps = _gold_steps()
        responses = [
            '{"action_name": "pickup", "arguments": ["robot_1", "pkg_1", "room_1"]}',
            *_direct_responses(),
        ]
        model = FakeModelBackend(responses)
        executor = SymbolicPlanExecutor()
        runner = ProposalExperimentRunner(
            planner=FakePlannerBackend(gold_steps),
            memory_policy=FullContextPolicy(),
            model=model,
            prompt_builder=PromptBuilder(),
            parser=StrictActionParser(valid_action_names=list(ACTION_SCHEMAS)),
            evaluator=HardConstraintEvaluator(executor=executor),
            trace_builder=OracleTraceBuilder(executor=executor),
            executor=executor,
            enable_symbolic_repair=True,
            max_symbolic_repair_attempts=1,
        )
        result = runner.run_once(
            task=task, prompt_mode=prompt_mode,
            max_new_tokens=128, temperature=0.0, max_rollout_steps=10,
        )

        assert result["evaluation"].goal_satisfied is True
        assert result["symbolic_failure_count"] == 1
        assert result["symbolic_repair_attempt_count"] == 1
        assert result["symbolic_repair_success_count"] == 1
        assert result["trace_records"][0]["symbolic_error"] == "precondition_violation"
        assert "package_inspected(pkg_1)" in result["trace_records"][0]["missing_preconditions"]
        assert result["trace_records"][1]["attempt_type"] == "symbolic_repair"
        assert "[Symbolic Feedback]" in model.user_prompts[1]
        assert result["predicted_steps"][0].action_name == "inspect_package"

    def test_symbolic_repair_failure_stops_with_original_failure(self) -> None:
        task = _make_task()
        gold_steps = _gold_steps()
        executor = SymbolicPlanExecutor()
        runner = ProposalExperimentRunner(
            planner=FakePlannerBackend(gold_steps),
            memory_policy=FullContextPolicy(),
            model=FakeModelBackend([
                '{"action_name": "pickup", "arguments": ["robot_1", "pkg_1", "room_1"]}',
                '{"action_name": "pickup", "arguments": ["robot_1", "pkg_1", "room_1"]}',
            ]),
            prompt_builder=PromptBuilder(),
            parser=StrictActionParser(valid_action_names=list(ACTION_SCHEMAS)),
            evaluator=HardConstraintEvaluator(executor=executor),
            trace_builder=OracleTraceBuilder(executor=executor),
            executor=executor,
            enable_symbolic_repair=True,
            max_symbolic_repair_attempts=1,
        )
        result = runner.run_once(
            task=task, prompt_mode="direct_action",
            max_new_tokens=128, temperature=0.0, max_rollout_steps=10,
        )

        assert result["evaluation"].failure_reason == "precondition_violation"
        assert result["evaluation"].valid_plan is False
        assert result["symbolic_repair_attempt_count"] == 1
        assert result["symbolic_repair_success_count"] == 0
        assert result["predicted_steps"] == []
        assert len(result["trace_records"]) == 2


class TestCheckpointing:
    def test_append_and_load_jsonl(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.jsonl"
        record1 = {"task_id": "t1", "condition": "full", "mode": "direct_action"}
        record2 = {"task_id": "t2", "condition": "trunc", "mode": "cot"}
        _append_jsonl(results_file, record1)
        _append_jsonl(results_file, record2)

        keys = _load_completed_keys(results_file)
        assert ("t1", "full", "direct_action") in keys
        assert ("t2", "trunc", "cot") in keys
        assert len(keys) == 2

    def test_load_from_nonexistent_file(self, tmp_path: Path) -> None:
        keys = _load_completed_keys(tmp_path / "missing.jsonl")
        assert keys == set()

    def test_load_handles_corrupt_lines(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.jsonl"
        results_file.write_text(
            '{"task_id": "t1", "condition": "full", "mode": "da"}\n'
            'not json\n'
            '{"task_id": "t2", "condition": "trunc", "mode": "cot"}\n'
        )
        keys = _load_completed_keys(results_file)
        assert len(keys) == 2


class TestBuildMemoryPolicy:
    def test_full(self) -> None:
        config = MemoryPolicyConfig(name="full", type="full")
        policy = build_memory_policy(config, "test-model")
        assert policy.name == "full"

    def test_unsupported_type(self) -> None:
        config = MemoryPolicyConfig(name="bad", type="rag")
        import pytest
        with pytest.raises(ValueError, match="Unsupported"):
            build_memory_policy(config, "test-model")


class TestModelOutputDir:
    def test_uses_model_label(self, tmp_path: Path) -> None:
        config = ProposalConfig()
        config.model.label = "Gemma 4"
        assert model_output_dir(tmp_path, config) == tmp_path / "gemma-4"

    def test_falls_back_to_api_model(self, tmp_path: Path) -> None:
        config = ProposalConfig()
        config.model.label = None
        config.model.api_model = "provider/model name"
        assert model_output_dir(tmp_path, config) == tmp_path / "provider-model-name"


class TestSerializeResult:
    def test_serializes_correctly(self) -> None:
        task = _make_task()
        from src.common.contracts import EvaluationBreakdown, MemoryContext, PlanTrace
        result = {
            "gold_plan": PlanTrace(
                steps=[
                    PlanStep(index=0, action_name="move", arguments=["r", "a", "b"]),
                ]
            ),
            "predicted_steps": [
                PlanStep(index=0, action_name="move", arguments=["r", "a", "b"]),
            ],
            "evaluation": EvaluationBreakdown(
                plan_accuracy=0.5, exact_match=False,
                precondition_violations=1, ordering_violations=0,
                grounding_errors=0, valid_plan=False, goal_satisfied=False,
                failure_reason="precondition_violation", notes=["test"],
            ),
            "response": ModelResponse(
                text="test", prompt_tokens=10, completion_tokens=5,
                raw_output="full output",
            ),
        }
        policy_config = MemoryPolicyConfig(name="full", type="full")
        record = _serialize_result(task, policy_config, "direct_action", 42, result)
        assert record["task_id"] == "test-1"
        assert record["condition"] == "full"
        assert record["mode"] == "direct_action"
        assert record["plan_accuracy"] == 0.5
        assert record["raw_output"] == "full output"
        assert record["gold_steps"] == [
            {"action_name": "move", "arguments": ["r", "a", "b"]}
        ]
        assert record["trace_file"] == "traces.jsonl"
        assert "timestamp" in record
        # Should be JSON-serializable
        json.dumps(record)
