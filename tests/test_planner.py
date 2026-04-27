"""Tests for src.runner.planner."""

import pytest

from src.common.contracts import PlanStep, PlanningTaskInstance
from src.generator.generate import ACTION_SCHEMAS
from src.runner.planner import SymbolicPlanExecutor


def _make_task() -> PlanningTaskInstance:
    """Create a simple logistics task for testing."""
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
        initial_facts=[
            "at(pkg_1, room_1)",
            "at(robot_1, room_1)",
            "connected(room_1, room_2)",
            "connected(room_2, room_1)",
            "connected(room_2, room_3)",
            "connected(room_3, room_2)",
            "accessible(pkg_1)",
            "gripper_calibrated(robot_1)",
            "gripper_clean(robot_1)",
            "gripper_inspected(robot_1)",
            "handempty(robot_1)",
            "mobility_ready(robot_1)",
            "stable(pkg_1)",
        ],
        goal_facts=["delivery_verified(pkg_1, room_3)"],
        metadata={
            "action_schemas": {
                name: list(args) for name, args in ACTION_SCHEMAS.items()
            }
        },
    )


def _valid_plan_steps() -> list[PlanStep]:
    """A correct plan with inspection and delivery verification."""
    return [
        PlanStep(index=0, action_name="inspect_package", arguments=["robot_1", "pkg_1", "room_1"]),
        PlanStep(index=1, action_name="pickup", arguments=["robot_1", "pkg_1", "room_1"]),
        PlanStep(index=2, action_name="verify_grasp", arguments=["robot_1", "pkg_1", "room_1"]),
        PlanStep(index=3, action_name="move", arguments=["robot_1", "room_1", "room_2"]),
        PlanStep(index=4, action_name="move", arguments=["robot_1", "room_2", "room_3"]),
        PlanStep(index=5, action_name="drop", arguments=["robot_1", "pkg_1", "room_3"]),
        PlanStep(index=6, action_name="verify_delivery", arguments=["robot_1", "pkg_1", "room_3"]),
    ]


class TestSymbolicPlanExecutor:
    def setup_method(self) -> None:
        self.executor = SymbolicPlanExecutor()
        self.task = _make_task()

    def test_initial_state(self) -> None:
        state = self.executor.initial_state(self.task)
        assert "at(robot_1, room_1)" in state
        assert "handempty(robot_1)" in state

    def test_goal_satisfied(self) -> None:
        state = ["delivery_verified(pkg_1, room_3)", "other_fact"]
        assert self.executor.goal_satisfied(self.task, state) is True

    def test_goal_not_satisfied(self) -> None:
        state = ["at(pkg_1, room_1)"]
        assert self.executor.goal_satisfied(self.task, state) is False

    def test_replay_valid_plan(self) -> None:
        steps = _valid_plan_steps()
        result = self.executor.replay(self.task, steps)
        assert result.goal_satisfied is True
        assert result.valid_plan is True
        assert result.grounding_errors == 0
        assert result.precondition_violations == 0
        assert result.executed_steps == 7
        assert len(result.episodes) == 7
        assert result.failure_reason is None

    def test_replay_generates_episodes(self) -> None:
        steps = _valid_plan_steps()
        result = self.executor.replay(self.task, steps)
        for i, episode in enumerate(result.episodes):
            assert episode.step_index == i
            assert "Step " in episode.text
            assert "Action:" in episode.text

    def test_replay_detects_grounding_error(self) -> None:
        steps = [
            PlanStep(index=0, action_name="move", arguments=["robot_1", "room_1", "nonexistent"]),
        ]
        result = self.executor.replay(self.task, steps)
        assert result.grounding_errors == 1
        assert result.failure_reason == "grounding_error"

    def test_replay_detects_precondition_violation(self) -> None:
        # Try to move from room_2 when robot is at room_1
        steps = [
            PlanStep(index=0, action_name="move", arguments=["robot_1", "room_2", "room_3"]),
        ]
        result = self.executor.replay(self.task, steps)
        assert result.precondition_violations == 1
        assert result.failure_reason == "precondition_violation"

    def test_replay_detects_wrong_arg_count(self) -> None:
        steps = [
            PlanStep(index=0, action_name="move", arguments=["robot_1", "room_1"]),
        ]
        result = self.executor.replay(self.task, steps)
        assert result.grounding_errors == 1

    def test_replay_detects_type_mismatch(self) -> None:
        # Try to move a package instead of a robot
        steps = [
            PlanStep(index=0, action_name="move", arguments=["pkg_1", "room_1", "room_2"]),
        ]
        result = self.executor.replay(self.task, steps)
        assert result.grounding_errors == 1

    def test_replay_empty_plan(self) -> None:
        result = self.executor.replay(self.task, [])
        assert result.executed_steps == 0
        assert result.goal_satisfied is False

    def test_annotate_step(self) -> None:
        step = PlanStep(index=0, action_name="move", arguments=["robot_1", "room_1", "room_2"])
        annotated = self.executor.annotate_step(self.task, step)
        assert len(annotated.preconditions) == 3
        assert "at(robot_1, room_1)" in annotated.preconditions
        assert "connected(room_1, room_2)" in annotated.preconditions
        assert "mobility_ready(robot_1)" in annotated.preconditions
        assert len(annotated.effects) == 2

    def test_all_supported_actions_have_preconditions_and_effects(self) -> None:
        examples = {
            "move": ["robot_1", "room_1", "room_2"],
            "inspect_gripper": ["robot_1", "room_1"],
            "clean_gripper": ["robot_1", "room_1"],
            "calibrate_gripper": ["robot_1", "room_1"],
            "inspect_package": ["robot_1", "pkg_1", "room_1"],
            "clear_obstruction": ["robot_1", "pkg_1", "room_1"],
            "pickup": ["robot_1", "pkg_1", "room_1"],
            "verify_grasp": ["robot_1", "pkg_1", "room_1"],
            "regrasp": ["robot_1", "pkg_1", "room_1"],
            "drop": ["robot_1", "pkg_1", "room_1"],
            "verify_delivery": ["robot_1", "pkg_1", "room_1"],
        }
        for action, args in examples.items():
            annotated = self.executor.annotate_step(
                self.task,
                PlanStep(index=0, action_name=action, arguments=args),
            )
            assert annotated.preconditions, action
            assert annotated.effects, action

    def test_pickup_before_package_inspection_violates_precondition(self) -> None:
        result = self.executor.replay(
            self.task,
            [PlanStep(index=0, action_name="pickup", arguments=["robot_1", "pkg_1", "room_1"])],
        )
        assert result.precondition_violations == 1
        assert result.failure_reason == "precondition_violation"

    def test_drop_before_grasp_verification_violates_precondition(self) -> None:
        steps = [
            PlanStep(index=0, action_name="inspect_package", arguments=["robot_1", "pkg_1", "room_1"]),
            PlanStep(index=1, action_name="pickup", arguments=["robot_1", "pkg_1", "room_1"]),
            PlanStep(index=2, action_name="drop", arguments=["robot_1", "pkg_1", "room_1"]),
        ]
        result = self.executor.replay(self.task, steps)
        assert result.precondition_violations == 1
        assert result.failure_reason == "precondition_violation"

    def test_regrasp_stable_package_violates_precondition(self) -> None:
        steps = [
            PlanStep(index=0, action_name="inspect_package", arguments=["robot_1", "pkg_1", "room_1"]),
            PlanStep(index=1, action_name="pickup", arguments=["robot_1", "pkg_1", "room_1"]),
            PlanStep(index=2, action_name="regrasp", arguments=["robot_1", "pkg_1", "room_1"]),
        ]
        result = self.executor.replay(self.task, steps)
        assert result.precondition_violations == 1
        assert result.failure_reason == "precondition_violation"

    def test_domain_assertion(self) -> None:
        """Guard against unsupported action domains."""
        task = _make_task()
        task.metadata["action_schemas"]["fly"] = ["robot", "room", "room"]
        steps = [PlanStep(index=0, action_name="fly", arguments=["robot_1", "room_1", "room_2"])]
        with pytest.raises(AssertionError, match="household_logistics"):
            self.executor.replay(task, steps)
