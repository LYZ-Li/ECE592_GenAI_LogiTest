"""Tests for src.eval.check."""

from src.common.contracts import PlanStep, PlanningTaskInstance
from src.eval.check import HardConstraintEvaluator
from src.generator.generate import ACTION_SCHEMAS


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


def _gold_steps() -> list[PlanStep]:
    return [
        PlanStep(index=0, action_name="inspect_package", arguments=["robot_1", "pkg_1", "room_1"]),
        PlanStep(index=1, action_name="pickup", arguments=["robot_1", "pkg_1", "room_1"]),
        PlanStep(index=2, action_name="verify_grasp", arguments=["robot_1", "pkg_1", "room_1"]),
        PlanStep(index=3, action_name="move", arguments=["robot_1", "room_1", "room_2"]),
        PlanStep(index=4, action_name="move", arguments=["robot_1", "room_2", "room_3"]),
        PlanStep(index=5, action_name="drop", arguments=["robot_1", "pkg_1", "room_3"]),
        PlanStep(index=6, action_name="verify_delivery", arguments=["robot_1", "pkg_1", "room_3"]),
    ]


class TestHardConstraintEvaluator:
    def setup_method(self) -> None:
        self.evaluator = HardConstraintEvaluator()
        self.task = _make_task()

    def test_perfect_plan(self) -> None:
        gold = _gold_steps()
        result = self.evaluator.evaluate(self.task, gold, gold)
        assert result.plan_accuracy == 1.0
        assert result.exact_match is True
        assert result.valid_plan is True
        assert result.goal_satisfied is True
        assert result.precondition_violations == 0
        assert result.ordering_violations == 0
        assert result.grounding_errors == 0

    def test_wrong_first_step(self) -> None:
        gold = _gold_steps()
        predicted = [
            PlanStep(index=0, action_name="move", arguments=["robot_1", "room_1", "room_2"]),
        ]
        result = self.evaluator.evaluate(self.task, gold, predicted)
        assert result.exact_match is False
        assert result.goal_satisfied is False

    def test_grounding_error(self) -> None:
        gold = _gold_steps()
        predicted = [
            PlanStep(index=0, action_name="move", arguments=["robot_1", "room_1", "nonexistent"]),
        ]
        result = self.evaluator.evaluate(self.task, gold, predicted)
        assert result.grounding_errors == 1
        assert result.valid_plan is False

    def test_empty_predicted(self) -> None:
        gold = _gold_steps()
        result = self.evaluator.evaluate(self.task, gold, [])
        assert result.plan_accuracy == 0.0
        assert result.exact_match is False
        assert result.goal_satisfied is False

    def test_ordering_violations(self) -> None:
        gold = _gold_steps()
        # Gold order: pickup(0), move-1-2(1), move-2-3(2), drop(3)
        # Predict drop then pickup — drop is at gold pos 3, pickup at gold pos 0
        # That's a violation: 0 < 3
        predicted = [
            PlanStep(index=0, action_name="drop", arguments=["robot_1", "pkg_1", "room_3"]),
            PlanStep(index=1, action_name="pickup", arguments=["robot_1", "pkg_1", "room_1"]),
        ]
        result = self.evaluator.evaluate(self.task, gold, predicted)
        assert result.ordering_violations >= 1
