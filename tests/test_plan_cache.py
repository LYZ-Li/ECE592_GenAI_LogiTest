"""Tests for persistent gold-plan cache."""

from __future__ import annotations

import json
from pathlib import Path

from src.common.contracts import PlanStep, PlanTrace, PlanningTaskInstance
from src.generator.generate import ACTION_SCHEMAS
from src.runner.plan_cache import ensure_gold_plans, task_hash
from src.runner.planner import SymbolicPlanExecutor


def _make_task() -> PlanningTaskInstance:
    return PlanningTaskInstance(
        task_id="logistics-easy-42-0",
        domain_name="household_logistics",
        problem_name="easy_instance_0",
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
        pddl_domain="domain",
        pddl_problem="problem",
        metadata={
            "difficulty": "easy",
            "instance_index": 0,
            "action_schemas": {
                name: list(args) for name, args in ACTION_SCHEMAS.items()
            },
        },
    )


def _gold_plan() -> PlanTrace:
    return PlanTrace(
        steps=[
            PlanStep(index=0, action_name="inspect_package", arguments=["robot_1", "pkg_1", "room_1"]),
            PlanStep(index=1, action_name="pickup", arguments=["robot_1", "pkg_1", "room_1"]),
            PlanStep(index=2, action_name="verify_grasp", arguments=["robot_1", "pkg_1", "room_1"]),
            PlanStep(index=3, action_name="move", arguments=["robot_1", "room_1", "room_2"]),
            PlanStep(index=4, action_name="drop", arguments=["robot_1", "pkg_1", "room_2"]),
            PlanStep(index=5, action_name="verify_delivery", arguments=["robot_1", "pkg_1", "room_2"]),
        ]
    )


class FakePlanner:
    def __init__(self) -> None:
        self.solve_calls = 0

    def solve(self, task: PlanningTaskInstance) -> PlanTrace:
        self.solve_calls += 1
        return _gold_plan()

    def validate(self, task, candidate_steps, gold_steps):
        raise NotImplementedError


def test_generates_and_reuses_gold_plan(tmp_path: Path) -> None:
    task = _make_task()
    planner = FakePlanner()
    executor = SymbolicPlanExecutor()

    first = ensure_gold_plans([task], planner, tmp_path, executor)
    assert planner.solve_calls == 1
    assert first[task.task_id]["cache_status"] == "generated"
    assert (tmp_path / f"{task.task_id}.json").exists()

    second = ensure_gold_plans([task], planner, tmp_path, executor)
    assert planner.solve_calls == 1
    assert second[task.task_id]["cache_status"] == "reused"
    assert [s.signature for s in second[task.task_id]["plan"].steps] == [
        "inspect_package robot_1 pkg_1 room_1",
        "pickup robot_1 pkg_1 room_1",
        "verify_grasp robot_1 pkg_1 room_1",
        "move robot_1 room_1 room_2",
        "drop robot_1 pkg_1 room_2",
        "verify_delivery robot_1 pkg_1 room_2",
    ]


def test_regenerates_stale_plan_cache(tmp_path: Path) -> None:
    task = _make_task()
    path = tmp_path / f"{task.task_id}.json"
    path.write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "task_hash": "stale",
                "pddl_problem_hash": "stale",
                "validation_status": "valid",
                "gold_steps": [],
            }
        )
    )

    planner = FakePlanner()
    plans = ensure_gold_plans([task], planner, tmp_path, SymbolicPlanExecutor())
    assert planner.solve_calls == 1
    assert plans[task.task_id]["cache_status"] == "generated"


def test_task_hash_is_deterministic() -> None:
    assert task_hash(_make_task()) == task_hash(_make_task())
