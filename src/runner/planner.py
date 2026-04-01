"""Planner interfaces, symbolic replay, and oracle trace building."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Protocol, Sequence, Set

from src.common.contracts import (
    EvaluationBreakdown,
    PlanStep,
    PlanTrace,
    PlanningTaskInstance,
    StepEvaluation,
    TraceEpisode,
)
from src.common.metrics import ordering_violations, plan_accuracy


@dataclass
class ExecutionResult:
    """Result of replaying a candidate action sequence symbolically."""

    final_state: List[str]
    episodes: List[TraceEpisode]
    step_diagnostics: List[StepEvaluation]
    valid_plan: bool
    goal_satisfied: bool
    executed_steps: int
    grounding_errors: int
    precondition_violations: int
    failure_reason: str | None = None


class PlannerBackend(Protocol):
    """Minimal interface for symbolic planning and candidate validation."""

    def solve(self, task: PlanningTaskInstance) -> PlanTrace: ...

    def validate(
        self,
        task: PlanningTaskInstance,
        candidate_steps: List[PlanStep],
        gold_steps: List[PlanStep],
    ) -> EvaluationBreakdown: ...


@dataclass
class SymbolicPlanExecutor:
    """Shared execution semantics for trace building and candidate validation.

    Hard-codes action semantics for the household_logistics domain
    (move, pickup, drop). An assertion guards against running on an
    unsupported domain.
    """

    def initial_state(self, task: PlanningTaskInstance) -> List[str]:
        return sorted(task.initial_facts)

    def goal_satisfied(
        self, task: PlanningTaskInstance, state: Sequence[str]
    ) -> bool:
        state_facts = set(state)
        return all(goal in state_facts for goal in task.goal_facts)

    def annotate_step(
        self, task: PlanningTaskInstance, step: PlanStep
    ) -> PlanStep:
        preconditions = self._preconditions(step)
        effects = self._effect_descriptions(step)
        return PlanStep(
            index=step.index,
            action_name=step.action_name,
            arguments=list(step.arguments),
            preconditions=preconditions,
            effects=effects,
            notes=dict(step.notes),
        )

    def replay(
        self, task: PlanningTaskInstance, steps: Sequence[PlanStep]
    ) -> ExecutionResult:
        """Replay an action sequence against the task's initial state.

        Args:
            task: The planning task instance.
            steps: Ordered action sequence to replay.

        Returns:
            ExecutionResult with episodes, diagnostics, and final state.
        """
        # Guard: only household_logistics actions are supported
        schemas = task.metadata.get("action_schemas", {})
        assert set(schemas.keys()) <= {"move", "pickup", "drop"}, (
            f"SymbolicPlanExecutor only supports household_logistics actions, "
            f"got: {set(schemas.keys())}"
        )

        state = set(task.initial_facts)
        episodes: List[TraceEpisode] = []
        diagnostics: List[StepEvaluation] = []
        grounding_errors = 0
        precondition_violations = 0

        for index, original_step in enumerate(steps):
            step = self.annotate_step(
                task,
                PlanStep(
                    index=index,
                    action_name=original_step.action_name,
                    arguments=list(original_step.arguments),
                    notes=dict(original_step.notes),
                ),
            )
            state_before = sorted(state)
            validation = self._validate_grounding(task, step)
            if validation["grounding_error"]:
                grounding_errors += 1
                diagnostics.append(
                    StepEvaluation(
                        step_index=index,
                        action_signature=step.signature,
                        legal=False,
                        grounding_error=True,
                        preconditions_satisfied=False,
                        goal_satisfied_after_step=self.goal_satisfied(
                            task, state_before
                        ),
                        notes=validation["notes"],
                    )
                )
                return ExecutionResult(
                    final_state=state_before,
                    episodes=episodes,
                    step_diagnostics=diagnostics,
                    valid_plan=False,
                    goal_satisfied=self.goal_satisfied(task, state_before),
                    executed_steps=len(episodes),
                    grounding_errors=grounding_errors,
                    precondition_violations=precondition_violations,
                    failure_reason="grounding_error",
                )

            missing_preconditions = [
                fact for fact in step.preconditions if fact not in state
            ]
            if missing_preconditions:
                precondition_violations += 1
                diagnostics.append(
                    StepEvaluation(
                        step_index=index,
                        action_signature=step.signature,
                        legal=False,
                        grounding_error=False,
                        preconditions_satisfied=False,
                        goal_satisfied_after_step=self.goal_satisfied(
                            task, state_before
                        ),
                        notes=[
                            "Missing preconditions: "
                            + "; ".join(missing_preconditions)
                        ],
                    )
                )
                return ExecutionResult(
                    final_state=state_before,
                    episodes=episodes,
                    step_diagnostics=diagnostics,
                    valid_plan=False,
                    goal_satisfied=self.goal_satisfied(task, state_before),
                    executed_steps=len(episodes),
                    grounding_errors=grounding_errors,
                    precondition_violations=precondition_violations,
                    failure_reason="precondition_violation",
                )

            state = self._apply_effects(state, step)
            state_after = sorted(state)
            diagnostics.append(
                StepEvaluation(
                    step_index=index,
                    action_signature=step.signature,
                    legal=True,
                    grounding_error=False,
                    preconditions_satisfied=True,
                    goal_satisfied_after_step=self.goal_satisfied(
                        task, state_after
                    ),
                    notes=[],
                )
            )
            episodes.append(
                TraceEpisode(
                    step_index=index,
                    state_before=state_before,
                    action_signature=step.signature,
                    state_after=state_after,
                    text=self._episode_text(
                        index=index,
                        state_before=state_before,
                        step=step,
                        state_after=state_after,
                    ),
                )
            )

        return ExecutionResult(
            final_state=sorted(state),
            episodes=episodes,
            step_diagnostics=diagnostics,
            valid_plan=self.goal_satisfied(task, state),
            goal_satisfied=self.goal_satisfied(task, state),
            executed_steps=len(episodes),
            grounding_errors=grounding_errors,
            precondition_violations=precondition_violations,
            failure_reason=None,
        )

    def _validate_grounding(
        self, task: PlanningTaskInstance, step: PlanStep
    ) -> Dict[str, object]:
        action_schemas = task.metadata.get("action_schemas", {})
        expected_types = action_schemas.get(step.action_name)
        if expected_types is None:
            return {
                "grounding_error": True,
                "notes": [f"Unknown action `{step.action_name}`."],
            }

        if len(expected_types) != len(step.arguments):
            return {
                "grounding_error": True,
                "notes": [
                    f"Action `{step.action_name}` expects {len(expected_types)} "
                    f"arguments but received {len(step.arguments)}.",
                ],
            }

        type_lookup = {
            obj: object_type
            for object_type, objects in task.typed_objects.items()
            for obj in objects
        }
        notes: List[str] = []
        grounding_error = False
        for argument, expected_type in zip(step.arguments, expected_types):
            actual_type = type_lookup.get(argument)
            if actual_type is None:
                notes.append(f"Unknown object `{argument}`.")
                grounding_error = True
            elif actual_type != expected_type:
                notes.append(
                    f"Object `{argument}` has type `{actual_type}` "
                    f"but `{step.action_name}` expects `{expected_type}`."
                )
                grounding_error = True

        return {"grounding_error": grounding_error, "notes": notes}

    def _preconditions(self, step: PlanStep) -> List[str]:
        arguments = step.arguments
        if step.action_name == "move" and len(arguments) == 3:
            robot, room_from, room_to = arguments
            return [
                f"at({robot}, {room_from})",
                f"connected({room_from}, {room_to})",
            ]
        if step.action_name == "pickup" and len(arguments) == 3:
            robot, package, room = arguments
            return [
                f"at({robot}, {room})",
                f"at({package}, {room})",
                f"handempty({robot})",
            ]
        if step.action_name == "drop" and len(arguments) == 3:
            robot, package, room = arguments
            return [f"at({robot}, {room})", f"holding({robot}, {package})"]
        return []

    def _effect_descriptions(self, step: PlanStep) -> List[str]:
        arguments = step.arguments
        if step.action_name == "move" and len(arguments) == 3:
            robot, room_from, room_to = arguments
            return [f"not at({robot}, {room_from})", f"at({robot}, {room_to})"]
        if step.action_name == "pickup" and len(arguments) == 3:
            robot, package, room = arguments
            return [
                f"not at({package}, {room})",
                f"not handempty({robot})",
                f"holding({robot}, {package})",
            ]
        if step.action_name == "drop" and len(arguments) == 3:
            robot, package, room = arguments
            return [
                f"not holding({robot}, {package})",
                f"handempty({robot})",
                f"at({package}, {room})",
            ]
        return []

    def _apply_effects(self, state: Set[str], step: PlanStep) -> Set[str]:
        new_state = set(state)
        arguments = step.arguments
        if step.action_name == "move":
            robot, room_from, room_to = arguments
            new_state.discard(f"at({robot}, {room_from})")
            new_state.add(f"at({robot}, {room_to})")
        elif step.action_name == "pickup":
            robot, package, room = arguments
            new_state.discard(f"at({package}, {room})")
            new_state.discard(f"handempty({robot})")
            new_state.add(f"holding({robot}, {package})")
        elif step.action_name == "drop":
            robot, package, room = arguments
            new_state.discard(f"holding({robot}, {package})")
            new_state.add(f"handempty({robot})")
            new_state.add(f"at({package}, {room})")
        return new_state

    @staticmethod
    def _episode_text(
        index: int,
        state_before: List[str],
        step: PlanStep,
        state_after: List[str],
    ) -> str:
        return "\n".join(
            [
                f"Step {index}",
                "State before:",
                "; ".join(state_before),
                f"Action: {step.signature}",
                "State after:",
                "; ".join(state_after),
            ]
        )


@dataclass
class OracleTraceBuilder:
    """Build text episodes from a gold symbolic plan."""

    executor: SymbolicPlanExecutor = field(default_factory=SymbolicPlanExecutor)

    def build_history(
        self,
        task: PlanningTaskInstance,
        plan: PlanTrace,
    ) -> PlanTrace:
        replay = self.executor.replay(task, plan.steps)
        if not replay.goal_satisfied:
            raise ValueError(
                "Oracle plan replay failed to satisfy the task goal."
            )
        return PlanTrace(steps=plan.steps, episodes=replay.episodes)


@dataclass
class FastDownwardPlannerBackend:
    """Solve PDDL tasks through Unified Planning and Fast Downward."""

    engine_name: str = "fast-downward"
    timeout_seconds: int = 30
    executor: SymbolicPlanExecutor = field(default_factory=SymbolicPlanExecutor)

    def solve(self, task: PlanningTaskInstance) -> PlanTrace:
        """Solve a PDDL task and return the oracle plan.

        Fix Q4: does NOT mutate task.metadata. Gold steps live in the
        returned PlanTrace only.
        """
        if not task.pddl_domain or not task.pddl_problem:
            raise ValueError(
                "PlanningTaskInstance must provide both pddl_domain and pddl_problem."
            )

        PDDLReader, OneshotPlanner = self._require_dependencies()
        with TemporaryDirectory(prefix="planner_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            domain_path = tmp_path / "domain.pddl"
            problem_path = tmp_path / "problem.pddl"
            domain_path.write_text(task.pddl_domain)
            problem_path.write_text(task.pddl_problem)

            reader = PDDLReader()
            problem = reader.parse_problem(str(domain_path), str(problem_path))
            with OneshotPlanner(
                name=self.engine_name, problem_kind=problem.kind
            ) as planner:
                try:
                    result = planner.solve(problem, timeout=self.timeout_seconds)
                except TypeError:
                    result = planner.solve(problem)

        plan = getattr(result, "plan", None)
        if plan is None:
            status = getattr(result, "status", "unknown")
            raise RuntimeError(
                f"Fast Downward did not produce a plan. Status: {status}"
            )

        raw_actions = list(getattr(plan, "actions", []))
        if not raw_actions:
            raise RuntimeError(
                "Fast Downward returned an empty plan for a non-trivial task."
            )

        steps: List[PlanStep] = []
        for index, action_instance in enumerate(raw_actions):
            arguments = [
                self._parameter_name(parameter)
                for parameter in action_instance.actual_parameters
            ]
            step = self.executor.annotate_step(
                task,
                PlanStep(
                    index=index,
                    action_name=action_instance.action.name,
                    arguments=arguments,
                ),
            )
            steps.append(step)

        # Fix Q4: do NOT mutate task.metadata
        return PlanTrace(steps=steps)

    def validate(
        self,
        task: PlanningTaskInstance,
        candidate_steps: List[PlanStep],
        gold_steps: List[PlanStep],
    ) -> EvaluationBreakdown:
        """Validate candidate steps against the task and gold plan.

        Fix Q6: uses shared metrics from src.common.metrics instead of
        duplicate local logic.
        """
        replay = self.executor.replay(task, candidate_steps)

        gold_sigs = [s.signature for s in gold_steps]
        pred_sigs = [s.signature for s in candidate_steps]
        acc = plan_accuracy(gold_sigs, pred_sigs)
        ord_violations = ordering_violations(gold_sigs, pred_sigs)
        exact = gold_sigs == pred_sigs

        failure_reason = replay.failure_reason
        if failure_reason is None and not replay.goal_satisfied:
            failure_reason = "goal_not_satisfied"
        valid = (
            replay.valid_plan and replay.goal_satisfied and failure_reason is None
        )

        notes: List[str] = []
        if not exact:
            notes.append(
                "Exact-match metrics are secondary to symbolic validity."
            )

        return EvaluationBreakdown(
            plan_accuracy=acc,
            exact_match=exact,
            precondition_violations=replay.precondition_violations,
            ordering_violations=ord_violations,
            grounding_errors=replay.grounding_errors,
            valid_plan=valid,
            goal_satisfied=replay.goal_satisfied,
            correct_but_suboptimal=valid and not exact,
            executed_steps=replay.executed_steps,
            failure_reason=None if valid else failure_reason,
            step_diagnostics=replay.step_diagnostics,
            notes=notes,
        )

    @staticmethod
    def _parameter_name(parameter: object) -> str:
        if hasattr(parameter, "object") and callable(
            getattr(parameter, "object")
        ):
            obj = parameter.object()
            return getattr(obj, "name", str(obj))
        return getattr(parameter, "name", str(parameter))

    @staticmethod
    def _require_dependencies():
        try:
            import up_fast_downward  # noqa: F401
            from unified_planning.io import PDDLReader
            from unified_planning.shortcuts import OneshotPlanner
        except ImportError as exc:
            raise RuntimeError(
                "FastDownwardPlannerBackend requires `unified-planning` and "
                "`up-fast-downward`. Install them before running the pipeline."
            ) from exc
        return PDDLReader, OneshotPlanner
