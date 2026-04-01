"""Symbolic evaluation of predicted action traces.

Uses the canonical ordering_violations and plan_accuracy from
src.common.metrics (fix Q6: no duplicate logic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.common.contracts import EvaluationBreakdown, PlanStep, PlanningTaskInstance
from src.common.metrics import ordering_violations, plan_accuracy
from src.runner.planner import SymbolicPlanExecutor


@dataclass
class HardConstraintEvaluator:
    """Evaluate a predicted action trace against symbolic task constraints."""

    executor: SymbolicPlanExecutor = field(default_factory=SymbolicPlanExecutor)

    def evaluate(
        self,
        task: PlanningTaskInstance,
        gold_steps: List[PlanStep],
        predicted_steps: List[PlanStep],
    ) -> EvaluationBreakdown:
        """Evaluate predicted steps against gold plan and task constraints.

        Args:
            task: The planning task instance.
            gold_steps: Oracle plan steps from Fast Downward.
            predicted_steps: Model-generated plan steps.

        Returns:
            EvaluationBreakdown with all four spec metrics.
        """
        replay = self.executor.replay(task, predicted_steps)
        gold_sigs = [step.signature for step in gold_steps]
        pred_sigs = [step.signature for step in predicted_steps]

        acc = plan_accuracy(gold_sigs, pred_sigs)
        ord_violations = ordering_violations(gold_sigs, pred_sigs)
        exact = gold_sigs == pred_sigs

        failure_reason = replay.failure_reason
        if failure_reason is None and not replay.goal_satisfied:
            failure_reason = "goal_not_satisfied"
        valid = replay.goal_satisfied and failure_reason is None
        correct_but_suboptimal = valid and not exact

        notes: List[str] = []
        if correct_but_suboptimal:
            notes.append(
                "Predicted plan is symbolically valid but differs from the oracle sequence."
            )
        if failure_reason:
            notes.append(f"Rollout stopped due to `{failure_reason}`.")

        return EvaluationBreakdown(
            plan_accuracy=acc,
            exact_match=exact,
            precondition_violations=replay.precondition_violations,
            ordering_violations=ord_violations,
            grounding_errors=replay.grounding_errors,
            valid_plan=valid,
            goal_satisfied=replay.goal_satisfied,
            correct_but_suboptimal=correct_but_suboptimal,
            executed_steps=replay.executed_steps,
            failure_reason=failure_reason,
            step_diagnostics=replay.step_diagnostics,
            notes=notes,
        )
