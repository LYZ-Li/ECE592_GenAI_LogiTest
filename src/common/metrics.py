"""Pure metric functions shared by planner and evaluator.

Extracted here to break the circular dependency between
src.runner.planner and src.eval.check (both need ordering_violations
and plan_accuracy but depend on each other for other reasons).
"""

from __future__ import annotations

from typing import List


def ordering_violations(
    gold_signatures: List[str], predicted_signatures: List[str]
) -> int:
    """Count monotonicity violations in predicted vs. gold action order.

    For each predicted action that appears in the gold plan, check whether
    its gold-plan position is strictly after the previous matched position.
    A violation occurs when a predicted action appears *earlier* in the gold
    plan than the previously matched action.

    Args:
        gold_signatures: Action signatures from the oracle plan, in order.
        predicted_signatures: Action signatures from the model's plan.

    Returns:
        Number of ordering violations.
    """
    gold_positions = {sig: idx for idx, sig in enumerate(gold_signatures)}
    violations = 0
    previous_position = -1
    for sig in predicted_signatures:
        position = gold_positions.get(sig)
        if position is None:
            continue
        if position < previous_position:
            violations += 1
        previous_position = position
    return violations


def plan_accuracy(
    gold_signatures: List[str], predicted_signatures: List[str]
) -> float:
    """Compute prefix-overlap accuracy between gold and predicted plans.

    Counts how many positions match between the two signature lists
    (compared element-wise from the start), divided by the gold plan length.

    Args:
        gold_signatures: Action signatures from the oracle plan, in order.
        predicted_signatures: Action signatures from the model's plan.

    Returns:
        Accuracy in [0.0, 1.0]. Returns 1.0 if both plans are empty,
        0.0 if only the gold plan is empty.
    """
    if not gold_signatures:
        return 1.0 if not predicted_signatures else 0.0
    overlap = sum(
        1
        for expected, actual in zip(gold_signatures, predicted_signatures)
        if expected == actual
    )
    return overlap / len(gold_signatures)
