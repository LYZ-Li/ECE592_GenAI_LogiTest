"""Natural-language instruction generation for the NL-to-Plan study."""

from __future__ import annotations

from typing import List

from src.common.contracts import PlanningTaskInstance


def generate_instruction(problem: PlanningTaskInstance) -> str:
    """Generate a deterministic natural-language instruction from a task.

    Reads from problem.metadata["deliveries"] to produce template-based
    NL instructions describing what needs to be accomplished, without
    revealing the symbolic goal predicates.

    Args:
        problem: A planning task instance with delivery metadata.

    Returns:
        A natural-language instruction string.
    """
    deliveries: List[dict] = problem.metadata.get("deliveries", [])
    if not deliveries:
        return problem.goal_text

    robot = problem.typed_objects.get("robot", ["robot_1"])[0]

    if len(deliveries) == 1:
        d = deliveries[0]
        return (
            f"Using {robot}, deliver {d['package']} "
            f"from {d['source']} to {d['destination']}."
        )

    clauses = [
        f"{d['package']} from {d['source']} to {d['destination']}"
        for d in deliveries
    ]
    if len(clauses) == 2:
        joined = f"{clauses[0]} and {clauses[1]}"
    else:
        joined = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"

    return f"Using {robot}, deliver {joined}."
