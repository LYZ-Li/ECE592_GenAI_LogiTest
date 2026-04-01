"""Core data contracts for the memory compression study."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass
class PlanStep:
    """A single grounded action in a symbolic or predicted plan."""

    index: int
    action_name: str
    arguments: List[str]
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        return " ".join([self.action_name, *self.arguments]).strip()


@dataclass
class TraceEpisode:
    """A single replayed transition in the execution history."""

    step_index: int
    state_before: List[str]
    action_signature: str
    state_after: List[str]
    text: str


@dataclass
class PlanTrace:
    """An ordered sequence of plan steps plus replayed transition episodes."""

    steps: List[PlanStep]
    episodes: List[TraceEpisode] = field(default_factory=list)

    def signatures(self) -> List[str]:
        return [step.signature for step in self.steps]


@dataclass
class PlanningTaskInstance:
    """A synthetic planning problem plus metadata used by the experiment."""

    task_id: str
    domain_name: str
    problem_name: str
    goal_text: str
    valid_objects: List[str]
    typed_objects: Dict[str, List[str]] = field(default_factory=dict)
    initial_facts: List[str] = field(default_factory=list)
    goal_facts: List[str] = field(default_factory=list)
    pddl_domain: str | None = None
    pddl_problem: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    """A single stored episode that can be truncated or replayed in prompts."""

    step_index: int
    text: str
    tags: List[str] = field(default_factory=list)
    token_count: int = 0


@dataclass
class MemoryContext:
    """The material actually sent to the model under a memory policy."""

    policy_name: str
    prompt_context: str
    source_records: List[MemoryRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Normalized model output used by the parser and evaluator."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    raw_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepEvaluation:
    """Per-step diagnostic emitted by symbolic validation."""

    step_index: int
    action_signature: str
    legal: bool
    grounding_error: bool
    preconditions_satisfied: bool
    goal_satisfied_after_step: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class EvaluationBreakdown:
    """Experiment metrics focused on symbolic action validity."""

    plan_accuracy: float
    exact_match: bool
    precondition_violations: int
    ordering_violations: int
    grounding_errors: int
    valid_plan: bool
    goal_satisfied: bool
    correct_but_suboptimal: bool = False
    executed_steps: int = 0
    failure_reason: str | None = None
    step_diagnostics: List[StepEvaluation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def total_dependency_failures(self) -> int:
        return (
            self.precondition_violations
            + self.ordering_violations
            + self.grounding_errors
        )


def build_memory_records(episodes: Sequence[TraceEpisode]) -> List[MemoryRecord]:
    """Convert replay episodes into memory records.

    Sets token_count=0 so that memory policies use their injected
    TokenCounter for accurate BPE counting (fix Q1).
    """
    return [
        MemoryRecord(
            step_index=episode.step_index,
            text=episode.text,
            token_count=0,
        )
        for episode in episodes
    ]
