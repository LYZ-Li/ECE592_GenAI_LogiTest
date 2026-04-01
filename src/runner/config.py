"""Configuration loader for the memory compression study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DatasetConfig:
    num_instances: int = 40
    domain_family: str = "household_logistics"
    difficulty_levels: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    max_plan_steps: int = 18
    max_objects_per_type: int = 8
    include_distractors: bool = True
    default_difficulty: str = "medium"


@dataclass
class PlannerConfig:
    backend: str = "fast_downward"
    use_unified_planning: bool = True
    validate_candidates: bool = True
    engine_name: str = "fast-downward"
    timeout_seconds: int = 30


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-8B"
    tokenizer_name_or_path: Optional[str] = None
    quantization: str = "4bit"
    max_new_tokens: int = 256
    temperature: float = 0.0
    device_map: str = "auto"
    enable_thinking: bool = False
    strict_json_output: bool = True
    trust_remote_code: bool = False


@dataclass
class MemoryPolicyConfig:
    name: str
    type: str
    max_context_tokens: Optional[int] = None
    recent_window_tokens: int = 1024
    summary_model: Optional[str] = None


@dataclass
class EvaluationConfig:
    require_parseable_action_block: bool = True
    allow_alternative_valid_plans: bool = True
    compute_precondition_violations: bool = True
    compute_ordering_violations: bool = True
    compute_grounding_errors: bool = True
    max_rollout_steps_multiplier: int = 2


@dataclass
class ExperimentConfig:
    name: str = "memory_compression_study"
    seed: int = 42
    output_dir: str = "results/"
    prompt_modes: List[str] = field(default_factory=lambda: ["direct_action", "cot"])
    save_intermediate_artifacts: bool = True


@dataclass
class ProposalConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    memory_policies: List[MemoryPolicyConfig] = field(default_factory=list)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(path: str | Path) -> ProposalConfig:
    """Load experiment config from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed ProposalConfig with validated memory policy settings.

    Raises:
        ValueError: If a memory policy config is missing required fields.
    """
    raw_path = Path(path)
    raw = {}
    if raw_path.exists():
        with open(raw_path) as handle:
            raw = yaml.safe_load(handle) or {}

    config = ProposalConfig()
    config.experiment = ExperimentConfig(**raw.get("experiment", {}))
    config.dataset = DatasetConfig(**raw.get("dataset", {}))
    config.planner = PlannerConfig(**raw.get("planner", {}))
    config.model = ModelConfig(**raw.get("model", {}))
    config.memory_policies = [
        MemoryPolicyConfig(**item) for item in raw.get("memory_policies", [])
    ]
    config.evaluation = EvaluationConfig(**raw.get("evaluation", {}))

    _validate_memory_policies(config.memory_policies)
    return config


def _validate_memory_policies(policies: List[MemoryPolicyConfig]) -> None:
    """Check that memory policy configs have required fields for their type."""
    for policy in policies:
        if policy.type == "truncation" and policy.max_context_tokens is None:
            raise ValueError(
                f"Truncation policy '{policy.name}' requires 'max_context_tokens'."
            )
        if policy.type == "summarization":
            if policy.max_context_tokens is None:
                raise ValueError(
                    f"Summarization policy '{policy.name}' requires 'max_context_tokens'."
                )
