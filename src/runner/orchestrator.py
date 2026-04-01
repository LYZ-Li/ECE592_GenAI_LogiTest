"""Experiment orchestrator: stepwise rollout loop and batch runner."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.common.contracts import (
    PlanStep,
    PlanTrace,
    PlanningTaskInstance,
    build_memory_records,
)
from src.eval.check import HardConstraintEvaluator
from src.runner.config import MemoryPolicyConfig, ProposalConfig
from src.runner.engine import ModelBackend, TransformersQwenBackend
from src.runner.inference import PromptBuilder, StrictActionParser
from src.runner.memory import (
    FullContextPolicy,
    MemoryPolicy,
    RecentWindowPolicy,
    SummarizationPolicy,
    TransformersTokenCounter,
)
from src.runner.planner import (
    FastDownwardPlannerBackend,
    OracleTraceBuilder,
    PlannerBackend,
    SymbolicPlanExecutor,
)

logger = logging.getLogger("memory_compression.orchestrator")


@dataclass
class ProposalExperimentRunner:
    """Wire planner, memory policy, model, parser, and evaluator together."""

    planner: PlannerBackend
    memory_policy: MemoryPolicy
    model: ModelBackend
    prompt_builder: PromptBuilder
    parser: StrictActionParser
    evaluator: HardConstraintEvaluator
    trace_builder: OracleTraceBuilder
    executor: SymbolicPlanExecutor = field(default_factory=SymbolicPlanExecutor)

    def run_once(
        self,
        task: PlanningTaskInstance,
        prompt_mode: str,
        max_new_tokens: int,
        temperature: float,
        max_rollout_steps: int,
    ) -> dict:
        """Run one task under one condition and prompt mode.

        Returns a dict with task, gold_plan, predicted_steps, evaluation, etc.
        """
        gold_plan = self.planner.solve(task)
        gold_trace = self.trace_builder.build_history(task, gold_plan)

        predicted_steps: List[PlanStep] = []
        predicted_episodes = []
        current_state = self.executor.initial_state(task)
        latest_response = None
        parse_error: str | None = None

        while len(predicted_steps) < max_rollout_steps:
            if self.executor.goal_satisfied(task, current_state):
                break

            history = build_memory_records(predicted_episodes)
            memory_context = self.memory_policy.prepare_context(
                history, task.goal_text
            )
            latest_response = self.model.generate(
                system_prompt=self.prompt_builder.system_prompt(prompt_mode),
                user_prompt=self.prompt_builder.user_prompt(
                    task=task,
                    memory_context=memory_context.prompt_context,
                    prompt_mode=prompt_mode,
                    current_state_facts=current_state,
                ),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            try:
                next_step = self.parser.parse(
                    latest_response.text, mode=prompt_mode
                )
            except ValueError as exc:
                parse_error = str(exc)
                break

            next_step.index = len(predicted_steps)
            candidate_steps = [*predicted_steps, next_step]
            replay = self.executor.replay(task, candidate_steps)
            predicted_steps = candidate_steps
            predicted_episodes = replay.episodes
            current_state = replay.final_state
            if replay.failure_reason:
                break

        evaluation = self.evaluator.evaluate(
            task, gold_plan.steps, predicted_steps
        )
        if parse_error:
            evaluation.failure_reason = "parse_error"
            evaluation.valid_plan = False
            evaluation.goal_satisfied = False
            evaluation.correct_but_suboptimal = False
            evaluation.notes.append(parse_error)

        final_history = build_memory_records(predicted_episodes)
        final_memory_context = self.memory_policy.prepare_context(
            final_history, task.goal_text
        )
        return {
            "task": task,
            "gold_plan": gold_plan,
            "gold_trace": gold_trace,
            "predicted_steps": predicted_steps,
            "predicted_trace": predicted_episodes,
            "memory_context": final_memory_context,
            "response": latest_response,
            "evaluation": evaluation,
        }


def build_memory_policy(
    policy_config: MemoryPolicyConfig, model_name_or_path: str
) -> MemoryPolicy:
    """Factory: create a MemoryPolicy from config.

    Args:
        policy_config: Memory policy configuration.
        model_name_or_path: Model name for tokenizer initialization.

    Returns:
        Configured MemoryPolicy instance.

    Raises:
        ValueError: If policy type is unsupported.
    """
    if policy_config.type == "full":
        return FullContextPolicy(name=policy_config.name)

    token_counter = TransformersTokenCounter(
        model_name_or_path=model_name_or_path
    )

    if policy_config.type == "truncation":
        if policy_config.max_context_tokens is None:
            raise ValueError(
                f"Truncation policy '{policy_config.name}' requires 'max_context_tokens'."
            )
        return RecentWindowPolicy(
            max_context_tokens=policy_config.max_context_tokens,
            token_counter=token_counter,
            name=policy_config.name,
        )

    if policy_config.type == "summarization":
        if policy_config.max_context_tokens is None:
            raise ValueError(
                f"Summarization policy '{policy_config.name}' requires 'max_context_tokens'."
            )
        return SummarizationPolicy(
            max_context_tokens=policy_config.max_context_tokens,
            recent_window_tokens=policy_config.recent_window_tokens,
            token_counter=token_counter,
            summary_model=policy_config.summary_model,
            name=policy_config.name,
        )

    raise ValueError(
        f"Unsupported memory policy type '{policy_config.type}'. "
        "Supported: full, truncation, summarization."
    )


def _build_runner(
    config: ProposalConfig,
    policy_config: MemoryPolicyConfig,
    task: PlanningTaskInstance,
) -> ProposalExperimentRunner:
    """Build a fully-wired experiment runner for one condition."""
    model_name = (
        config.model.tokenizer_name_or_path or config.model.name_or_path
    )
    memory_policy = build_memory_policy(policy_config, model_name)
    executor = SymbolicPlanExecutor()

    # Fix Q13: derive valid_action_names from task metadata
    valid_action_names = list(
        task.metadata.get("action_schemas", {}).keys()
    )

    return ProposalExperimentRunner(
        planner=FastDownwardPlannerBackend(
            engine_name=config.planner.engine_name,
            timeout_seconds=config.planner.timeout_seconds,
            executor=executor,
        ),
        memory_policy=memory_policy,
        model=TransformersQwenBackend(
            model_name_or_path=config.model.name_or_path,
            tokenizer_name_or_path=config.model.tokenizer_name_or_path,
            quantization=config.model.quantization,
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
        ),
        prompt_builder=PromptBuilder(),
        parser=StrictActionParser(valid_action_names=valid_action_names),
        evaluator=HardConstraintEvaluator(executor=executor),
        trace_builder=OracleTraceBuilder(executor=executor),
        executor=executor,
    )


def _serialize_result(
    task: PlanningTaskInstance,
    policy_config: MemoryPolicyConfig,
    mode: str,
    seed: int,
    result: dict,
) -> Dict[str, Any]:
    """Serialize a single trial result for JSONL output."""
    evaluation = result["evaluation"]
    response = result["response"]
    return {
        "task_id": task.task_id,
        "condition": policy_config.name,
        "mode": mode,
        "seed": seed,
        "predicted_steps": [
            {"action_name": s.action_name, "arguments": s.arguments}
            for s in result["predicted_steps"]
        ],
        "plan_accuracy": evaluation.plan_accuracy,
        "exact_match": evaluation.exact_match,
        "precondition_violations": evaluation.precondition_violations,
        "ordering_violations": evaluation.ordering_violations,
        "grounding_errors": evaluation.grounding_errors,
        "valid_plan": evaluation.valid_plan,
        "goal_satisfied": evaluation.goal_satisfied,
        "correct_but_suboptimal": evaluation.correct_but_suboptimal,
        "executed_steps": evaluation.executed_steps,
        "failure_reason": evaluation.failure_reason,
        "prompt_tokens": response.prompt_tokens if response else 0,
        "completion_tokens": response.completion_tokens if response else 0,
        "raw_output": response.raw_output if response else "",
        "notes": evaluation.notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _load_completed_keys(results_file: Path) -> Set[Tuple[str, str, str]]:
    """Load already-completed (task_id, condition, mode) keys from JSONL."""
    completed: Set[Tuple[str, str, str]] = set()
    if not results_file.exists():
        return completed
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add(
                    (record["task_id"], record["condition"], record["mode"])
                )
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def _append_jsonl(results_file: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON record to the results file (crash-safe)."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run_batch(
    config: ProposalConfig,
    dataset: List[PlanningTaskInstance],
    output_dir: Path,
) -> None:
    """Run the full experiment: all tasks x conditions x modes.

    Writes results incrementally to JSONL. On restart, completed
    trials are skipped (checkpointing via skip-if-exists).

    Args:
        config: Full experiment configuration.
        dataset: List of generated task instances.
        output_dir: Directory for results output.
    """
    results_file = output_dir / "results.jsonl"
    completed = _load_completed_keys(results_file)
    total = len(dataset) * len(config.memory_policies) * len(config.experiment.prompt_modes)
    done = len(completed)

    logger.info(
        "Starting batch: %d tasks x %d conditions x %d modes = %d trials (%d already done)",
        len(dataset),
        len(config.memory_policies),
        len(config.experiment.prompt_modes),
        total,
        done,
    )

    for task in dataset:
        for policy_config in config.memory_policies:
            for mode in config.experiment.prompt_modes:
                key = (task.task_id, policy_config.name, mode)
                if key in completed:
                    continue

                logger.info("Running: %s / %s / %s", *key)
                runner = _build_runner(config, policy_config, task)

                max_rollout = (
                    config.dataset.max_plan_steps
                    * config.evaluation.max_rollout_steps_multiplier
                )
                result = runner.run_once(
                    task=task,
                    prompt_mode=mode,
                    max_new_tokens=config.model.max_new_tokens,
                    temperature=config.model.temperature,
                    max_rollout_steps=max_rollout,
                )

                record = _serialize_result(
                    task, policy_config, mode, config.experiment.seed, result
                )
                _append_jsonl(results_file, record)
                runner.model.clear_memory()
                done += 1
                logger.info("Completed %d/%d", done, total)

    logger.info("Batch complete: %d/%d trials in %s", done, total, results_file)
