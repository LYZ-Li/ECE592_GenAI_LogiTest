"""Experiment orchestrator: stepwise rollout loop and batch runner."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.common.contracts import (
    MemoryContext,
    ModelResponse,
    PlanStep,
    PlanTrace,
    PlanningTaskInstance,
    build_memory_records,
)
from src.eval.check import HardConstraintEvaluator
from src.runner.config import MemoryPolicyConfig, ProposalConfig
from src.runner.engine import ModelBackend, build_model_backend
from src.runner.inference import (
    PromptBuilder,
    StrictActionParser,
    build_incident_history_records,
)
from src.runner.memory import (
    FullContextPolicy,
    MemoryPolicy,
    RecentWindowPolicy,
    SimpleTokenCounter,
    SummarizationPolicy,
    TransformersTokenCounter,
)
from src.runner.plan_cache import ensure_gold_plans
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
    validation_backend: str = "symbolic"
    incident_history_count: int = 0
    enable_symbolic_repair: bool = True
    max_symbolic_repair_attempts: int = 1

    def run_once(
        self,
        task: PlanningTaskInstance,
        prompt_mode: str,
        max_new_tokens: int,
        temperature: float,
        max_rollout_steps: int,
        gold_plan: PlanTrace | None = None,
        condition_name: str = "",
        seed: int = 42,
        enable_parse_recovery: bool = True,
        external_validation: bool = True,
    ) -> dict:
        """Run one task under one condition and prompt mode.

        Returns a dict with task, gold_plan, predicted_steps, evaluation, etc.
        """
        if gold_plan is None:
            gold_plan = self.planner.solve(task)
        gold_trace = self.trace_builder.build_history(task, gold_plan)

        predicted_steps: List[PlanStep] = []
        predicted_episodes = []
        current_state = self.executor.initial_state(task)
        latest_response = None
        parse_error: str | None = None
        parse_error_count = 0
        repair_attempt_count = 0
        symbolic_repair_attempt_count = 0
        symbolic_repair_success_count = 0
        symbolic_failure_count = 0
        terminal_symbolic_replay = None
        trace_records: List[Dict[str, Any]] = []

        while len(predicted_steps) < max_rollout_steps:
            if self.executor.goal_satisfied(task, current_state):
                break

            incident_records = build_incident_history_records(
                task, seed, self.incident_history_count
            )
            incident_context = "\n\n".join(record.text for record in incident_records)
            history = build_memory_records(predicted_episodes)
            memory_context = self.memory_policy.prepare_context(
                history, task.goal_text
            )
            step_index = len(predicted_steps)
            expected_gold_step = (
                gold_plan.steps[step_index]
                if step_index < len(gold_plan.steps)
                else None
            )
            next_step: PlanStep | None = None
            replay = None

            attempts = [
                {
                    "attempt_type": "primary",
                    "mode": prompt_mode,
                    "repair_attempted": False,
                    "repair_context": None,
                    "symbolic_repair_attempt": 0,
                }
            ]

            while attempts:
                attempt = attempts.pop(0)
                attempt_mode = attempt["mode"]
                system_prompt = self.prompt_builder.system_prompt(attempt_mode)
                user_prompt = self.prompt_builder.user_prompt(
                    task=task,
                    memory_context=memory_context.prompt_context,
                    prompt_mode=attempt_mode,
                    current_state_facts=current_state,
                    incident_context=incident_context,
                    repair_context=attempt.get("repair_context"),
                )
                latest_response = self.model.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                attempt_parse_error: str | None = None
                parsed_step: PlanStep | None = None
                attempt_replay = None
                try:
                    parsed_step = self.parser.parse(
                        latest_response.text, mode=attempt_mode
                    )
                    parsed_step.index = step_index
                    parsed_step = self.executor.annotate_step(task, parsed_step)
                    attempt_replay = self.executor.replay(
                        task, [*predicted_steps, parsed_step]
                    )
                except ValueError as exc:
                    attempt_parse_error = str(exc)
                    parse_error_count += 1

                trace_records.append(
                    _build_trace_record(
                        task=task,
                        condition=condition_name or self.memory_policy.name,
                        mode=prompt_mode,
                        seed=seed,
                        step_index=step_index,
                        attempt_type=attempt["attempt_type"],
                        repair_attempted=attempt["repair_attempted"],
                        memory_policy=self.memory_policy.name,
                        memory_context=memory_context,
                        current_state=current_state,
                        expected_gold_step=expected_gold_step,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response=latest_response,
                        parsed_step=parsed_step,
                        parse_error=attempt_parse_error,
                        replay=attempt_replay,
                        max_new_tokens=max_new_tokens,
                    )
                )

                symbolic_error = _symbolic_error(attempt_replay)
                if parsed_step is not None and symbolic_error:
                    symbolic_failure_count += 1
                    terminal_symbolic_replay = attempt_replay
                    if (
                        self.enable_symbolic_repair
                        and attempt["symbolic_repair_attempt"]
                        < self.max_symbolic_repair_attempts
                    ):
                        repair_attempt_count += 1
                        symbolic_repair_attempt_count += 1
                        attempts.append(
                            {
                                "attempt_type": "symbolic_repair",
                                "mode": "direct_action",
                                "repair_attempted": True,
                                "repair_context": _repair_context(
                                    parsed_step, attempt_replay
                                ),
                                "symbolic_repair_attempt": (
                                    attempt["symbolic_repair_attempt"] + 1
                                ),
                            }
                        )
                        continue

                if parsed_step is not None and not symbolic_error:
                    next_step = parsed_step
                    replay = attempt_replay
                    parse_error = None
                    if attempt["attempt_type"] == "symbolic_repair":
                        symbolic_repair_success_count += 1
                        terminal_symbolic_replay = None
                    break

                parse_error = attempt_parse_error
                if (
                    prompt_mode == "cot"
                    and enable_parse_recovery
                    and attempt["attempt_type"] == "primary"
                ):
                    repair_attempt_count += 1
                    attempts.append(
                        {
                            "attempt_type": "parse_recovery",
                            "mode": "direct_action",
                            "repair_attempted": True,
                            "repair_context": None,
                            "symbolic_repair_attempt": 0,
                        }
                    )

            if next_step is None or replay is None:
                break

            candidate_steps = [*predicted_steps, next_step]
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
        if terminal_symbolic_replay is not None:
            evaluation.failure_reason = terminal_symbolic_replay.failure_reason
            evaluation.valid_plan = False
            evaluation.goal_satisfied = False
            evaluation.correct_but_suboptimal = False
            evaluation.precondition_violations = max(
                evaluation.precondition_violations,
                terminal_symbolic_replay.precondition_violations,
            )
            evaluation.grounding_errors = max(
                evaluation.grounding_errors,
                terminal_symbolic_replay.grounding_errors,
            )
            evaluation.step_diagnostics = terminal_symbolic_replay.step_diagnostics
            evaluation.notes.append(
                "Rollout stopped after symbolic repair failed or was unavailable."
            )

        external_validation_status = (
            _external_validate(self.planner, task, predicted_steps)
            if external_validation
            else {"status": "disabled"}
        )
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
            "trace_records": trace_records,
            "parse_error_count": parse_error_count,
            "repair_attempt_count": repair_attempt_count,
            "symbolic_repair_attempt_count": symbolic_repair_attempt_count,
            "symbolic_repair_success_count": symbolic_repair_success_count,
            "symbolic_failure_count": symbolic_failure_count,
            "validation_backend": self.validation_backend,
            "external_validation_status": external_validation_status,
        }


def _build_trace_record(
    task: PlanningTaskInstance,
    condition: str,
    mode: str,
    seed: int,
    step_index: int,
    attempt_type: str,
    repair_attempted: bool,
    memory_policy: str,
    memory_context: MemoryContext,
    current_state: List[str],
    expected_gold_step: PlanStep | None,
    system_prompt: str,
    user_prompt: str,
    response: ModelResponse,
    parsed_step: PlanStep | None,
    parse_error: str | None,
    replay: Any,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Build a full per-model-call trace record."""
    symbolic_error = _symbolic_error(replay)
    missing_preconditions = _missing_preconditions(replay)
    return {
        "task_id": task.task_id,
        "difficulty": task.metadata.get("difficulty", "unknown"),
        "instance_index": task.metadata.get("instance_index"),
        "condition": condition,
        "mode": mode,
        "seed": seed,
        "step_index": step_index,
        "attempt_type": attempt_type,
        "repair_attempted": repair_attempted,
        "memory_policy": memory_policy,
        "memory_context_metadata": memory_context.metadata,
        "memory_context": memory_context.prompt_context,
        "current_state": current_state,
        "expected_gold_action": _step_to_record(expected_gold_step),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_output": response.raw_output,
        "model_text": response.text,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "finish_reason": response.metadata.get("finish_reason"),
        "response_metadata": response.metadata,
        "is_truncated": _response_is_truncated(response, max_new_tokens),
        "parsed_action": _step_to_record(parsed_step),
        "parse_error": parse_error,
        "symbolic_error": symbolic_error,
        "missing_preconditions": missing_preconditions,
        "replay_result": _replay_to_record(replay),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _symbolic_error(replay: Any) -> str | None:
    """Return the replay failure reason when an attempted action is symbolic invalid."""
    if replay is None:
        return None
    if replay.failure_reason in {"precondition_violation", "grounding_error"}:
        return replay.failure_reason
    return None


def _missing_preconditions(replay: Any) -> List[str]:
    """Extract missing preconditions from replay diagnostics."""
    if replay is None or not replay.step_diagnostics:
        return []
    notes = replay.step_diagnostics[-1].notes
    for note in notes:
        prefix = "Missing preconditions: "
        if note.startswith(prefix):
            return [
                item.strip()
                for item in note.removeprefix(prefix).split(";")
                if item.strip()
            ]
    return []


def _repair_context(step: PlanStep, replay: Any) -> str:
    """Build symbolic feedback for one replacement-action retry."""
    lines = [
        f"Rejected action: {step.signature}",
        f"Symbolic error: {_symbolic_error(replay) or 'unknown'}",
    ]
    missing = _missing_preconditions(replay)
    if missing:
        lines.append("Missing preconditions: " + "; ".join(missing))
    elif replay is not None and replay.step_diagnostics:
        notes = replay.step_diagnostics[-1].notes
        if notes:
            lines.append("Validator notes: " + "; ".join(notes))
    lines.append(
        "Choose a replacement action whose requirements are already satisfied by Current State."
    )
    return "\n".join(lines)


def _step_to_record(step: PlanStep | None) -> Dict[str, Any] | None:
    if step is None:
        return None
    return {
        "index": step.index,
        "action_name": step.action_name,
        "arguments": step.arguments,
        "signature": step.signature,
        "preconditions": step.preconditions,
        "effects": step.effects,
        "notes": step.notes,
    }


def _replay_to_record(replay: Any) -> Dict[str, Any] | None:
    if replay is None:
        return None
    return {
        "final_state": replay.final_state,
        "valid_plan": replay.valid_plan,
        "goal_satisfied": replay.goal_satisfied,
        "executed_steps": replay.executed_steps,
        "grounding_errors": replay.grounding_errors,
        "precondition_violations": replay.precondition_violations,
        "failure_reason": replay.failure_reason,
        "step_diagnostics": [
            asdict(item) for item in replay.step_diagnostics
        ],
    }


def _response_is_truncated(response: ModelResponse, max_new_tokens: int) -> bool:
    finish_reason = str(response.metadata.get("finish_reason", "")).lower()
    return finish_reason == "length" or response.completion_tokens >= max_new_tokens


def _external_validate(
    planner: PlannerBackend,
    task: PlanningTaskInstance,
    predicted_steps: List[PlanStep],
) -> Dict[str, Any]:
    validator = getattr(planner, "external_validate", None)
    if validator is None:
        return {"status": "unavailable", "message": "planner has no external validator"}
    try:
        return validator(task, predicted_steps)
    except Exception as exc:  # pragma: no cover - depends on optional validator stack
        return {"status": "error", "message": str(exc)}


def build_memory_policy(
    policy_config: MemoryPolicyConfig,
    model_name_or_path: str,
    backend_type: str = "transformers",
) -> MemoryPolicy:
    """Factory: create a MemoryPolicy from config.

    Args:
        policy_config: Memory policy configuration.
        model_name_or_path: Model name for tokenizer initialization.
        backend_type: Backend type — controls token counter selection.

    Returns:
        Configured MemoryPolicy instance.

    Raises:
        ValueError: If policy type is unsupported.
    """
    if policy_config.type == "full":
        return FullContextPolicy(name=policy_config.name)

    if backend_type in ("anthropic_api", "openai_compatible"):
        token_counter = SimpleTokenCounter()
    else:
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
    shared_model: ModelBackend | None = None,
) -> ProposalExperimentRunner:
    """Build a fully-wired experiment runner for one condition.

    Args:
        config: Full experiment configuration.
        policy_config: Memory policy for this condition.
        task: The current task (used to derive valid action names).
        shared_model: Optional pre-built model backend to reuse across trials.
    """
    model_name = (
        config.model.tokenizer_name_or_path or config.model.name_or_path
    )
    memory_policy = build_memory_policy(
        policy_config, model_name, backend_type=config.model.backend_type
    )
    executor = SymbolicPlanExecutor()

    # Fix Q13: derive valid_action_names from task metadata
    valid_action_names = list(
        task.metadata.get("action_schemas", {}).keys()
    )

    model = shared_model or build_model_backend(config.model)

    return ProposalExperimentRunner(
        planner=FastDownwardPlannerBackend(
            engine_name=config.planner.engine_name,
            timeout_seconds=config.planner.timeout_seconds,
            executor=executor,
        ),
        memory_policy=memory_policy,
        model=model,
        prompt_builder=PromptBuilder(
            context_padding=config.dataset.include_context_padding,
            padding_seed=config.experiment.seed,
        ),
        parser=StrictActionParser(valid_action_names=valid_action_names),
        evaluator=HardConstraintEvaluator(executor=executor),
        trace_builder=OracleTraceBuilder(executor=executor),
        executor=executor,
        incident_history_count=(
            config.dataset.incident_history_count
            if config.dataset.include_incident_history
            else 0
        ),
        enable_symbolic_repair=config.evaluation.enable_symbolic_repair,
        max_symbolic_repair_attempts=config.evaluation.max_symbolic_repair_attempts,
    )


def model_output_dir(base_output_dir: Path, config: ProposalConfig) -> Path:
    """Return the model-scoped output directory for an experiment run."""
    raw_label = (
        config.model.label
        or config.model.api_model
        or config.model.name_or_path
        or "unknown-model"
    )
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_label).strip("-._").lower()
    return base_output_dir / (label or "unknown-model")


def _serialize_result(
    task: PlanningTaskInstance,
    policy_config: MemoryPolicyConfig,
    mode: str,
    seed: int,
    result: dict,
    trace_file: str = "traces.jsonl",
) -> Dict[str, Any]:
    """Serialize a single trial result for JSONL output."""
    evaluation = result["evaluation"]
    response = result["response"]
    return {
        "task_id": task.task_id,
        "difficulty": task.metadata.get("difficulty", "unknown"),
        "instance_index": task.metadata.get("instance_index"),
        "condition": policy_config.name,
        "mode": mode,
        "seed": seed,
        "gold_steps": [
            {"action_name": s.action_name, "arguments": s.arguments}
            for s in result["gold_plan"].steps
        ],
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
        "parse_error_count": result.get("parse_error_count", 0),
        "repair_attempt_count": result.get("repair_attempt_count", 0),
        "symbolic_repair_attempt_count": result.get(
            "symbolic_repair_attempt_count", 0
        ),
        "symbolic_repair_success_count": result.get(
            "symbolic_repair_success_count", 0
        ),
        "symbolic_failure_count": result.get("symbolic_failure_count", 0),
        "validation_backend": result.get("validation_backend", "symbolic"),
        "external_validation_status": result.get(
            "external_validation_status", {"status": "not_run"}
        ),
        "trace_file": trace_file,
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
    model_dir = model_output_dir(output_dir, config)
    results_file = model_dir / "results" / "results.jsonl"
    traces_file = model_dir / "traces" / "traces.jsonl"
    completed = _load_completed_keys(results_file)
    total = len(dataset) * len(config.memory_policies) * len(config.experiment.prompt_modes)
    selected_keys = {
        (task.task_id, policy.name, mode)
        for task in dataset
        for policy in config.memory_policies
        for mode in config.experiment.prompt_modes
    }
    done = len(completed & selected_keys)

    logger.info(
        "Starting batch: %d tasks x %d conditions x %d modes = %d trials (%d already done)",
        len(dataset),
        len(config.memory_policies),
        len(config.experiment.prompt_modes),
        total,
        done,
    )

    plan_executor = SymbolicPlanExecutor()
    plan_planner = FastDownwardPlannerBackend(
        engine_name=config.planner.engine_name,
        timeout_seconds=config.planner.timeout_seconds,
        executor=plan_executor,
    )
    gold_plans = ensure_gold_plans(
        dataset=dataset,
        planner=plan_planner,
        cache_dir=Path(config.planner.plan_cache_dir),
        executor=plan_executor,
        external_validation=config.planner.external_validation,
    )

    # Build the model backend once and share across all trials.
    shared_model = build_model_backend(config.model)

    for task in dataset:
        for policy_config in config.memory_policies:
            for mode in config.experiment.prompt_modes:
                key = (task.task_id, policy_config.name, mode)
                if key in completed:
                    continue

                logger.info("Running: %s / %s / %s", *key)
                runner = _build_runner(config, policy_config, task, shared_model=shared_model)

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
                    gold_plan=gold_plans[task.task_id]["plan"],
                    condition_name=policy_config.name,
                    seed=config.experiment.seed,
                    enable_parse_recovery=config.evaluation.enable_parse_recovery,
                    external_validation=config.planner.external_validation,
                )

                record = _serialize_result(
                    task,
                    policy_config,
                    mode,
                    config.experiment.seed,
                    result,
                    trace_file=str(traces_file),
                )
                for trace_record in result.get("trace_records", []):
                    _append_jsonl(traces_file, trace_record)
                _append_jsonl(results_file, record)
                runner.model.clear_memory()
                done += 1
                logger.info("Completed %d/%d", done, total)

    logger.info("Batch complete: %d/%d trials in %s", done, total, results_file)
