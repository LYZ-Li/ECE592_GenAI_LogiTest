"""Persistent gold-plan cache for reproducible experiment runs."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.common.contracts import PlanStep, PlanTrace, PlanningTaskInstance
from src.runner.planner import PlannerBackend, SymbolicPlanExecutor

logger = logging.getLogger("memory_compression.plan_cache")


def task_hash(task: PlanningTaskInstance) -> str:
    """Return a stable hash for fields that define a planning task."""
    payload = {
        "task_id": task.task_id,
        "domain_name": task.domain_name,
        "problem_name": task.problem_name,
        "goal_text": task.goal_text,
        "valid_objects": task.valid_objects,
        "typed_objects": task.typed_objects,
        "initial_facts": task.initial_facts,
        "goal_facts": task.goal_facts,
        "pddl_domain": task.pddl_domain,
        "pddl_problem": task.pddl_problem,
        "metadata": task.metadata,
    }
    return _sha256_json(payload)


def pddl_problem_hash(task: PlanningTaskInstance) -> str:
    """Return a stable hash of the PDDL domain/problem pair."""
    return _sha256_json(
        {"pddl_domain": task.pddl_domain, "pddl_problem": task.pddl_problem}
    )


def ensure_gold_plans(
    dataset: List[PlanningTaskInstance],
    planner: PlannerBackend,
    cache_dir: Path,
    executor: SymbolicPlanExecutor,
    external_validation: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Load or generate valid cached gold plans for every selected task.

    Args:
        dataset: Tasks selected for this run.
        planner: Planner backend used to generate missing/stale gold plans.
        cache_dir: Directory containing one JSON cache file per task.
        executor: Symbolic executor for local gold-plan validation.
        external_validation: Whether to attempt optional external validation.

    Returns:
        Mapping from task_id to plan cache metadata. Each value contains a
        ``plan`` key with a PlanTrace and ``record`` with the JSON metadata.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    plans: Dict[str, Dict[str, Any]] = {}
    for task in dataset:
        cached = load_cached_gold_plan(task, cache_dir, executor)
        if cached is not None:
            cached["cache_status"] = "reused"
            plans[task.task_id] = cached
            continue

        logger.info("Generating gold plan for %s", task.task_id)
        plan = planner.solve(task)
        record = build_cache_record(
            task=task,
            plan=plan,
            executor=executor,
            planner=planner,
            external_validation=external_validation,
        )
        if record["validation_status"] != "valid":
            raise RuntimeError(
                f"Generated gold plan for {task.task_id} is invalid: "
                f"{record['validation_status']}"
            )
        cache_path(cache_dir, task.task_id).write_text(
            json.dumps(record, indent=2, ensure_ascii=False)
        )
        plans[task.task_id] = {
            "plan": plan,
            "record": record,
            "cache_status": "generated",
        }
    return plans


def load_cached_gold_plan(
    task: PlanningTaskInstance,
    cache_dir: Path,
    executor: SymbolicPlanExecutor,
) -> Dict[str, Any] | None:
    """Load a valid cached gold plan, returning None if missing or stale."""
    path = cache_path(cache_dir, task.task_id)
    if not path.exists():
        return None

    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Ignoring corrupt plan cache file: %s", path)
        return None

    if record.get("task_hash") != task_hash(task):
        return None
    if record.get("pddl_problem_hash") != pddl_problem_hash(task):
        return None
    if record.get("validation_status") != "valid":
        return None

    steps = [_step_from_dict(item) for item in record.get("gold_steps", [])]
    if not steps:
        return None

    replay = executor.replay(task, steps)
    if not replay.goal_satisfied or replay.failure_reason is not None:
        return None

    plan = PlanTrace(steps=steps, episodes=replay.episodes)
    return {"plan": plan, "record": record}


def build_cache_record(
    task: PlanningTaskInstance,
    plan: PlanTrace,
    executor: SymbolicPlanExecutor,
    planner: PlannerBackend,
    external_validation: bool = True,
) -> Dict[str, Any]:
    """Build the JSON record written to ``data/plans/<task_id>.json``."""
    replay = executor.replay(task, plan.steps)
    external_status = (
        _external_validate(planner, task, plan.steps)
        if external_validation
        else {"status": "disabled"}
    )
    validation_status = (
        "valid" if replay.goal_satisfied and replay.failure_reason is None else "invalid"
    )
    difficulty = task.metadata.get("difficulty", "unknown")
    return {
        "task_id": task.task_id,
        "difficulty": difficulty,
        "seed": _seed_from_task_id(task.task_id),
        "instance_index": _instance_index_from_task(task),
        "task_hash": task_hash(task),
        "pddl_problem_hash": pddl_problem_hash(task),
        "gold_steps": [_step_to_dict(step) for step in plan.steps],
        "gold_signatures": [step.signature for step in plan.steps],
        "planner_metadata": {
            "backend": planner.__class__.__name__,
            "engine_name": getattr(planner, "engine_name", None),
            "timeout_seconds": getattr(planner, "timeout_seconds", None),
        },
        "validation_status": validation_status,
        "symbolic_validation": {
            "goal_satisfied": replay.goal_satisfied,
            "failure_reason": replay.failure_reason,
            "executed_steps": replay.executed_steps,
            "grounding_errors": replay.grounding_errors,
            "precondition_violations": replay.precondition_violations,
        },
        "external_validation_status": external_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def cache_path(cache_dir: Path, task_id: str) -> Path:
    """Return the cache path for a task id."""
    return cache_dir / f"{task_id}.json"


def _external_validate(
    planner: PlannerBackend,
    task: PlanningTaskInstance,
    steps: List[PlanStep],
) -> Dict[str, Any]:
    validator = getattr(planner, "external_validate", None)
    if validator is None:
        return {"status": "unavailable", "message": "planner has no external validator"}
    try:
        result = validator(task, steps)
    except Exception as exc:  # pragma: no cover - depends on optional planner stack
        return {"status": "error", "message": str(exc)}
    return result


def _step_to_dict(step: PlanStep) -> Dict[str, Any]:
    return asdict(step)


def _step_from_dict(payload: Dict[str, Any]) -> PlanStep:
    return PlanStep(
        index=int(payload["index"]),
        action_name=str(payload["action_name"]),
        arguments=list(payload.get("arguments", [])),
        preconditions=list(payload.get("preconditions", [])),
        effects=list(payload.get("effects", [])),
        notes=dict(payload.get("notes", {})),
    )


def _sha256_json(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _seed_from_task_id(task_id: str) -> int | None:
    parts = task_id.split("-")
    if len(parts) < 4:
        return None
    try:
        return int(parts[-2])
    except ValueError:
        return None


def _instance_index_from_task(task: PlanningTaskInstance) -> int | None:
    if "instance_index" in task.metadata:
        try:
            return int(task.metadata["instance_index"])
        except (TypeError, ValueError):
            return None
    parts = task.task_id.split("-")
    if len(parts) < 4:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None
