"""Canonical multi-model analysis pipeline for the memory compression study.

This module normalizes result and trace logs across model runs, enriches them
with publication-safe derived metrics, and exports canonical tables used by the
report and figure build.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CONDITION_ORDER = [
    "full_context",
    "truncation_1024",
    "truncation_2048",
    "summarization_1024",
    "summarization_2048",
]
MODE_ORDER = ["direct_action", "cot"]
DIFFICULTY_ORDER = ["easy", "medium", "hard", "very_hard", "extreme"]
ATTEMPT_TYPE_ORDER = {"primary": 0, "parse_recovery": 1, "symbolic_repair": 2}
KNOWN_ACTIONS = {
    "calibrate_gripper",
    "clean_gripper",
    "clear_obstruction",
    "drop",
    "inspect_gripper",
    "inspect_package",
    "move",
    "pickup",
    "regrasp",
    "verify_delivery",
    "verify_grasp",
}


@dataclass(frozen=True)
class ModelRun:
    """A discovered result file together with trace metadata."""

    label: str
    model_name: str
    results_path: Path
    traces_path: Path
    schema_class: str


@dataclass
class CanonicalBundle:
    """In-memory canonical outputs."""

    plans: list[dict[str, Any]]
    trials: list[dict[str, Any]]
    attempt_coverage: list[dict[str, Any]]
    step_diagnostics: list[dict[str, Any]]
    case_studies: dict[str, Any]
    summary: dict[str, Any]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_difficulty(task_id: str) -> str:
    """Extract difficulty from a task id like ``logistics-easy-42-0``."""
    parts = task_id.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def trace_path_for_results(results_path: str | Path) -> Path:
    """Infer the sibling trace path for a result file."""
    path = Path(results_path)
    if (
        len(path.parts) >= 4
        and path.parts[-2] == "results"
        and path.parts[-4] == "results"
    ):
        return path.parents[1] / "traces" / "traces.jsonl"
    return path.parent / "traces.jsonl"


def discover_model_runs(root: str | Path = "results") -> list[ModelRun]:
    """Discover result files from canonical and legacy layouts."""
    root_path = Path(root)
    runs: list[ModelRun] = []
    seen: set[Path] = set()

    for results_path in sorted(root_path.glob("*/results/results.jsonl")):
        model_dir = results_path.parents[1]
        runs.append(
            ModelRun(
                label=model_dir.name,
                model_name=model_dir.name,
                results_path=results_path,
                traces_path=model_dir / "traces" / "traces.jsonl",
                schema_class="model_scoped",
            )
        )
        seen.add(results_path.resolve())

    for results_path in sorted(root_path.glob("*/results.jsonl")):
        resolved = results_path.resolve()
        if resolved in seen:
            continue
        model_dir = results_path.parent
        runs.append(
            ModelRun(
                label=f"{model_dir.name} (legacy)",
                model_name=model_dir.name,
                results_path=results_path,
                traces_path=model_dir / "traces.jsonl",
                schema_class="legacy_flat",
            )
        )
        seen.add(resolved)

    top_level = root_path / "results.jsonl"
    if top_level.exists() and top_level.resolve() not in seen:
        runs.append(
            ModelRun(
                label="top-level legacy",
                model_name="top-level legacy",
                results_path=top_level,
                traces_path=root_path / "traces.jsonl",
                schema_class="top_level_legacy",
            )
        )

    return runs


def load_plan_catalog(plans_dir: str | Path = "data/plans") -> list[dict[str, Any]]:
    """Load canonical plan metadata from ``data/plans``."""
    plans: list[dict[str, Any]] = []
    for path in sorted(Path(plans_dir).glob("*.json")):
        payload = json.loads(path.read_text())
        gold_steps = _ensure_step_list(payload.get("gold_steps"))
        plans.append(
            {
                "task_id": payload.get("task_id"),
                "difficulty": payload.get("difficulty")
                or extract_difficulty(str(payload.get("task_id", ""))),
                "seed": payload.get("seed"),
                "instance_index": payload.get("instance_index"),
                "gold_steps": gold_steps,
                "gold_step_count": len(gold_steps),
                "source_file": str(path),
            }
        )
    return plans


def build_canonical_bundle(
    results_root: str | Path = "results",
    plans_dir: str | Path = "data/plans",
) -> CanonicalBundle:
    """Build the canonical data bundle from raw result and trace logs."""
    runs = discover_model_runs(results_root)
    plans = load_plan_catalog(plans_dir)
    plan_index = {plan["task_id"]: plan for plan in plans if plan.get("task_id")}
    canonical_task_by_difficulty = {
        plan["difficulty"]: plan["task_id"] for plan in plans if plan.get("difficulty")
    }

    raw_trials: list[dict[str, Any]] = []
    for run in runs:
        for record in load_jsonl(run.results_path):
            raw_trials.append(
                _normalize_trial_record(
                    record=record,
                    run=run,
                    plan_index=plan_index,
                    canonical_task_by_difficulty=canonical_task_by_difficulty,
                )
            )

    trial_lookup = {
        _trial_key(record): record
        for record in raw_trials
    }
    raw_steps = _load_and_enrich_steps(runs, trial_lookup)
    _attach_trial_trace_metrics(raw_trials, raw_steps)
    attempt_coverage = build_attempt_coverage(
        runs=runs,
        plans=plans,
        trials=raw_trials,
    )
    case_studies = build_case_studies(raw_trials, raw_steps)
    summary = build_summary(raw_trials, attempt_coverage, case_studies)
    return CanonicalBundle(
        plans=plans,
        trials=sorted(raw_trials, key=_trial_sort_key),
        attempt_coverage=sorted(attempt_coverage, key=_coverage_sort_key),
        step_diagnostics=sorted(raw_steps, key=_step_sort_key),
        case_studies=case_studies,
        summary=summary,
    )


def export_bundle(bundle: CanonicalBundle, analysis_dir: str | Path) -> None:
    """Write canonical bundle artifacts under ``analysis/canonical``."""
    output_dir = Path(analysis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trials.csv", bundle.trials)
    _write_csv(output_dir / "attempt_coverage.csv", bundle.attempt_coverage)
    _write_csv(output_dir / "step_diagnostics.csv", bundle.step_diagnostics)
    (output_dir / "case_studies.json").write_text(
        json.dumps(bundle.case_studies, indent=2, ensure_ascii=True)
    )
    (output_dir / "summary.json").write_text(
        json.dumps(bundle.summary, indent=2, ensure_ascii=True)
    )


def build_attempt_coverage(
    runs: list[ModelRun],
    plans: list[dict[str, Any]],
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one row per expected (model, difficulty, condition, mode) cell."""
    coverage_rows: list[dict[str, Any]] = []
    coverage_lookup = {
        (
            record["model"],
            record.get("canonical_task_id") or record.get("task_id"),
            record["condition"],
            record["mode"],
        ): record
        for record in trials
        if record.get("canonical_task_id") or record.get("task_id")
    }

    for run in runs:
        for plan in plans:
            for condition in CONDITION_ORDER:
                for mode in MODE_ORDER:
                    key = (
                        run.model_name,
                        plan["task_id"],
                        condition,
                        mode,
                    )
                    trial = coverage_lookup.get(key)
                    coverage_rows.append(
                        {
                            "model": run.model_name,
                            "label": run.label,
                            "schema_class": run.schema_class,
                            "coverage_bucket": _coverage_bucket(run),
                            "difficulty": plan["difficulty"],
                            "expected_task_id": plan["task_id"],
                            "condition": condition,
                            "mode": mode,
                            "attempt_status": "attempted" if trial else "not_attempted",
                            "observed_task_id": trial.get("task_id") if trial else None,
                            "traces_available": bool(run.traces_path.exists()),
                            "goal_satisfied": trial.get("goal_satisfied") if trial else None,
                            "failure_reason": trial.get("failure_reason") if trial else None,
                            "outcome_bucket": trial.get("outcome_bucket") if trial else "not_attempted",
                        }
                    )
    return coverage_rows


def build_summary(
    trials: list[dict[str, Any]],
    attempt_coverage: list[dict[str, Any]],
    case_studies: dict[str, Any],
) -> dict[str, Any]:
    """Build top-level summary data for the report."""
    model_names = sorted({row["model"] for row in trials})
    canonical_models = sorted(
        {
            row["model"]
            for row in attempt_coverage
            if row["coverage_bucket"] == "canonical"
        }
    )
    legacy_models = sorted(
        {
            row["model"]
            for row in attempt_coverage
            if row["coverage_bucket"] != "canonical"
        }
    )
    attempted_rows = [row for row in attempt_coverage if row["attempt_status"] == "attempted"]
    canonical_attempted = [
        row for row in attempted_rows if row["coverage_bucket"] == "canonical"
    ]
    summary_by_model: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in attempt_coverage}):
        coverage_rows = [row for row in attempt_coverage if row["model"] == model]
        attempted = [row for row in coverage_rows if row["attempt_status"] == "attempted"]
        success = [row for row in attempted if row.get("goal_satisfied") is True]
        failures = Counter(row.get("outcome_bucket", "unknown") for row in attempted)
        summary_by_model.append(
            {
                "model": model,
                "coverage_bucket": coverage_rows[0]["coverage_bucket"] if coverage_rows else "unknown",
                "attempted": len(attempted),
                "expected": len(coverage_rows),
                "attempt_rate": _ratio(len(attempted), len(coverage_rows)),
                "goal_satisfied": len(success),
                "goal_satisfaction_rate": _ratio(len(success), len(attempted)),
                "outcomes": dict(sorted(failures.items())),
            }
        )

    return {
        "models_discovered": len(sorted({row["model"] for row in attempt_coverage})),
        "trial_rows": len(trials),
        "coverage_rows": len(attempt_coverage),
        "attempted_trials_all_models": len(attempted_rows),
        "attempted_trials_canonical_models": len(canonical_attempted),
        "expected_trials_canonical_models": len(
            [row for row in attempt_coverage if row["coverage_bucket"] == "canonical"]
        ),
        "goal_satisfied_all_models": len(
            [row for row in attempted_rows if row.get("goal_satisfied") is True]
        ),
        "goal_satisfied_canonical_models": len(
            [row for row in canonical_attempted if row.get("goal_satisfied") is True]
        ),
        "canonical_models": canonical_models,
        "legacy_models": legacy_models,
        "model_names": model_names,
        "summary_by_model": summary_by_model,
        "case_studies_present": sorted(case_studies.get("cases", {}).keys()),
    }


def build_case_studies(
    trials: list[dict[str, Any]],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select representative cases used in report figures."""
    cases: dict[str, Any] = {}
    cases["qwen_token_exhaustion"] = _build_case(
        steps,
        lambda row: row["model"] == "qwen3.5-9b"
        and row.get("parse_failure_subtype") == "empty_truncated_output",
        "Qwen3.5-9B token exhaustion",
        "Internal reasoning consumed the token budget before visible JSON was emitted.",
    )
    cases["qwen_wrong_key"] = _build_case(
        steps,
        lambda row: row["model"] == "qwen3.5-9b"
        and row.get("parse_failure_subtype") == "schema_wrong_key",
        "Qwen3.5-9B wrong-key schema drift",
        "The reasoning remains correct, but the final object uses `action` instead of `action_name`.",
    )
    cases["gemma_context_overflow"] = _build_case(
        steps,
        lambda row: row["model"] == "gemma4-27b"
        and row.get("parse_failure_subtype") == "context_overflow_truncation",
        "Gemma4-27B context overflow",
        "Prompt history grows until the model truncates mid-sentence on very long tasks.",
    )
    cases["llama3b_step0"] = _build_case(
        steps,
        lambda row: row["model"] == "llama-3.2-3b"
        and row["step_index"] == 0
        and row.get("symbolic_error") == "precondition_violation",
        "LLaMA-3.2-3B step-0 crash",
        "The model outputs valid JSON but skips the mandatory inspection step immediately.",
    )
    cases["llama8b_hallucinated_action"] = _build_case(
        steps,
        lambda row: row["model"] == "llama3-8b"
        and row.get("parse_failure_subtype") == "hallucinated_action_name",
        "LLaMA3-8B hallucinated action",
        "The action schema is structurally correct, but the action name is invented.",
    )
    cases["minimax_precondition_skip"] = _build_case(
        steps,
        lambda row: row["model"] == "minimax-m2.5"
        and row.get("symbolic_error") == "precondition_violation",
        "Minimax-M2.5 skipped preparatory step",
        "A correct action family is chosen before the required inspection fact exists.",
    )
    boundary_case = _build_case(
        steps,
        lambda row: row["model"] == "deepseek-v3.2"
        and row.get("dependency_boundary_violation") is True,
        "DeepSeek truncation boundary violation",
        "A missing prerequisite aligns with a gold prerequisite step outside the visible tail window.",
    )
    if boundary_case:
        boundary_case["strict_boundary_violation"] = True
    else:
        boundary_case = _build_case(
            steps,
            lambda row: row["model"] == "deepseek-v3.2"
            and row.get("task_id") == "logistics-medium-42-0"
            and row.get("condition") == "truncation_2048"
            and row.get("mode") == "cot"
            and row.get("symbolic_error") == "precondition_violation"
            and str(row.get("memory_policy") or "").startswith(("truncation", "summarization")),
            "DeepSeek truncation-era dependency failure",
            "The closest boundary-style case appears under truncation, but the conservative boundary metric remains false because the gold producer step still falls inside the visible tail.",
        )
        if not boundary_case:
            boundary_case = _build_case(
                steps,
                lambda row: row["model"] == "deepseek-v3.2"
                and row.get("symbolic_error") == "precondition_violation"
                and str(row.get("memory_policy") or "").startswith(("truncation", "summarization")),
                "DeepSeek truncation-era dependency failure",
                "The closest boundary-style case appears under truncation, but the conservative boundary metric remains false because the gold producer step still falls inside the visible tail.",
            )
        if boundary_case:
            boundary_case["strict_boundary_violation"] = False
    cases["deepseek_boundary_violation"] = boundary_case
    cases["deepseek_loop"] = _build_loop_case(trials)
    return {
        "cases": cases,
        "overview": _build_overview_case(trials),
    }


def classify_parse_failure(trace_row: dict[str, Any]) -> str | None:
    """Classify the parse failure subtype for one trace row."""
    parse_error = trace_row.get("parse_error")
    if not parse_error:
        return None

    raw_output = str(trace_row.get("raw_output") or "")
    is_truncated = bool(trace_row.get("is_truncated"))
    action_match = re.search(r'"action_name"\s*:\s*"([^"]+)"', raw_output)
    if is_truncated and not raw_output.strip():
        return "empty_truncated_output"
    if '"action"' in raw_output and '"action_name"' not in raw_output:
        return "schema_wrong_key"
    if action_match and action_match.group(1) not in KNOWN_ACTIONS:
        return "hallucinated_action_name"
    if is_truncated and raw_output.strip():
        return "context_overflow_truncation"
    return "schema_noncompliance"


def compute_loop_depth(predicted_steps: Any) -> int:
    """Return the maximum repetition count of any predicted action signature."""
    signatures = [_action_signature(step) for step in _ensure_step_list(predicted_steps)]
    signatures = [signature for signature in signatures if signature]
    if not signatures:
        return 0
    counts = Counter(signatures)
    return max(counts.values())


def _load_and_enrich_steps(
    runs: list[ModelRun],
    trial_lookup: dict[tuple[str, str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load and enrich trace rows across all available trace files."""
    enriched: list[dict[str, Any]] = []
    for run in runs:
        if not run.traces_path.exists():
            continue
        for row in load_jsonl(run.traces_path):
            normalized = dict(row)
            normalized["model"] = run.model_name
            normalized["label"] = run.label
            normalized["schema_class"] = run.schema_class
            normalized["coverage_bucket"] = _coverage_bucket(run)
            normalized["traces_available"] = True
            normalized["task_id"] = str(normalized.get("task_id"))
            normalized["difficulty"] = normalized.get("difficulty") or extract_difficulty(
                normalized["task_id"]
            )
            normalized["parse_failure_subtype"] = classify_parse_failure(normalized)
            normalized["step_index"] = int(normalized.get("step_index", 0) or 0)
            normalized["visible_window_start"] = _visible_window_start(normalized)
            trial = trial_lookup.get(_trial_key(normalized))
            if trial:
                normalized["canonical_task_id"] = trial.get("canonical_task_id")
                normalized["dependency_boundary_missing_facts"] = _boundary_facts(
                    normalized, trial
                )
            else:
                normalized["canonical_task_id"] = None
                normalized["dependency_boundary_missing_facts"] = []
            normalized["dependency_boundary_violation"] = bool(
                normalized["dependency_boundary_missing_facts"]
            )
            enriched.append(normalized)
    return enriched


def _attach_trial_trace_metrics(
    trials: list[dict[str, Any]],
    steps: list[dict[str, Any]],
) -> None:
    """Attach trace-derived fields onto normalized trial rows."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        grouped[_trial_key(step)].append(step)

    for trial in trials:
        key = _trial_key(trial)
        trace_rows = sorted(grouped.get(key, []), key=_step_sort_key)
        trial["traces_available"] = bool(trace_rows) or bool(trial.get("traces_available"))
        trial["trace_row_count"] = len(trace_rows)
        trial["step0_compliance"] = _step0_compliance(trace_rows)
        first_failure = _first_failure(trace_rows, trial)
        trial["first_failure_step"] = first_failure["step"]
        trial["first_failure_type"] = first_failure["type"]
        trial["parse_failure_subtype"] = first_failure["parse_subtype"]
        trial["dependency_boundary_violation"] = any(
            row.get("dependency_boundary_violation") for row in trace_rows
        )
        trial["dependency_boundary_missing_facts"] = sorted(
            {
                fact
                for row in trace_rows
                for fact in row.get("dependency_boundary_missing_facts", [])
            }
        )


def _normalize_trial_record(
    record: dict[str, Any],
    run: ModelRun,
    plan_index: dict[str, dict[str, Any]],
    canonical_task_by_difficulty: dict[str, str],
) -> dict[str, Any]:
    """Normalize one result record into canonical trial shape."""
    task_id = str(record.get("task_id"))
    difficulty = record.get("difficulty") or extract_difficulty(task_id)
    plan = plan_index.get(task_id)
    canonical_task_id = task_id if plan else canonical_task_by_difficulty.get(difficulty)
    gold_steps = list(plan["gold_steps"]) if plan else _ensure_step_list(record.get("gold_steps"))
    predicted_steps = _ensure_step_list(record.get("predicted_steps"))
    gold_step_count = len(gold_steps) if gold_steps else None

    trial: dict[str, Any] = {
        "model": run.model_name,
        "label": run.label,
        "schema_class": run.schema_class,
        "coverage_bucket": _coverage_bucket(run),
        "task_id": task_id,
        "canonical_task_id": canonical_task_id,
        "difficulty": difficulty,
        "instance_index": record.get("instance_index"),
        "condition": record.get("condition"),
        "mode": record.get("mode"),
        "seed": record.get("seed"),
        "attempt_status": "attempted",
        "traces_available": run.traces_path.exists(),
        "plan_accuracy": _float_or_none(record.get("plan_accuracy")),
        "exact_match": _as_bool(record.get("exact_match")),
        "valid_plan": _as_bool(record.get("valid_plan")),
        "goal_satisfied": _as_bool(record.get("goal_satisfied")),
        "failure_reason": record.get("failure_reason"),
        "executed_steps": _int_or_zero(record.get("executed_steps")),
        "gold_step_count": gold_step_count,
        "predicted_step_count": len(predicted_steps),
        "parse_error_count": _int_or_zero(record.get("parse_error_count")),
        "repair_attempt_count": _int_or_zero(record.get("repair_attempt_count")),
        "symbolic_failure_count": _int_or_zero(record.get("symbolic_failure_count")),
        "symbolic_repair_attempt_count": _int_or_zero(
            record.get("symbolic_repair_attempt_count")
        ),
        "symbolic_repair_success_count": _int_or_zero(
            record.get("symbolic_repair_success_count")
        ),
        "precondition_violations": _int_or_zero(record.get("precondition_violations")),
        "ordering_violations": _int_or_zero(record.get("ordering_violations")),
        "grounding_errors": _int_or_zero(record.get("grounding_errors")),
        "prompt_tokens": _int_or_zero(record.get("prompt_tokens")),
        "completion_tokens": _int_or_zero(record.get("completion_tokens")),
        "validation_backend": record.get("validation_backend"),
        "timestamp": record.get("timestamp"),
        "trace_file": record.get("trace_file"),
        "correct_but_suboptimal": _as_bool(record.get("correct_but_suboptimal")),
        "repair_rate_per_step": _ratio(
            _int_or_zero(record.get("repair_attempt_count")),
            _int_or_zero(record.get("executed_steps")),
        ),
        "loop_depth": compute_loop_depth(predicted_steps),
        "predicted_signature_counts": dict(
            Counter(
                signature
                for signature in (_action_signature(step) for step in predicted_steps)
                if signature
            )
        ),
        "notes": record.get("notes") or [],
        "source_file": str(run.results_path),
        "gold_steps": gold_steps,
        "predicted_steps": predicted_steps,
    }
    trial["outcome_bucket"] = _outcome_bucket(trial)
    return trial


def _outcome_bucket(trial: dict[str, Any]) -> str:
    """Map a trial row onto a publication-safe terminal outcome bucket."""
    if trial.get("goal_satisfied"):
        return "success"
    reason = str(trial.get("failure_reason") or "").lower()
    if "parse" in reason:
        return "parse_error"
    if "grounding" in reason:
        return "grounding_error"
    if "precondition" in reason:
        return "precondition_violation"
    if "goal" in reason:
        return "goal_not_satisfied"
    if trial.get("goal_satisfied") is False:
        return "goal_not_satisfied"
    return "unknown_failure"


def _visible_window_start(trace_row: dict[str, Any]) -> int:
    """Approximate the earliest verbatim step visible in memory for this row."""
    step_index = int(trace_row.get("step_index", 0) or 0)
    metadata = trace_row.get("memory_context_metadata") or {}
    policy = str(trace_row.get("memory_policy") or "")
    if policy == "full_context":
        return 0
    if policy.startswith("truncation") or policy == "recent_window":
        record_count = _int_or_zero(metadata.get("record_count"))
        return max(0, step_index - record_count)
    if policy.startswith("summarization") or policy == "summarization":
        recent_count = _int_or_zero(metadata.get("recent_count"))
        if recent_count <= 0:
            recent_count = _int_or_zero(metadata.get("record_count"))
        return max(0, step_index - recent_count)
    return 0


def _boundary_facts(
    trace_row: dict[str, Any],
    trial: dict[str, Any],
) -> list[str]:
    """Return missing preconditions whose gold-producing step is outside the window."""
    if trace_row.get("symbolic_error") != "precondition_violation":
        return []
    policy = str(trace_row.get("memory_policy") or "")
    if not (
        policy.startswith("truncation")
        or policy.startswith("summarization")
        or policy in {"recent_window", "summarization"}
    ):
        return []
    visible_window_start = int(trace_row.get("visible_window_start", 0) or 0)
    if visible_window_start <= 0:
        return []
    producer_steps = _gold_fact_producer_steps(trial.get("gold_steps") or [])
    boundary_facts: list[str] = []
    for fact in trace_row.get("missing_preconditions", []) or []:
        producer_step = producer_steps.get(fact)
        if producer_step is not None and producer_step < visible_window_start:
            boundary_facts.append(fact)
    return boundary_facts


def _gold_fact_producer_steps(gold_steps: list[dict[str, Any]]) -> dict[str, int]:
    """Map persistent facts to the first gold step index that establishes them."""
    producers: dict[str, int] = {}
    for idx, step in enumerate(gold_steps):
        for effect in step.get("effects", []) or []:
            if isinstance(effect, str) and not effect.startswith("not "):
                producers.setdefault(effect, idx)
    return producers


def _first_failure(
    trace_rows: list[dict[str, Any]],
    trial: dict[str, Any],
) -> dict[str, Any]:
    """Return the first failure event for a trial."""
    for row in trace_rows:
        if row.get("parse_error"):
            return {
                "step": row["step_index"],
                "type": "parse_error",
                "parse_subtype": row.get("parse_failure_subtype"),
            }
        if row.get("symbolic_error"):
            return {
                "step": row["step_index"],
                "type": row.get("symbolic_error"),
                "parse_subtype": None,
            }
    if trial.get("goal_satisfied") is False:
        return {
            "step": trial.get("executed_steps"),
            "type": "goal_not_satisfied",
            "parse_subtype": None,
        }
    return {"step": None, "type": None, "parse_subtype": None}


def _step0_compliance(trace_rows: list[dict[str, Any]]) -> bool | None:
    """Whether the primary step-0 attempt avoided parse and symbolic failures."""
    for row in trace_rows:
        if row.get("step_index") != 0:
            continue
        if row.get("attempt_type") != "primary":
            continue
        return not row.get("parse_error") and not row.get("symbolic_error")
    return None


def _build_case(
    steps: list[dict[str, Any]],
    predicate: Any,
    title: str,
    summary: str,
) -> dict[str, Any] | None:
    """Select one trace-backed case study row."""
    for row in sorted(steps, key=_step_sort_key):
        if not predicate(row):
            continue
        return {
            "title": title,
            "summary": summary,
            "model": row["model"],
            "task_id": row["task_id"],
            "difficulty": row["difficulty"],
            "condition": row["condition"],
            "mode": row["mode"],
            "step_index": row["step_index"],
            "attempt_type": row.get("attempt_type"),
            "parse_failure_subtype": row.get("parse_failure_subtype"),
            "symbolic_error": row.get("symbolic_error"),
            "missing_preconditions": row.get("missing_preconditions", []),
            "dependency_boundary_missing_facts": row.get(
                "dependency_boundary_missing_facts", []
            ),
            "expected_gold_action": row.get("expected_gold_action"),
            "parsed_action": row.get("parsed_action"),
            "raw_output_snippet": _snippet(row.get("raw_output")),
            "memory_context_snippet": _snippet(row.get("memory_context")),
            "visible_window_start": row.get("visible_window_start"),
            "finish_reason": row.get("finish_reason"),
            "is_truncated": row.get("is_truncated"),
        }
    return None


def _build_loop_case(trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build the long-horizon looping case from the most repetitive failed trial."""
    candidates = [
        row
        for row in trials
        if row["model"] == "deepseek-v3.2"
        and row.get("goal_satisfied") is False
        and row.get("failure_reason") == "goal_not_satisfied"
    ]
    if not candidates:
        return None
    chosen = sorted(
        candidates,
        key=lambda row: (
            -int(row.get("loop_depth") or 0),
            -int(row.get("executed_steps") or 0),
        ),
    )[0]
    repeated = Counter(chosen.get("predicted_signature_counts", {}))
    top_repeats = [
        {"signature": signature, "count": count}
        for signature, count in repeated.most_common(5)
    ]
    return {
        "title": "DeepSeek 120-step loop",
        "summary": "The model keeps executing legal actions but loses global task tracking.",
        "model": chosen["model"],
        "task_id": chosen["task_id"],
        "difficulty": chosen["difficulty"],
        "condition": chosen["condition"],
        "mode": chosen["mode"],
        "executed_steps": chosen["executed_steps"],
        "gold_step_count": chosen["gold_step_count"],
        "loop_depth": chosen["loop_depth"],
        "top_repeats": top_repeats,
        "failure_reason": chosen["failure_reason"],
        "plan_accuracy": chosen["plan_accuracy"],
    }


def _build_overview_case(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all discovered models for the overview figure."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[trial["model"]].append(trial)

    rows: list[dict[str, Any]] = []
    for model in sorted(grouped):
        model_trials = grouped[model]
        outcomes = Counter(row["outcome_bucket"] for row in model_trials)
        attempted = len(model_trials)
        success = outcomes.get("success", 0)
        rows.append(
            {
                "model": model,
                "attempted": attempted,
                "goal_satisfied": success,
                "goal_satisfaction_rate": _ratio(success, attempted),
                "outcomes": dict(sorted(outcomes.items())),
                "median_loop_depth": _median(
                    [int(row.get("loop_depth") or 0) for row in model_trials]
                ),
            }
        )
    return {"models": rows}


def _snippet(value: Any, limit: int = 320) -> str:
    """Return a compact single-string snippet for figure annotations."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV with stable field order."""
    if not rows:
        path.write_text("")
        return
    fieldnames = _fieldnames(rows)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _fieldnames(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Collect field names in first-seen order."""
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def _csv_value(value: Any) -> Any:
    """Convert nested values to JSON strings for CSV export."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    return value


def _trial_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return the canonical trial key."""
    return (
        str(record.get("model")),
        str(record.get("task_id")),
        str(record.get("condition")),
        str(record.get("mode")),
    )


def _trial_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Stable sort key for trial rows."""
    return (
        _coverage_bucket_rank(record.get("coverage_bucket")),
        str(record.get("model")),
        _difficulty_rank(record.get("difficulty")),
        _condition_rank(record.get("condition")),
        _mode_rank(record.get("mode")),
    )


def _coverage_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Stable sort key for attempt coverage rows."""
    return (
        _coverage_bucket_rank(record.get("coverage_bucket")),
        str(record.get("model")),
        _difficulty_rank(record.get("difficulty")),
        _condition_rank(record.get("condition")),
        _mode_rank(record.get("mode")),
    )


def _step_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Stable sort key for step diagnostic rows."""
    return (
        str(record.get("model")),
        str(record.get("task_id")),
        _condition_rank(record.get("condition")),
        _mode_rank(record.get("mode")),
        int(record.get("step_index", 0) or 0),
        ATTEMPT_TYPE_ORDER.get(str(record.get("attempt_type")), 99),
    )


def _difficulty_rank(value: Any) -> int:
    """Return canonical difficulty rank."""
    if value in DIFFICULTY_ORDER:
        return DIFFICULTY_ORDER.index(str(value))
    return len(DIFFICULTY_ORDER)


def _condition_rank(value: Any) -> int:
    """Return canonical condition rank."""
    if value in CONDITION_ORDER:
        return CONDITION_ORDER.index(str(value))
    return len(CONDITION_ORDER)


def _mode_rank(value: Any) -> int:
    """Return canonical mode rank."""
    if value in MODE_ORDER:
        return MODE_ORDER.index(str(value))
    return len(MODE_ORDER)


def _coverage_bucket(run: ModelRun) -> str:
    """Return the publication cohort bucket for a run."""
    if run.schema_class == "model_scoped" and run.traces_path.exists():
        return "canonical"
    return "legacy"


def _coverage_bucket_rank(value: Any) -> int:
    """Sort canonical rows before legacy rows."""
    return 0 if value == "canonical" else 1


def _action_signature(step: dict[str, Any] | Any) -> str:
    """Convert an action dict into a compact signature string."""
    if not isinstance(step, dict):
        return str(step)
    name = step.get("action_name") or step.get("action")
    arguments = step.get("arguments") or []
    if not name:
        return ""
    return " ".join([str(name), *[str(argument) for argument in arguments]]).strip()


def _ensure_step_list(value: Any) -> list[dict[str, Any]]:
    """Parse a step list from list, JSON string, or repr string."""
    if isinstance(value, list):
        return [step for step in value if isinstance(step, dict)]
    if isinstance(value, str) and value.strip():
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(value)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [step for step in parsed if isinstance(step, dict)]
    return []


def _as_bool(value: Any) -> bool | None:
    """Best-effort boolean coercion."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _float_or_none(value: Any) -> float | None:
    """Best-effort float coercion."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    """Best-effort integer coercion with zero fallback."""
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return a safe ratio or ``None`` when the denominator is zero."""
    if denominator <= 0:
        return None
    return numerator / denominator


def _median(values: list[int]) -> float | None:
    """Return the median of a list or ``None`` for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2
