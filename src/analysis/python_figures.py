"""Python-native report figure generation without third-party plotting libs."""

from __future__ import annotations

import json
import subprocess
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.analysis.canonical import CONDITION_ORDER, DIFFICULTY_ORDER


OUTCOME_COLORS = {
    "success": (0.09, 0.42, 0.32),
    "parse_error": (0.70, 0.29, 0.25),
    "precondition_violation": (0.71, 0.48, 0.07),
    "grounding_error": (0.36, 0.56, 0.85),
    "goal_not_satisfied": (0.18, 0.37, 0.55),
    "unknown_failure": (0.45, 0.49, 0.54),
    "not_attempted": (0.82, 0.84, 0.87),
}
INK = (0.11, 0.14, 0.19)
MUTED = (0.40, 0.44, 0.50)
PANEL = (1.0, 1.0, 1.0)
LINE = (0.85, 0.88, 0.91)
BACKGROUND = (0.97, 0.98, 0.99)
BLUE = (0.09, 0.31, 0.48)
PALE_BLUE = (0.95, 0.97, 0.99)
DARK_BLOCK = (0.07, 0.10, 0.16)
LIGHT_TEXT = (0.89, 0.92, 0.95)


class PdfCanvas:
    """Very small single-page PDF canvas."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = []

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: tuple[float, float, float] | None = None,
        stroke: tuple[float, float, float] | None = None,
        line_width: float = 1.0,
    ) -> None:
        bottom = self.height - y - height
        cmd = ["q"]
        if fill:
            cmd.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg")
        if stroke:
            cmd.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG")
            cmd.append(f"{line_width:.2f} w")
        cmd.append(f"{x:.2f} {bottom:.2f} {width:.2f} {height:.2f} re")
        if fill and stroke:
            cmd.append("B")
        elif fill:
            cmd.append("f")
        else:
            cmd.append("S")
        cmd.append("Q")
        self.commands.append("\n".join(cmd))

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: tuple[float, float, float] = LINE,
        width: float = 1.0,
    ) -> None:
        top1 = self.height - y1
        top2 = self.height - y2
        self.commands.append(
            "\n".join(
                [
                    "q",
                    f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG",
                    f"{width:.2f} w",
                    f"{x1:.2f} {top1:.2f} m",
                    f"{x2:.2f} {top2:.2f} l",
                    "S",
                    "Q",
                ]
            )
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: int = 12,
        font: str = "F1",
        color: tuple[float, float, float] = INK,
    ) -> None:
        baseline = self.height - y - size
        escaped = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        self.commands.append(
            "\n".join(
                [
                    "q",
                    f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg",
                    "BT",
                    f"/{font} {size} Tf",
                    f"1 0 0 1 {x:.2f} {baseline:.2f} Tm",
                    f"({escaped}) Tj",
                    "ET",
                    "Q",
                ]
            )
        )

    def text_block(
        self,
        x: float,
        y: float,
        width: float,
        text: str,
        *,
        size: int = 12,
        font: str = "F1",
        color: tuple[float, float, float] = INK,
        line_gap: int = 4,
    ) -> float:
        lines = _wrap_text(text, width, size)
        cursor = y
        for line in lines:
            self.text(x, cursor, line, size=size, font=font, color=color)
            cursor += size + line_gap
        return cursor

    def save(self, path: str | Path) -> None:
        stream = ("\n".join(self.commands) + "\n").encode("latin-1", errors="replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                "/Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R /F3 7 0 R >> >> >>"
            ).encode("latin-1"),
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
        ]
        content = b"%PDF-1.4\n"
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(content))
            content += f"{index} 0 obj\n".encode("ascii")
            content += obj + b"\nendobj\n"
        xref_offset = len(content)
        content += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
        content += b"0000000000 65535 f \n"
        for offset in offsets[1:]:
            content += f"{offset:010d} 00000 n \n".encode("ascii")
        content += (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
        Path(path).write_bytes(content)


def generate_python_report_figures(bundle: Any, figures_dir: str | Path) -> list[str]:
    """Generate direct PDF and PNG report figures."""
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "overview_all_models": _build_overview_figure,
        "parse_failure_taxonomy": _build_parse_figure,
        "capability_baselines": _build_baseline_figure,
        "deepseek_boundary_violation": _build_boundary_figure,
        "deepseek_loop_collapse": _build_loop_figure,
        "extreme_horizon_ceiling": _build_extreme_figure,
    }
    generated: list[str] = []
    for stem, builder in builders.items():
        canvas = PdfCanvas(width=1240, height=980)
        canvas.rect(0, 0, 1240, 980, fill=BACKGROUND)
        builder(canvas, bundle)
        pdf_path = output_dir / f"{stem}.pdf"
        png_path = output_dir / f"{stem}.png"
        canvas.save(pdf_path)
        _pdf_to_png(pdf_path, png_path)
        generated.append(stem)
    (output_dir / "manifest.json").write_text(
        json.dumps({"figure_stems": generated, "mode": "python-native"}, indent=2)
    )
    return generated


def _build_overview_figure(canvas: PdfCanvas, bundle: Any) -> None:
    summary = bundle.summary
    _header(
        canvas,
        title="All-model overview, coverage, and terminal failure taxonomy",
        lede=(
            "The canonical cohort contains nine comparable model-scoped runs. A separate legacy Claude "
            "run is shown for context, but denominator-based coverage claims are computed on the canonical cohort only."
        ),
        metrics=[
            ("Canonical attempted", str(summary["attempted_trials_canonical_models"])),
            ("Canonical expected", str(summary["expected_trials_canonical_models"])),
            ("All attempted", str(summary["attempted_trials_all_models"])),
            ("Canonical successes", str(summary["goal_satisfied_canonical_models"])),
        ],
    )
    _section_box(canvas, 40, 190, 730, 720, "Model outcome profiles")
    _section_box(canvas, 790, 190, 410, 720, "Coverage by difficulty tier")

    model_rows = summary["summary_by_model"]
    y = 232
    for row in model_rows:
        canvas.text(60, y, row["model"], size=12, font="F2")
        canvas.text(250, y, f"{row['attempted']}/{row['expected']} attempted", size=10, color=MUTED)
        canvas.text(420, y, _percent(row["goal_satisfaction_rate"]), size=10, color=MUTED)
        _stacked_bar(canvas, 520, y - 4, 220, 14, row["outcomes"], row["attempted"])
        y += 56

    coverage_lookup = defaultdict(int)
    for row in bundle.attempt_coverage:
        if row["attempt_status"] == "attempted":
            coverage_lookup[(row["model"], row["difficulty"])] += 1

    headers = ["Model", "Easy", "Medium", "Hard", "Very Hard", "Extreme"]
    rows = []
    for row in model_rows:
        rows.append(
            [row["model"]]
            + [f"{coverage_lookup[(row['model'], difficulty)]}/10" for difficulty in DIFFICULTY_ORDER]
        )
    _table(canvas, 808, 232, 374, headers, rows, [124, 48, 56, 48, 68, 48], row_height=28)
    _callout(
        canvas,
        808,
        640,
        374,
        115,
        "Missing cells are treated as not attempted, not failures. This matters most for the universally absent "
        "extreme task and for the manually halted small-model sweeps.",
    )


def _build_parse_figure(canvas: PdfCanvas, bundle: Any) -> None:
    parse_counts = Counter(
        row["parse_failure_subtype"]
        for row in bundle.step_diagnostics
        if row.get("parse_failure_subtype")
    )
    _header(
        canvas,
        title="Parse failure taxonomy",
        lede=(
            "The canonical pipeline separates empty truncated outputs, wrong-key schema drift, hallucinated action names, "
            "and non-empty context-overflow truncation so the report can tie claims to the actual mechanism."
        ),
        metrics=[
            ("Empty truncated", str(parse_counts.get("empty_truncated_output", 0))),
            ("Wrong key", str(parse_counts.get("schema_wrong_key", 0))),
            ("Hallucinated action", str(parse_counts.get("hallucinated_action_name", 0))),
            ("Context overflow", str(parse_counts.get("context_overflow_truncation", 0))),
        ],
    )
    cards = [
        bundle.case_studies["cases"].get("qwen_token_exhaustion"),
        bundle.case_studies["cases"].get("qwen_wrong_key"),
        bundle.case_studies["cases"].get("gemma_context_overflow"),
    ]
    x_positions = [40, 430, 820]
    for x, case in zip(x_positions, cards):
        _case_card(canvas, x, 210, 360, 690, case)


def _build_baseline_figure(canvas: PdfCanvas, bundle: Any) -> None:
    _header(
        canvas,
        title="Capability-baseline failures",
        lede=(
            "These models never fully enter the memory-compression regime. Their failures are worth reporting, but they "
            "belong in a capability-baseline frame rather than in the causal memory analysis."
        ),
        metrics=[
            ("LLaMA-3B theme", "step-0 crash"),
            ("LLaMA-8B theme", "hallucinated actions"),
            ("Minimax theme", "rare skipped prerequisites"),
            ("Interpretation", "baseline limits"),
        ],
    )
    cards = [
        bundle.case_studies["cases"].get("llama3b_step0"),
        bundle.case_studies["cases"].get("llama8b_hallucinated_action"),
        bundle.case_studies["cases"].get("minimax_precondition_skip"),
    ]
    x_positions = [40, 430, 820]
    for x, case in zip(x_positions, cards):
        _case_card(canvas, x, 210, 360, 690, case)


def _build_boundary_figure(canvas: PdfCanvas, bundle: Any) -> None:
    case = bundle.case_studies["cases"].get("deepseek_boundary_violation")
    trial = _find_trial(bundle.trials, case)
    boundary_facts = case.get("dependency_boundary_missing_facts", []) if case else []
    strict_boundary = bool(case.get("strict_boundary_violation")) if case else False
    title = (
        "DeepSeek truncation boundary violation"
        if strict_boundary
        else "DeepSeek truncation dependency failure"
    )
    lede = (
        "This is the clearest positive memory-compression result in the dataset: a required precondition aligns "
        "with a gold prerequisite step that sits outside the visible tail window at failure time."
        if strict_boundary
        else "The strongest truncation-era precondition failure does not satisfy the conservative boundary metric, "
        "but it still shows the model inferring that a prerequisite had been met when the current state says otherwise."
    )
    _header(
        canvas,
        title=title,
        lede=lede,
        metrics=[
            ("Missing fact", ", ".join(boundary_facts or case.get("missing_preconditions", [])) if case else "n/a"),
            ("Window start", str(case.get("visible_window_start")) if case else "n/a"),
            ("Failure step", str(case.get("step_index")) if case else "n/a"),
            ("Strict boundary", "yes" if strict_boundary else "no"),
        ],
    )
    _section_box(canvas, 40, 210, 560, 690, "Gold prerequisite context")
    _section_box(canvas, 630, 210, 570, 690, "Predicted trajectory around failure")
    fact = (boundary_facts or case.get("missing_preconditions", []))[0] if case and (boundary_facts or case.get("missing_preconditions")) else None
    nearby_gold = _gold_context_rows(trial, fact)
    gold_rows = [[str(row["index"]), row["action"], row["effects"]] for row in nearby_gold]
    _table(canvas, 58, 250, 524, ["Step", "Gold action", "Effects"], gold_rows, [42, 190, 292], row_height=28)
    _callout(
        canvas,
        58,
        540,
        524,
        120,
        (
            "The boundary metric marks only missing facts whose gold producer step lies before the active recent tail. "
            "This case meets that threshold."
            if strict_boundary
            else "The boundary metric remains false here because the gold producer step is still inside the active tail. "
            "The failure is still valuable: the model claims a prerequisite is satisfied even though the symbolic state says it is not."
        ),
    )
    predicted_rows = _predicted_slice(trial, case.get("step_index") if case else None)
    _table(
        canvas,
        648,
        250,
        534,
        ["Step", "Predicted action"],
        [[str(row["index"]), row["action"]] for row in predicted_rows],
        [52, 462],
        row_height=28,
    )
    _code_block(canvas, 648, 390, 534, 126, "Model output snippet", case.get("raw_output_snippet", "") if case else "")
    _code_block(canvas, 648, 545, 534, 200, "Memory context snippet", case.get("memory_context_snippet", "") if case else "")


def _build_loop_figure(canvas: PdfCanvas, bundle: Any) -> None:
    case = bundle.case_studies["cases"].get("deepseek_loop")
    trial = _find_trial(bundle.trials, case)
    _header(
        canvas,
        title="DeepSeek long-horizon tracking collapse",
        lede=(
            "These runs are not parser crashes. DeepSeek remains symbolically valid for the entire run, then exceeds the "
            "step cap because it loses global accounting of which deliveries are still outstanding."
        ),
        metrics=[
            ("Executed steps", str(case.get("executed_steps")) if case else "n/a"),
            ("Gold steps", str(case.get("gold_step_count")) if case else "n/a"),
            ("Loop depth", str(case.get("loop_depth")) if case else "n/a"),
            ("Plan accuracy", _percent(case.get("plan_accuracy")) if case else "n/a"),
        ],
    )
    _section_box(canvas, 40, 210, 500, 690, "Repeated action signatures")
    _section_box(canvas, 570, 210, 630, 690, "Trajectory preview")
    repeats = case.get("top_repeats", []) if case else []
    _table(
        canvas,
        58,
        250,
        464,
        ["Signature", "Count"],
        [[row["signature"], str(row["count"])] for row in repeats],
        [358, 86],
        row_height=30,
    )
    _callout(
        canvas,
        58,
        460,
        464,
        124,
        "In the canonical report, tracking collapse is reserved for overlong executions with substantial repetition, "
        "not for every failed trial.",
    )
    early = "\n".join(_sequence_lines(trial, 0, 10))
    late = "\n".join(
        _sequence_lines(trial, max(0, int(case.get("executed_steps", 0)) - 10), int(case.get("executed_steps", 0)))
    )
    _code_block(canvas, 588, 250, 594, 220, "Early execution", early)
    _code_block(canvas, 588, 500, 594, 220, "Late execution", late)


def _build_extreme_figure(canvas: PdfCanvas, bundle: Any) -> None:
    extreme_attempts = _extreme_attempts(bundle.attempt_coverage)
    extreme_successes = _extreme_successes(bundle.attempt_coverage)
    _header(
        canvas,
        title="Extreme-task horizon ceiling",
        lede=(
            "The jump from very hard to extreme nearly doubles the oracle plan length. Extreme-tier coverage is sparse rather than "
            "absent: only a subset of models logged those cells, and only four canonical runs reached the goal."
        ),
        metrics=[
            ("Very hard gold steps", str(_plan_steps(bundle.plans, "very_hard"))),
            ("Extreme gold steps", str(_plan_steps(bundle.plans, "extreme"))),
            ("Extreme attempts logged", str(extreme_attempts)),
            ("Extreme successes", str(extreme_successes)),
        ],
    )
    _section_box(canvas, 40, 210, 560, 690, "Oracle plan length by difficulty")
    _section_box(canvas, 630, 210, 570, 690, "Extreme-tier attempt coverage")
    max_steps = max(plan["gold_step_count"] for plan in bundle.plans)
    y = 270
    for difficulty in DIFFICULTY_ORDER:
        steps = _plan_steps(bundle.plans, difficulty)
        if not steps:
            continue
        width = 360 * (steps / max_steps)
        canvas.text(60, y, difficulty.replace("_", " ").title(), size=12, font="F2")
        canvas.rect(220, y - 2, 360, 16, fill=(0.93, 0.95, 0.97), stroke=LINE)
        canvas.rect(220, y - 2, width, 16, fill=(0.15, 0.44, 0.65))
        canvas.text(590, y, f"{steps} steps", size=11, color=MUTED)
        y += 70
    extreme_rows = []
    lookup = defaultdict(int)
    for row in bundle.attempt_coverage:
        if row["difficulty"] == "extreme" and row["attempt_status"] == "attempted":
            lookup[row["model"]] += 1
    success_lookup = defaultdict(int)
    for row in bundle.attempt_coverage:
        if (
            row["difficulty"] == "extreme"
            and row["attempt_status"] == "attempted"
            and row.get("goal_satisfied") is True
        ):
            success_lookup[row["model"]] += 1
    for model in sorted({row["model"] for row in bundle.attempt_coverage}):
        extreme_rows.append([model, f"{lookup[model]}/10", str(success_lookup[model])])
    _table(
        canvas,
        648,
        250,
        534,
        ["Model", "Logged attempts", "Successes"],
        extreme_rows,
        [328, 118, 88],
        row_height=28,
    )
    _callout(
        canvas,
        648,
        620,
        534,
        120,
        "The canonical build separates sparse logging from symbolic failure. Extreme-tier claims should therefore be framed as "
        "directional case evidence, not as a complete cross-model comparison.",
    )


def _header(
    canvas: PdfCanvas,
    *,
    title: str,
    lede: str,
    metrics: list[tuple[str, str]],
) -> None:
    canvas.text(40, 34, "Canonical Figure Group", size=11, font="F2", color=BLUE)
    canvas.text_block(40, 56, 760, title, size=26, font="F2")
    canvas.text_block(40, 102, 760, lede, size=13, color=MUTED)
    x = 840
    y = 34
    for index, (caption, value) in enumerate(metrics):
        tile_x = x + (index % 2) * 170
        tile_y = y + (index // 2) * 74
        _metric_tile(canvas, tile_x, tile_y, 150, 58, caption, value)


def _metric_tile(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    caption: str,
    value: str,
) -> None:
    canvas.rect(x, y, width, height, fill=PANEL, stroke=LINE)
    canvas.text_block(x + 12, y + 12, width - 24, value, size=18, font="F2", color=BLUE)
    canvas.text_block(x + 12, y + 36, width - 24, caption, size=9, color=MUTED)


def _section_box(canvas: PdfCanvas, x: float, y: float, width: float, height: float, title: str) -> None:
    canvas.rect(x, y, width, height, fill=PANEL, stroke=LINE)
    canvas.text(x + 18, y + 18, title, size=16, font="F2")


def _stacked_bar(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    outcomes: dict[str, int],
    attempted: int,
) -> None:
    canvas.rect(x, y, width, height, fill=(0.94, 0.95, 0.97), stroke=LINE)
    if attempted <= 0:
        return
    cursor = x
    for key, value in sorted(outcomes.items()):
        if value <= 0:
            continue
        segment = width * (value / attempted)
        canvas.rect(cursor, y, segment, height, fill=OUTCOME_COLORS.get(key, OUTCOME_COLORS["unknown_failure"]))
        cursor += segment


def _table(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float],
    *,
    row_height: float = 26,
) -> None:
    canvas.rect(x, y, width, row_height, fill=PALE_BLUE, stroke=LINE)
    cursor_x = x
    for header, col_width in zip(headers, col_widths):
        canvas.line(cursor_x, y, cursor_x, y + row_height + row_height * len(rows), color=LINE)
        canvas.text_block(cursor_x + 6, y + 8, col_width - 12, header, size=10, font="F2")
        cursor_x += col_width
    canvas.line(x + width, y, x + width, y + row_height + row_height * len(rows), color=LINE)
    canvas.line(x, y + row_height, x + width, y + row_height, color=LINE)
    current_y = y + row_height
    for row in rows:
        canvas.rect(x, current_y, width, row_height, fill=PANEL, stroke=LINE)
        cursor_x = x
        for value, col_width in zip(row, col_widths):
            canvas.text_block(cursor_x + 6, current_y + 8, col_width - 12, value, size=9, color=INK)
            cursor_x += col_width
        current_y += row_height


def _callout(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
) -> None:
    canvas.rect(x, y, width, height, fill=PALE_BLUE, stroke=LINE)
    canvas.rect(x, y, 4, height, fill=BLUE)
    canvas.text_block(x + 14, y + 16, width - 28, text, size=11, color=(0.17, 0.26, 0.35))


def _code_block(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    text: str,
) -> None:
    canvas.text(x, y - 4, title.upper(), size=9, font="F2", color=MUTED)
    canvas.rect(x, y + 14, width, height, fill=DARK_BLOCK, stroke=(0.16, 0.20, 0.28))
    canvas.text_block(x + 14, y + 30, width - 28, text or "(none)", size=10, font="F3", color=LIGHT_TEXT, line_gap=3)


def _case_card(
    canvas: PdfCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    case: dict[str, Any] | None,
) -> None:
    canvas.rect(x, y, width, height, fill=PANEL, stroke=LINE)
    if not case:
        canvas.text(x + 18, y + 18, "Case unavailable", size=16, font="F2")
        return
    canvas.text_block(x + 18, y + 18, width - 36, case["title"], size=16, font="F2")
    cursor = canvas.text_block(x + 18, y + 46, width - 36, case["summary"], size=11, color=MUTED)
    rows = []
    for label, value in [
        ("Model", case.get("model")),
        ("Task", case.get("task_id")),
        ("Condition", case.get("condition")),
        ("Mode", case.get("mode")),
        ("Step", str(case.get("step_index")) if case.get("step_index") is not None else None),
        ("Subtype", case.get("parse_failure_subtype") or case.get("symbolic_error")),
    ]:
        if value not in (None, ""):
            rows.append([label, str(value)])
    _table(canvas, x + 18, cursor + 10, width - 36, ["Field", "Value"], rows, [92, width - 36 - 92], row_height=24)
    _code_block(canvas, x + 18, cursor + 150, width - 36, 170, "Output snippet", case.get("raw_output_snippet", ""))
    _code_block(canvas, x + 18, cursor + 350, width - 36, 170, "Memory snippet", case.get("memory_context_snippet", ""))


def _find_trial(trials: list[dict[str, Any]], case: dict[str, Any] | None) -> dict[str, Any] | None:
    if not case:
        return None
    for trial in trials:
        if (
            trial["model"] == case.get("model")
            and trial["task_id"] == case.get("task_id")
            and trial["condition"] == case.get("condition")
            and trial["mode"] == case.get("mode")
        ):
            return trial
    return None


def _gold_context_rows(trial: dict[str, Any] | None, fact: str | None) -> list[dict[str, Any]]:
    if not trial or not fact:
        return []
    target_index = None
    for index, step in enumerate(trial.get("gold_steps", [])):
        if fact in (step.get("effects") or []):
            target_index = index
            break
    if target_index is None:
        return []
    start = max(0, target_index - 2)
    stop = min(len(trial.get("gold_steps", [])), target_index + 3)
    rows = []
    for index in range(start, stop):
        step = trial["gold_steps"][index]
        rows.append(
            {
                "index": index,
                "action": _signature(step),
                "effects": ", ".join(step.get("effects", []) or []),
            }
        )
    return rows


def _predicted_slice(trial: dict[str, Any] | None, center: int | None) -> list[dict[str, Any]]:
    if not trial or center is None:
        return []
    predicted = trial.get("predicted_steps", [])
    start = max(0, int(center) - 3)
    stop = min(len(predicted), int(center) + 2)
    return [{"index": index, "action": _signature(predicted[index])} for index in range(start, stop)]


def _sequence_lines(trial: dict[str, Any] | None, start: int, stop: int) -> list[str]:
    if not trial:
        return []
    predicted = trial.get("predicted_steps", [])
    return [f"{index:02d}: {_signature(predicted[index])}" for index in range(max(0, start), min(stop, len(predicted)))]


def _plan_steps(plans: list[dict[str, Any]], difficulty: str) -> int:
    for plan in plans:
        if plan["difficulty"] == difficulty:
            return int(plan["gold_step_count"])
    return 0


def _extreme_attempts(coverage: list[dict[str, Any]]) -> int:
    return sum(
        1 for row in coverage if row["difficulty"] == "extreme" and row["attempt_status"] == "attempted"
    )


def _extreme_successes(coverage: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in coverage
        if row["difficulty"] == "extreme"
        and row["attempt_status"] == "attempted"
        and row.get("goal_satisfied") is True
    )


def _signature(step: dict[str, Any]) -> str:
    name = step.get("action_name") or step.get("action") or "?"
    arguments = step.get("arguments") or []
    return f"{name}({', '.join(str(argument) for argument in arguments)})"


def _wrap_text(text: str, width: float, size: int) -> list[str]:
    if not text:
        return [""]
    max_chars = max(12, int(width / max(size * 0.54, 1)))
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=max_chars, break_long_words=True))
    return lines


def _percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return str(value)


def _pdf_to_png(pdf_path: Path, png_path: Path) -> None:
    subprocess.run(
        ["sips", "-s", "format", "png", str(pdf_path), "--out", str(png_path)],
        check=True,
        capture_output=True,
        text=True,
    )
