"""Canonical analysis and publication-figure orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.analysis.canonical import CanonicalBundle, build_canonical_bundle, export_bundle
from src.analysis.python_figures import generate_python_report_figures


def generate_publication_bundle(
    results_root: str | Path,
    analysis_dir: str | Path,
    figures_dir: str | Path,
    plans_dir: str | Path = "data/plans",
) -> CanonicalBundle:
    """Build the canonical tables plus report-ready PDF and PNG figures."""
    bundle = build_canonical_bundle(results_root=results_root, plans_dir=plans_dir)
    export_bundle(bundle, analysis_dir)
    generate_python_report_figures(bundle, figures_dir)
    _write_report_metrics(bundle, figures_dir)
    return bundle


def generate_legacy_single_run_summary(
    results_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Deprecated compatibility path for one ``results.jsonl`` file."""
    results_path = Path(results_dir) / "results.jsonl"
    records = []
    if results_path.exists():
        with open(results_path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    goal_satisfied = sum(1 for record in records if record.get("goal_satisfied"))
    parse_failures = sum(
        1 for record in records if int(record.get("parse_error_count", 0) or 0) > 0
    )
    summary = {
        "mode": "deprecated-single-run",
        "records": len(records),
        "goal_satisfied": goal_satisfied,
        "parse_failures": parse_failures,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "legacy_single_run_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def _write_report_metrics(bundle: CanonicalBundle, figures_dir: str | Path) -> None:
    """Write a small JSON helper with headline metrics for the LaTeX report."""
    report_path = Path(figures_dir) / "report_metrics.json"
    report_path.write_text(json.dumps(bundle.summary, indent=2))
