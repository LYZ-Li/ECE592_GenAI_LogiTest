"""Entry point for ``python -m src.analysis``."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.plot import (
    generate_legacy_single_run_summary,
    generate_publication_bundle,
)


def main() -> None:
    """Parse CLI arguments and run the requested analysis mode."""
    parser = argparse.ArgumentParser(description="Generate canonical analysis artifacts")
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Root directory containing per-model result folders",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="analysis/canonical",
        help="Output directory for canonical CSV and JSON tables",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="paper/figures",
        help="Output directory for report figure PDF/PNG assets",
    )
    parser.add_argument(
        "--plans-dir",
        type=str,
        default="data/plans",
        help="Directory containing canonical plan JSON files",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help=(
            "Deprecated compatibility path: directory containing one results.jsonl "
            "file for a single-run summary"
        ),
    )
    args = parser.parse_args()

    if args.results_root:
        generate_publication_bundle(
            results_root=Path(args.results_root),
            analysis_dir=Path(args.analysis_dir),
            figures_dir=Path(args.figures_dir),
            plans_dir=Path(args.plans_dir),
        )
        return

    if args.results:
        generate_legacy_single_run_summary(
            results_dir=Path(args.results),
            output_dir=Path(args.analysis_dir),
        )
        return

    parser.error("Provide either --results-root for canonical analysis or --results for compatibility mode.")


if __name__ == "__main__":
    main()
