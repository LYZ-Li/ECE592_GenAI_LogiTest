"""Entry point for `python -m src.analysis`."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis plots")
    parser.add_argument(
        "--results", type=str, required=True,
        help="Path to results directory containing results.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures (default: results_dir/figures/)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"

    from src.analysis.plot import generate_all_figures
    generate_all_figures(results_dir, output_dir)


main()
