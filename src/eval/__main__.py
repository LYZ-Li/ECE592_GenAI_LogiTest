"""Entry point for `python -m src.eval`."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument(
        "--results", type=str, required=True,
        help="Path to results directory containing results.jsonl",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: results_dir/summary.csv)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results)
    results_file = results_dir / "results.jsonl"
    if not results_file.exists():
        print(f"No results.jsonl found in {results_dir}")
        return

    records: List[Dict[str, Any]] = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found in results.jsonl")
        return

    # Aggregate by (condition, mode)
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in records:
        groups[(r["condition"], r["mode"])].append(r)

    output_path = Path(args.output) if args.output else results_dir / "summary.csv"
    metrics = [
        "plan_accuracy", "precondition_violations",
        "ordering_violations", "grounding_errors",
        "valid_plan", "goal_satisfied",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "mode", "n_trials"] + [
            f"{m}_mean" for m in metrics
        ])
        for (condition, mode), group in sorted(groups.items()):
            n = len(group)
            means = []
            for m in metrics:
                values = [float(r[m]) for r in group]
                means.append(sum(values) / len(values) if values else 0.0)
            writer.writerow([condition, mode, n] + [f"{v:.4f}" for v in means])

    print(f"Summary written to {output_path} ({len(groups)} groups, {len(records)} trials)")


if __name__ == "__main__":
    main()
