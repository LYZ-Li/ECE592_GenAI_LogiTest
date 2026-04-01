"""Plotting and statistical analysis for the memory compression study."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/Colab use
import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "plan_accuracy",
    "precondition_violations",
    "ordering_violations",
    "grounding_errors",
]

CONDITION_ORDER = ["full_context", "truncation_4096", "summarization_4096"]
MODE_ORDER = ["direct_action", "cot"]


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load results.jsonl into a DataFrame.

    Args:
        results_dir: Directory containing results.jsonl.

    Returns:
        DataFrame with one row per trial.
    """
    results_file = results_dir / "results.jsonl"
    records: List[Dict[str, Any]] = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def plot_condition_comparison(
    df: pd.DataFrame, output_dir: Path
) -> None:
    """Create a 3x2 grid of bar charts: rows=metrics, cols=prompt mode.

    Args:
        df: Results DataFrame.
        output_dir: Directory to save the figure.
    """
    fig, axes = plt.subplots(
        len(METRICS), len(MODE_ORDER),
        figsize=(12, 3 * len(METRICS)),
        squeeze=False,
    )

    for col_idx, mode in enumerate(MODE_ORDER):
        mode_df = df[df["mode"] == mode]
        for row_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            grouped = mode_df.groupby("condition")[metric].mean()
            # Reindex to ensure consistent order
            conditions = [c for c in CONDITION_ORDER if c in grouped.index]
            values = [grouped.get(c, 0) for c in conditions]
            bars = ax.bar(range(len(conditions)), values, color=["#4C72B0", "#DD8452", "#55A868"])
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
            if row_idx == 0:
                ax.set_title(mode.replace("_", " ").title(), fontsize=11, fontweight="bold")
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

    fig.suptitle("Memory Condition x Prompt Mode Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "condition_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a heatmap of mean metrics by condition x mode.

    Args:
        df: Results DataFrame.
        output_dir: Directory to save the figure.
    """
    pivot_data = {}
    for metric in METRICS:
        pivot = df.pivot_table(
            values=metric, index="condition", columns="mode", aggfunc="mean"
        )
        pivot_data[metric] = pivot

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4), squeeze=False)
    for idx, metric in enumerate(METRICS):
        ax = axes[0, idx]
        pivot = pivot_data[metric]
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(metric.replace("_", " ").title(), fontsize=9)
        # Add text annotations
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Metric Heatmap: Condition x Mode", fontsize=12)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "metric_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_failure_breakdown(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of failure reasons by condition.

    Args:
        df: Results DataFrame.
        output_dir: Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    failure_df = df[df["failure_reason"].notna()]
    if failure_df.empty:
        ax.text(0.5, 0.5, "No failures recorded", ha="center", va="center", transform=ax.transAxes)
    else:
        cross = pd.crosstab(failure_df["condition"], failure_df["failure_reason"])
        cross.plot(kind="bar", ax=ax, rot=45)
        ax.set_ylabel("Count")
        ax.set_title("Failure Reasons by Memory Condition")
        ax.legend(title="Failure Reason", fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "failure_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(results_dir: Path, output_dir: Path) -> None:
    """Generate all analysis figures.

    Args:
        results_dir: Directory containing results.jsonl.
        output_dir: Directory to save figures.
    """
    df = load_results(results_dir)
    print(f"Loaded {len(df)} trial records")

    plot_condition_comparison(df, output_dir)
    print(f"  Saved condition_comparison.png")

    plot_metric_heatmap(df, output_dir)
    print(f"  Saved metric_heatmap.png")

    plot_failure_breakdown(df, output_dir)
    print(f"  Saved failure_breakdown.png")

    print(f"All figures saved to {output_dir}/")
