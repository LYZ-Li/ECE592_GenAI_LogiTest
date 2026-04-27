"""Entry point for `python -m src.runner`."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import yaml

from src.common.logging import setup_logging
from src.common.reproducibility import set_global_seed
from src.generator.generate import LinearLogisticsTaskGenerator
from src.runner.config import ProposalConfig, load_config
from src.runner.orchestrator import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the memory compression experiment")
    parser.add_argument(
        "--config", type=str, default="configs/experiment.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--n-instances", type=int, default=None,
        help="Override number of instances from config",
    )
    parser.add_argument(
        "--model-config", type=str, default=None,
        help="Optional model config YAML to overlay onto config.model",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override experiment output directory",
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        help="Run only this memory condition (by name)",
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        help="Run only this prompt mode (direct_action or cot)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model_config:
        _apply_model_config(config, Path(args.model_config))
    if args.output_dir:
        config.experiment.output_dir = args.output_dir
    setup_logging(level=args.log_level, experiment_name=config.experiment.name)
    set_global_seed(config.experiment.seed)

    # Override config if CLI args provided
    if args.condition:
        config.memory_policies = [
            p for p in config.memory_policies if p.name == args.condition
        ]
        if not config.memory_policies:
            raise ValueError(f"No memory policy named '{args.condition}' in config.")
    if args.mode:
        config.experiment.prompt_modes = [args.mode]

    # Generate dataset
    generator = LinearLogisticsTaskGenerator(
        include_distractors=config.dataset.include_distractors
    )
    if args.n_instances is not None:
        dataset = generator.generate_balanced_dataset(
            total_instances=args.n_instances,
            seed=config.experiment.seed,
            difficulty_levels=config.dataset.difficulty_levels,
        )
    elif config.dataset.instances_per_level is not None:
        dataset = generator.generate_balanced_dataset(
            instances_per_level=config.dataset.instances_per_level,
            seed=config.experiment.seed,
            difficulty_levels=config.dataset.difficulty_levels,
        )
    else:
        dataset = generator.generate_balanced_dataset(
            total_instances=config.dataset.num_instances,
            seed=config.experiment.seed,
            difficulty_levels=config.dataset.difficulty_levels,
        )

    # Run
    output_dir = Path(config.experiment.output_dir)
    run_batch(config, dataset, output_dir)


def _apply_model_config(config: ProposalConfig, path: Path) -> None:
    """Overlay a model YAML onto config.model, ignoring non-model labels."""
    raw = yaml.safe_load(path.read_text()) or {}
    model_fields = {field.name for field in fields(config.model)}
    for key, value in raw.items():
        if key in model_fields:
            setattr(config.model, key, value)


if __name__ == "__main__":
    main()
