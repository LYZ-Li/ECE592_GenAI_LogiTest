"""Entry point for `python -m src.runner`."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.logging import setup_logging
from src.common.reproducibility import set_global_seed
from src.generator.generate import LinearLogisticsTaskGenerator
from src.runner.config import load_config
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
    setup_logging(level=args.log_level, experiment_name=config.experiment.name)
    set_global_seed(config.experiment.seed)

    # Override config if CLI args provided
    n_instances = args.n_instances or config.dataset.num_instances
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
    dataset = generator.generate_dataset(count=n_instances, seed=config.experiment.seed)

    # Run
    output_dir = Path(config.experiment.output_dir)
    run_batch(config, dataset, output_dir)


main()
