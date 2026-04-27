"""Tests for src.runner.config."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.runner.config import (
    ExperimentConfig,
    MemoryPolicyConfig,
    ProposalConfig,
    load_config,
)


class TestLoadConfig:
    def test_loads_full_config(self, tmp_path: Path) -> None:
        config_data = {
            "experiment": {"name": "test", "seed": 123},
            "dataset": {"num_instances": 10},
            "model": {"name_or_path": "test-model"},
            "memory_policies": [
                {"name": "full", "type": "full"},
                {"name": "trunc", "type": "truncation", "max_context_tokens": 2048},
            ],
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config_data))

        config = load_config(config_path)
        assert config.experiment.name == "test"
        assert config.experiment.seed == 123
        assert config.dataset.num_instances == 10
        assert config.model.name_or_path == "test-model"
        assert len(config.memory_policies) == 2
        assert config.memory_policies[1].max_context_tokens == 2048

    def test_dataset_instances_per_level(self, tmp_path: Path) -> None:
        config_data = {"dataset": {"instances_per_level": 1}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config_data))
        config = load_config(config_path)
        assert config.dataset.instances_per_level == 1

    def test_model_default_max_new_tokens_is_large(self) -> None:
        config = ProposalConfig()
        assert config.model.max_new_tokens == 8192

    def test_defaults_when_file_missing(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.experiment.seed == 42
        assert config.experiment.prompt_modes == ["direct_action", "cot"]

    def test_defaults_prompt_modes_include_cot(self) -> None:
        config = ExperimentConfig()
        assert "cot" in config.prompt_modes
        assert "direct_action" in config.prompt_modes

    def test_summarization_config_fields(self) -> None:
        policy = MemoryPolicyConfig(
            name="sum",
            type="summarization",
            max_context_tokens=4096,
            recent_window_tokens=1024,
            summary_model="Qwen/Qwen3-1.7B",
        )
        assert policy.summary_model == "Qwen/Qwen3-1.7B"
        assert policy.recent_window_tokens == 1024


class TestValidation:
    def test_truncation_requires_max_tokens(self, tmp_path: Path) -> None:
        config_data = {
            "memory_policies": [
                {"name": "bad_trunc", "type": "truncation"},
            ],
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config_data))
        with pytest.raises(ValueError, match="max_context_tokens"):
            load_config(config_path)

    def test_summarization_requires_max_tokens(self, tmp_path: Path) -> None:
        config_data = {
            "memory_policies": [
                {"name": "bad_sum", "type": "summarization"},
            ],
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config_data))
        with pytest.raises(ValueError, match="max_context_tokens"):
            load_config(config_path)

    def test_full_policy_needs_no_extra_fields(self, tmp_path: Path) -> None:
        config_data = {
            "memory_policies": [{"name": "full", "type": "full"}],
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config_data))
        config = load_config(config_path)
        assert len(config.memory_policies) == 1
