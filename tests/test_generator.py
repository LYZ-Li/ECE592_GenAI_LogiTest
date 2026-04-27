"""Tests for src.generator.generate."""

import json

import pytest

from src.generator.generate import (
    ACTION_SCHEMAS,
    LinearLogisticsTaskGenerator,
    save_dataset,
    serialize_task,
)


class TestLinearLogisticsTaskGenerator:
    def setup_method(self) -> None:
        self.generator = LinearLogisticsTaskGenerator(include_distractors=True)

    def test_generate_instance_easy(self) -> None:
        task = self.generator.generate_instance(seed=42, difficulty="easy", instance_index=0)
        assert task.task_id == "logistics-easy-42-0"
        assert task.domain_name == "household_logistics"
        assert len(task.goal_facts) == 1
        assert task.goal_facts[0].startswith("delivery_verified(")
        assert "gripper_clean(robot_1)" in task.initial_facts
        assert task.pddl_domain is not None
        assert task.pddl_problem is not None

    def test_generate_instance_medium(self) -> None:
        task = self.generator.generate_instance(seed=42, difficulty="medium", instance_index=0)
        assert len(task.goal_facts) == 2
        assert task.metadata["dirty_gripper"] is True
        assert len(task.metadata["obstructed_packages"]) == 1

    def test_generate_instance_hard(self) -> None:
        task = self.generator.generate_instance(seed=42, difficulty="hard", instance_index=0)
        assert len(task.goal_facts) == 3
        assert len(task.metadata["unstable_packages"]) == 1

    def test_generate_instance_very_hard(self) -> None:
        task = self.generator.generate_instance(seed=42, difficulty="very_hard", instance_index=0)
        assert len(task.goal_facts) == 5
        assert len(task.typed_objects["room"]) == 8

    def test_generate_instance_extreme(self) -> None:
        task = self.generator.generate_instance(seed=42, difficulty="extreme", instance_index=0)
        assert len(task.goal_facts) == 8
        assert len(task.typed_objects["room"]) == 12

    def test_invalid_difficulty(self) -> None:
        with pytest.raises(ValueError, match="Unsupported difficulty"):
            self.generator.generate_instance(seed=42, difficulty="impossible", instance_index=0)

    def test_reproducibility(self) -> None:
        t1 = self.generator.generate_instance(seed=42, difficulty="medium", instance_index=0)
        t2 = self.generator.generate_instance(seed=42, difficulty="medium", instance_index=0)
        assert t1.initial_facts == t2.initial_facts
        assert t1.goal_facts == t2.goal_facts
        assert t1.pddl_problem == t2.pddl_problem

    def test_different_seeds_produce_different_tasks(self) -> None:
        t1 = self.generator.generate_instance(seed=1, difficulty="medium", instance_index=0)
        t2 = self.generator.generate_instance(seed=2, difficulty="medium", instance_index=0)
        # Destinations are now randomized (Q12 fix), so different seeds should
        # produce different delivery configs
        assert t1.task_id != t2.task_id

    def test_action_schemas_in_metadata(self) -> None:
        task = self.generator.generate_instance(seed=42, difficulty="easy", instance_index=0)
        schemas = task.metadata["action_schemas"]
        assert schemas == ACTION_SCHEMAS
        assert len(schemas) == 11
        assert schemas["move"] == ["robot", "room", "room"]
        assert schemas["inspect_gripper"] == ["robot", "room"]
        assert schemas["verify_delivery"] == ["robot", "package", "room"]

    def test_destination_diversity_q12(self) -> None:
        """Q12 fix: destinations should not always be rooms[-1]."""
        destinations = set()
        for i in range(20):
            task = self.generator.generate_instance(seed=i, difficulty="medium", instance_index=0)
            for delivery in task.metadata["deliveries"]:
                destinations.add(delivery["destination"])
        # With 20 different seeds and randomized destinations, we should see
        # more than 1 unique destination
        assert len(destinations) > 1

    def test_typed_objects_present(self) -> None:
        task = self.generator.generate_instance(seed=42, difficulty="medium", instance_index=0)
        assert "robot" in task.typed_objects
        assert "package" in task.typed_objects
        assert "room" in task.typed_objects

    def test_pddl_domain_contains_actions(self) -> None:
        task = self.generator.generate_instance(seed=42, difficulty="easy", instance_index=0)
        for action in ACTION_SCHEMAS:
            assert f"(:action {action}" in task.pddl_domain

    def test_difficulty_adds_dependency_facts(self) -> None:
        easy = self.generator.generate_instance(seed=42, difficulty="easy", instance_index=0)
        hard = self.generator.generate_instance(seed=42, difficulty="hard", instance_index=0)

        assert not easy.metadata["obstructed_packages"]
        assert not easy.metadata["unstable_packages"]
        assert "gripper_dirty(robot_1)" not in easy.initial_facts

        assert hard.metadata["obstructed_packages"]
        assert hard.metadata["unstable_packages"]
        assert "gripper_dirty(robot_1)" in hard.initial_facts
        assert any(f.startswith("obstructed(") for f in hard.initial_facts)
        assert any(f.startswith("unstable(") for f in hard.initial_facts)


class TestGenerateDataset:
    def test_correct_count(self) -> None:
        gen = LinearLogisticsTaskGenerator()
        dataset = gen.generate_dataset(count=6, seed=42)
        assert len(dataset) == 6

    def test_cycles_difficulties(self) -> None:
        gen = LinearLogisticsTaskGenerator()
        dataset = gen.generate_dataset(count=10, seed=42)
        difficulties = [t.metadata["difficulty"] for t in dataset]
        assert difficulties == [
            "easy", "medium", "hard", "very_hard", "extreme",
            "easy", "medium", "hard", "very_hard", "extreme",
        ]

    def test_cycles_difficulty_subset(self) -> None:
        gen = LinearLogisticsTaskGenerator()
        dataset = gen.generate_dataset(count=6, seed=42, difficulty_levels=["easy", "hard"])
        difficulties = [t.metadata["difficulty"] for t in dataset]
        assert difficulties == ["easy", "hard", "easy", "hard", "easy", "hard"]

    def test_balanced_one_per_level(self) -> None:
        gen = LinearLogisticsTaskGenerator()
        dataset = gen.generate_balanced_dataset(seed=42, instances_per_level=1)
        difficulties = [t.metadata["difficulty"] for t in dataset]
        assert difficulties == ["easy", "medium", "hard", "very_hard", "extreme"]
        assert [t.metadata["instance_index"] for t in dataset] == [0, 0, 0, 0, 0]

    def test_balanced_total_instances_selects_first_two_per_level(self) -> None:
        gen = LinearLogisticsTaskGenerator()
        dataset = gen.generate_balanced_dataset(seed=42, total_instances=10)
        difficulties = [t.metadata["difficulty"] for t in dataset]
        indices = [t.metadata["instance_index"] for t in dataset]
        assert difficulties == [
            "easy", "medium", "hard", "very_hard", "extreme",
            "easy", "medium", "hard", "very_hard", "extreme",
        ]
        assert indices == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    def test_balanced_rejects_non_divisible_total(self) -> None:
        gen = LinearLogisticsTaskGenerator()
        with pytest.raises(ValueError, match="divide evenly"):
            gen.generate_balanced_dataset(seed=42, total_instances=7)


class TestSerialization:
    def test_serialize_roundtrip(self) -> None:
        gen = LinearLogisticsTaskGenerator()
        task = gen.generate_instance(seed=42, difficulty="easy", instance_index=0)
        data = serialize_task(task)
        assert data["task_id"] == task.task_id
        assert data["goal_facts"] == task.goal_facts

    def test_save_dataset(self, tmp_path) -> None:
        gen = LinearLogisticsTaskGenerator()
        dataset = gen.generate_dataset(count=3, seed=42)
        paths = save_dataset(dataset, tmp_path / "instances")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            data = json.loads(p.read_text())
            assert "task_id" in data
