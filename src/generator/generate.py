"""Synthetic planning-task generator for the memory compression study."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from src.common.contracts import PlanningTaskInstance


@dataclass(frozen=True)
class DifficultySpec:
    rooms: int
    packages: int
    distractors: int
    deliveries: int


class LinearLogisticsTaskGenerator:
    """Generate linear household logistics tasks with typed objects and PDDL."""

    def __init__(self, include_distractors: bool = True) -> None:
        self.include_distractors = include_distractors
        self._difficulty_specs: Dict[str, DifficultySpec] = {
            "easy": DifficultySpec(rooms=3, packages=1, distractors=1, deliveries=1),
            "medium": DifficultySpec(rooms=4, packages=2, distractors=1, deliveries=2),
            "hard": DifficultySpec(rooms=5, packages=3, distractors=2, deliveries=3),
        }

    def generate_instance(
        self,
        seed: int,
        difficulty: str,
        instance_index: int,
    ) -> PlanningTaskInstance:
        """Generate a single PDDL logistics task instance.

        Args:
            seed: Base random seed for reproducibility.
            difficulty: One of "easy", "medium", "hard".
            instance_index: Index within the dataset batch.

        Returns:
            A fully populated PlanningTaskInstance with PDDL strings.

        Raises:
            ValueError: If difficulty is not recognized.
        """
        if difficulty not in self._difficulty_specs:
            raise ValueError(f"Unsupported difficulty: {difficulty}")

        rng = random.Random(f"{seed}:{difficulty}:{instance_index}")
        spec = self._difficulty_specs[difficulty]

        rooms = [f"room_{index}" for index in range(1, spec.rooms + 1)]
        robot = "robot_1"
        active_packages = [f"pkg_{index}" for index in range(1, spec.packages + 1)]
        distractor_packages = (
            [f"distractor_pkg_{index}" for index in range(1, spec.distractors + 1)]
            if self.include_distractors
            else []
        )
        all_packages = active_packages + distractor_packages
        typed_objects = {
            "robot": [robot],
            "package": all_packages,
            "room": rooms,
        }

        robot_start = rooms[0]
        initial_facts: List[str] = [
            self._fact("at", robot, robot_start),
            self._fact("handempty", robot),
        ]
        package_locations: Dict[str, str] = {}
        deliveries: List[dict] = []

        for index, package in enumerate(active_packages):
            start_room = rooms[index % max(len(rooms) - 1, 1)]
            if start_room == rooms[-1]:
                start_room = rooms[0]
            destination_candidates = [room for room in rooms if room != start_room]
            # Fix Q12: randomize destination instead of always using rooms[-1]
            destination = rng.choice(destination_candidates)
            package_locations[package] = start_room
            initial_facts.append(self._fact("at", package, start_room))
            deliveries.append(
                {
                    "package": package,
                    "source": start_room,
                    "destination": destination,
                }
            )

        for package in distractor_packages:
            room = rng.choice(rooms)
            package_locations[package] = room
            initial_facts.append(self._fact("at", package, room))

        connections = self._line_connections(rooms)
        initial_facts.extend(connections)

        goal_facts = [
            self._fact("at", delivery["package"], delivery["destination"])
            for delivery in deliveries[: spec.deliveries]
        ]
        goal_text = self._goal_text(deliveries[: spec.deliveries], robot, rooms)

        metadata = {
            "difficulty": difficulty,
            "deliveries": deliveries[: spec.deliveries],
            "connections": connections,
            "action_schemas": {
                "move": ["robot", "room", "room"],
                "pickup": ["robot", "package", "room"],
                "drop": ["robot", "package", "room"],
            },
        }

        return PlanningTaskInstance(
            task_id=f"logistics-{difficulty}-{seed}-{instance_index}",
            domain_name="household_logistics",
            problem_name=f"{difficulty}_instance_{instance_index}",
            goal_text=goal_text,
            valid_objects=[robot, *all_packages, *rooms],
            typed_objects=typed_objects,
            initial_facts=sorted(initial_facts),
            goal_facts=sorted(goal_facts),
            pddl_domain=self._build_domain_pddl(),
            pddl_problem=self._build_problem_pddl(
                problem_name=f"{difficulty}_instance_{instance_index}",
                rooms=rooms,
                robot=robot,
                packages=all_packages,
                initial_facts=initial_facts,
                goal_facts=goal_facts,
            ),
            metadata=metadata,
        )

    def generate_dataset(
        self, count: int, seed: int
    ) -> List[PlanningTaskInstance]:
        """Generate a batch of instances cycling through difficulty levels.

        Args:
            count: Total number of instances to generate.
            seed: Base random seed.

        Returns:
            List of PlanningTaskInstance objects.
        """
        difficulties = list(self._difficulty_specs.keys())
        dataset: List[PlanningTaskInstance] = []
        for index in range(count):
            difficulty = difficulties[index % len(difficulties)]
            dataset.append(
                self.generate_instance(
                    seed=seed,
                    difficulty=difficulty,
                    instance_index=index,
                )
            )
        return dataset

    @staticmethod
    def _fact(name: str, *args: str) -> str:
        return f"{name}({', '.join(args)})"

    @staticmethod
    def _line_connections(rooms: List[str]) -> List[str]:
        facts: List[str] = []
        for left, right in zip(rooms, rooms[1:]):
            facts.append(f"connected({left}, {right})")
            facts.append(f"connected({right}, {left})")
        return facts

    @staticmethod
    def _goal_text(deliveries: List[dict], robot: str, rooms: List[str]) -> str:
        clauses = [
            f"move {delivery['package']} from {delivery['source']} to {delivery['destination']}"
            for delivery in deliveries
        ]
        clause_text = "; ".join(clauses)
        return (
            f"Using {robot}, complete the following deliveries in the linear room layout "
            f"{', '.join(rooms)}: {clause_text}."
        )

    @staticmethod
    def _build_domain_pddl() -> str:
        return """(define (domain household_logistics)
  (:requirements :strips :typing)
  (:types robot package room)
  (:predicates
    (at-robot ?r - robot ?room - room)
    (at-package ?p - package ?room - room)
    (holding ?r - robot ?p - package)
    (handempty ?r - robot)
    (connected ?from - room ?to - room)
  )

  (:action move
    :parameters (?r - robot ?from - room ?to - room)
    :precondition (and
      (at-robot ?r ?from)
      (connected ?from ?to)
    )
    :effect (and
      (not (at-robot ?r ?from))
      (at-robot ?r ?to)
    )
  )

  (:action pickup
    :parameters (?r - robot ?p - package ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (at-package ?p ?room)
      (handempty ?r)
    )
    :effect (and
      (not (at-package ?p ?room))
      (not (handempty ?r))
      (holding ?r ?p)
    )
  )

  (:action drop
    :parameters (?r - robot ?p - package ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (holding ?r ?p)
    )
    :effect (and
      (not (holding ?r ?p))
      (handempty ?r)
      (at-package ?p ?room)
    )
  )
)"""

    @staticmethod
    def _to_pddl_fact(fact: str) -> str:
        name, remainder = fact.split("(", 1)
        args = remainder.rstrip(")")
        tokens = [token.strip() for token in args.split(",") if token.strip()]
        if name == "at":
            if tokens[0].startswith("robot"):
                return f"(at-robot {tokens[0]} {tokens[1]})"
            return f"(at-package {tokens[0]} {tokens[1]})"
        if name == "holding":
            return f"(holding {tokens[0]} {tokens[1]})"
        if name == "handempty":
            return f"(handempty {tokens[0]})"
        if name == "connected":
            return f"(connected {tokens[0]} {tokens[1]})"
        raise ValueError(f"Unsupported fact for PDDL export: {fact}")

    def _build_problem_pddl(
        self,
        problem_name: str,
        rooms: List[str],
        robot: str,
        packages: List[str],
        initial_facts: List[str],
        goal_facts: List[str],
    ) -> str:
        room_objects = " ".join(rooms)
        package_objects = " ".join(packages) if packages else ""
        objects_lines = [
            f"    {robot} - robot",
            f"    {room_objects} - room",
        ]
        if package_objects:
            objects_lines.append(f"    {package_objects} - package")
        init_lines = "\n".join(
            f"    {self._to_pddl_fact(fact)}" for fact in sorted(initial_facts)
        )
        goal_lines = "\n".join(
            f"      {self._to_pddl_fact(fact)}" for fact in sorted(goal_facts)
        )
        return f"""(define (problem {problem_name})
  (:domain household_logistics)
  (:objects
{chr(10).join(objects_lines)}
  )
  (:init
{init_lines}
  )
  (:goal
    (and
{goal_lines}
    )
  )
)"""


def serialize_task(task: PlanningTaskInstance) -> dict:
    """Serialize a PlanningTaskInstance to a JSON-compatible dict."""
    return asdict(task)


def save_dataset(
    dataset: List[PlanningTaskInstance], output_dir: Path
) -> List[Path]:
    """Write each task instance as a JSON file.

    Args:
        dataset: List of generated task instances.
        output_dir: Directory to write instance files to.

    Returns:
        List of paths to written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for task in dataset:
        path = output_dir / f"{task.task_id}.json"
        path.write_text(json.dumps(serialize_task(task), indent=2))
        paths.append(path)
    return paths


def cli_main() -> None:
    """CLI entry point for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate PDDL logistics instances")
    parser.add_argument("--n-instances", type=int, default=40)
    parser.add_argument("--domain", type=str, default="logistics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/instances")
    parser.add_argument("--include-distractors", action="store_true", default=True)
    args = parser.parse_args()

    generator = LinearLogisticsTaskGenerator(
        include_distractors=args.include_distractors
    )
    dataset = generator.generate_dataset(count=args.n_instances, seed=args.seed)
    paths = save_dataset(dataset, Path(args.output_dir))
    print(f"Generated {len(paths)} instances in {args.output_dir}/")


if __name__ == "__main__":
    cli_main()
