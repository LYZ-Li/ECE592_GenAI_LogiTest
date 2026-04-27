"""Synthetic planning-task generator for the memory compression study."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from src.common.contracts import PlanningTaskInstance

ACTION_SCHEMAS = {
    "move": ["robot", "room", "room"],
    "inspect_gripper": ["robot", "room"],
    "clean_gripper": ["robot", "room"],
    "calibrate_gripper": ["robot", "room"],
    "inspect_package": ["robot", "package", "room"],
    "clear_obstruction": ["robot", "package", "room"],
    "pickup": ["robot", "package", "room"],
    "verify_grasp": ["robot", "package", "room"],
    "regrasp": ["robot", "package", "room"],
    "drop": ["robot", "package", "room"],
    "verify_delivery": ["robot", "package", "room"],
}


@dataclass(frozen=True)
class DifficultySpec:
    rooms: int
    packages: int
    distractors: int
    deliveries: int
    obstructed: int
    unstable: int
    dirty_gripper: bool


class LinearLogisticsTaskGenerator:
    """Generate linear household logistics tasks with typed objects and PDDL."""

    def __init__(self, include_distractors: bool = True) -> None:
        self.include_distractors = include_distractors
        self._difficulty_specs: Dict[str, DifficultySpec] = {
            "easy": DifficultySpec(
                rooms=3, packages=1, distractors=1, deliveries=1,
                obstructed=0, unstable=0, dirty_gripper=False,
            ),
            "medium": DifficultySpec(
                rooms=4, packages=2, distractors=1, deliveries=2,
                obstructed=1, unstable=0, dirty_gripper=True,
            ),
            "hard": DifficultySpec(
                rooms=5, packages=3, distractors=2, deliveries=3,
                obstructed=1, unstable=1, dirty_gripper=True,
            ),
            "very_hard": DifficultySpec(
                rooms=8, packages=5, distractors=4, deliveries=5,
                obstructed=2, unstable=2, dirty_gripper=True,
            ),
            "extreme": DifficultySpec(
                rooms=12, packages=8, distractors=6, deliveries=8,
                obstructed=3, unstable=3, dirty_gripper=True,
            ),
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
            self._fact("mobility_ready", robot),
        ]
        if spec.dirty_gripper:
            initial_facts.append(self._fact("gripper_dirty", robot))
        else:
            initial_facts.extend([
                self._fact("gripper_inspected", robot),
                self._fact("gripper_clean", robot),
                self._fact("gripper_calibrated", robot),
            ])
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

        delivery_packages = [delivery["package"] for delivery in deliveries[: spec.deliveries]]
        obstructed_packages = set(rng.sample(
            delivery_packages,
            k=min(spec.obstructed, len(delivery_packages)),
        ))
        unstable_candidates = [
            package for package in delivery_packages if package not in obstructed_packages
        ]
        if len(unstable_candidates) < min(spec.unstable, len(delivery_packages)):
            unstable_candidates = delivery_packages
        unstable_packages = set(rng.sample(
            unstable_candidates,
            k=min(spec.unstable, len(unstable_candidates)),
        ))

        for package in all_packages:
            if package in obstructed_packages:
                initial_facts.append(self._fact("obstructed", package))
            else:
                initial_facts.append(self._fact("accessible", package))
            if package in unstable_packages:
                initial_facts.append(self._fact("unstable", package))
            else:
                initial_facts.append(self._fact("stable", package))

        connections = self._line_connections(rooms)
        initial_facts.extend(connections)

        goal_facts = [
            self._fact(
                "delivery_verified",
                delivery["package"],
                delivery["destination"],
            )
            for delivery in deliveries[: spec.deliveries]
        ]
        goal_text = self._goal_text(
            deliveries[: spec.deliveries],
            robot,
            rooms,
            sorted(obstructed_packages),
            sorted(unstable_packages),
            spec.dirty_gripper,
        )

        metadata = {
            "difficulty": difficulty,
            "instance_index": instance_index,
            "deliveries": deliveries[: spec.deliveries],
            "connections": connections,
            "obstructed_packages": sorted(obstructed_packages),
            "unstable_packages": sorted(unstable_packages),
            "dirty_gripper": spec.dirty_gripper,
            "action_schemas": {
                name: list(argument_types)
                for name, argument_types in ACTION_SCHEMAS.items()
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
        self,
        count: int,
        seed: int,
        difficulty_levels: list[str] | None = None,
    ) -> List[PlanningTaskInstance]:
        """Generate a batch of instances cycling through difficulty levels.

        Args:
            count: Total number of instances to generate.
            seed: Base random seed.
            difficulty_levels: Subset of difficulty names to cycle through.
                Defaults to all registered difficulties.

        Returns:
            List of PlanningTaskInstance objects.
        """
        difficulties = difficulty_levels or list(self._difficulty_specs.keys())
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

    def generate_balanced_dataset(
        self,
        seed: int,
        difficulty_levels: list[str] | None = None,
        instances_per_level: int | None = None,
        total_instances: int | None = None,
    ) -> List[PlanningTaskInstance]:
        """Generate a deterministic balanced dataset by difficulty level.

        Args:
            seed: Base random seed.
            difficulty_levels: Ordered difficulty names to include.
            instances_per_level: Number of instances for each difficulty.
            total_instances: Total number of instances. Must divide evenly by
                the number of selected difficulty levels.

        Returns:
            Tasks ordered by per-level index, then difficulty order. For five
            levels and two instances per level, the order is easy-0,
            medium-0, ..., extreme-0, easy-1, ..., extreme-1.

        Raises:
            ValueError: If counts are missing, invalid, or not evenly balanced.
        """
        difficulties = difficulty_levels or list(self._difficulty_specs.keys())
        if not difficulties:
            raise ValueError("At least one difficulty level is required.")
        for difficulty in difficulties:
            if difficulty not in self._difficulty_specs:
                raise ValueError(f"Unsupported difficulty: {difficulty}")

        if instances_per_level is None:
            if total_instances is None:
                raise ValueError(
                    "Either instances_per_level or total_instances is required."
                )
            if total_instances % len(difficulties) != 0:
                raise ValueError(
                    "total_instances must divide evenly across difficulty_levels "
                    f"({total_instances} over {len(difficulties)} levels)."
                )
            instances_per_level = total_instances // len(difficulties)

        if instances_per_level < 1:
            raise ValueError("instances_per_level must be at least 1.")

        dataset: List[PlanningTaskInstance] = []
        for instance_index in range(instances_per_level):
            for difficulty in difficulties:
                dataset.append(
                    self.generate_instance(
                        seed=seed,
                        difficulty=difficulty,
                        instance_index=instance_index,
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
    def _goal_text(
        deliveries: List[dict],
        robot: str,
        rooms: List[str],
        obstructed_packages: List[str],
        unstable_packages: List[str],
        dirty_gripper: bool,
    ) -> str:
        clauses = [
            f"deliver and verify {delivery['package']} from "
            f"{delivery['source']} to {delivery['destination']}"
            for delivery in deliveries
        ]
        clause_text = "; ".join(clauses)
        constraints: List[str] = []
        if dirty_gripper:
            constraints.append("the gripper starts dirty and must be inspected, cleaned, and calibrated")
        if obstructed_packages:
            constraints.append(
                "clear obstructions before pickup for "
                + ", ".join(obstructed_packages)
            )
        if unstable_packages:
            constraints.append(
                "use regrasp verification for unstable packages "
                + ", ".join(unstable_packages)
            )
        constraint_text = " ".join(constraints)
        return (
            f"Using {robot}, complete the following deliveries in the linear room layout "
            f"{', '.join(rooms)}: {clause_text}. {constraint_text}".strip()
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
    (gripper-inspected ?r - robot)
    (gripper-dirty ?r - robot)
    (gripper-clean ?r - robot)
    (gripper-calibrated ?r - robot)
    (mobility-ready ?r - robot)
    (package-inspected ?p - package)
    (obstructed ?p - package)
    (accessible ?p - package)
    (stable ?p - package)
    (unstable ?p - package)
    (grasp-verified ?r - robot ?p - package)
    (delivery-verified ?p - package ?room - room)
  )

  (:action move
    :parameters (?r - robot ?from - room ?to - room)
    :precondition (and
      (at-robot ?r ?from)
      (connected ?from ?to)
      (mobility-ready ?r)
    )
    :effect (and
      (not (at-robot ?r ?from))
      (at-robot ?r ?to)
    )
  )

  (:action inspect_gripper
    :parameters (?r - robot ?room - room)
    :precondition (and
      (at-robot ?r ?room)
    )
    :effect (and
      (gripper-inspected ?r)
    )
  )

  (:action clean_gripper
    :parameters (?r - robot ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (gripper-inspected ?r)
      (gripper-dirty ?r)
    )
    :effect (and
      (not (gripper-dirty ?r))
      (gripper-clean ?r)
    )
  )

  (:action calibrate_gripper
    :parameters (?r - robot ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (gripper-clean ?r)
    )
    :effect (and
      (gripper-calibrated ?r)
    )
  )

  (:action inspect_package
    :parameters (?r - robot ?p - package ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (at-package ?p ?room)
    )
    :effect (and
      (package-inspected ?p)
    )
  )

  (:action clear_obstruction
    :parameters (?r - robot ?p - package ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (at-package ?p ?room)
      (package-inspected ?p)
      (obstructed ?p)
    )
    :effect (and
      (not (obstructed ?p))
      (accessible ?p)
    )
  )

  (:action pickup
    :parameters (?r - robot ?p - package ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (at-package ?p ?room)
      (handempty ?r)
      (gripper-inspected ?r)
      (gripper-clean ?r)
      (gripper-calibrated ?r)
      (package-inspected ?p)
      (accessible ?p)
    )
    :effect (and
      (not (at-package ?p ?room))
      (not (handempty ?r))
      (not (mobility-ready ?r))
      (holding ?r ?p)
    )
  )

  (:action verify_grasp
    :parameters (?r - robot ?p - package ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (holding ?r ?p)
      (stable ?p)
    )
    :effect (and
      (grasp-verified ?r ?p)
      (mobility-ready ?r)
    )
  )

  (:action regrasp
    :parameters (?r - robot ?p - package ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (holding ?r ?p)
      (unstable ?p)
    )
    :effect (and
      (grasp-verified ?r ?p)
      (mobility-ready ?r)
    )
  )

  (:action drop
    :parameters (?r - robot ?p - package ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (holding ?r ?p)
      (grasp-verified ?r ?p)
    )
    :effect (and
      (not (holding ?r ?p))
      (not (grasp-verified ?r ?p))
      (handempty ?r)
      (at-package ?p ?room)
    )
  )

  (:action verify_delivery
    :parameters (?r - robot ?p - package ?room - room)
    :precondition (and
      (at-robot ?r ?room)
      (at-package ?p ?room)
    )
    :effect (and
      (delivery-verified ?p ?room)
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
        if name == "gripper_inspected":
            return f"(gripper-inspected {tokens[0]})"
        if name == "gripper_dirty":
            return f"(gripper-dirty {tokens[0]})"
        if name == "gripper_clean":
            return f"(gripper-clean {tokens[0]})"
        if name == "gripper_calibrated":
            return f"(gripper-calibrated {tokens[0]})"
        if name == "mobility_ready":
            return f"(mobility-ready {tokens[0]})"
        if name == "package_inspected":
            return f"(package-inspected {tokens[0]})"
        if name == "obstructed":
            return f"(obstructed {tokens[0]})"
        if name == "accessible":
            return f"(accessible {tokens[0]})"
        if name == "stable":
            return f"(stable {tokens[0]})"
        if name == "unstable":
            return f"(unstable {tokens[0]})"
        if name == "grasp_verified":
            return f"(grasp-verified {tokens[0]} {tokens[1]})"
        if name == "delivery_verified":
            return f"(delivery-verified {tokens[0]} {tokens[1]})"
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
    parser.add_argument("--instances-per-level", type=int, default=None)
    parser.add_argument("--domain", type=str, default="logistics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/instances")
    parser.add_argument("--include-distractors", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    generator = LinearLogisticsTaskGenerator(
        include_distractors=args.include_distractors
    )
    if args.instances_per_level is not None:
        dataset = generator.generate_balanced_dataset(
            seed=args.seed,
            instances_per_level=args.instances_per_level,
        )
    else:
        dataset = generator.generate_balanced_dataset(
            seed=args.seed,
            total_instances=args.n_instances,
        )
    paths = save_dataset(dataset, Path(args.output_dir))
    print(f"Generated {len(paths)} instances in {args.output_dir}/")


if __name__ == "__main__":
    cli_main()
