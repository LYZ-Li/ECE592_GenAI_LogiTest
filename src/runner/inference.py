"""Prompting, parsing, and prompt construction for the memory compression study."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import List

from src.common.contracts import MemoryRecord, ModelResponse, PlanStep, PlanningTaskInstance
from src.runner.planner import SymbolicPlanExecutor

_ROOM_ADJECTIVES = [
    "spacious", "dimly-lit", "cluttered", "well-organized", "dusty",
    "freshly-painted", "narrow", "high-ceilinged", "cozy", "industrial",
]

_ROOM_FEATURES = [
    "A large window overlooks the courtyard.",
    "Several shelves line the east wall, filled with miscellaneous items.",
    "The floor is covered with worn tile in a checkered pattern.",
    "An old clock on the wall ticks quietly.",
    "Fluorescent lights hum overhead, casting a bluish glow.",
    "A faded poster is taped to the door.",
    "The room smells faintly of cleaning solution.",
    "A stack of empty boxes sits in the corner.",
    "Cable management trays run along the ceiling.",
    "The ventilation duct rattles periodically.",
    "A small whiteboard hangs on the wall with faded writing.",
    "The paint on the walls is peeling near the baseboards.",
    "A fire extinguisher is mounted beside the doorframe.",
    "There is a slight echo when you walk across the hard floor.",
    "Daylight filters through frosted glass panels above the door.",
]

INCIDENT_HISTORY_LIBRARY = [
    "Prior incident: Dirty gripper residue was discovered during inspection, requiring cleaning before pickup.",
    "Prior incident: A smooth package failed the initial grasp check, requiring a regrasp before transport.",
    "Prior incident: Two stacked items shifted after contact, requiring the target package to be re-secured.",
    "Prior incident: An obstructing package blocked access to the target, requiring obstruction clearing first.",
    "Prior incident: Gripper calibration drift was detected after a long move, requiring recalibration.",
    "Prior incident: A package label was partially occluded, requiring package inspection before pickup.",
    "Prior incident: A package slipped during grasp verification, requiring regrasp before drop.",
    "Prior incident: A destination bin required re-inspection before final delivery verification.",
    "Prior incident: Dust on the gripper caused a weak hold, requiring clean-and-calibrate recovery.",
    "Prior incident: A nearby distractor package was mistaken for the target, requiring an object grounding check.",
]


def build_incident_history_records(
    task: PlanningTaskInstance,
    seed: int,
    count: int,
) -> List[MemoryRecord]:
    """Build deterministic operational-noise records for a task prompt."""
    if count <= 0:
        return []
    rng = random.Random(f"{seed}:{task.task_id}:incident_history")
    indices = list(range(len(INCIDENT_HISTORY_LIBRARY)))
    rng.shuffle(indices)
    selected = indices[: min(count, len(indices))]
    return [
        MemoryRecord(
            step_index=-(position + 1),
            text=INCIDENT_HISTORY_LIBRARY[index],
            tags=["incident_history", "background_noise"],
            token_count=0,
        )
        for position, index in enumerate(selected)
    ]


@dataclass
class PromptBuilder:
    """Build prompts for both direct-action and chain-of-thought modes."""

    context_padding: bool = False
    padding_seed: int = 42
    executor: SymbolicPlanExecutor = field(default_factory=SymbolicPlanExecutor)

    def _generate_room_description(self, room_name: str) -> str:
        """Generate a deterministic irrelevant environmental description."""
        rng = random.Random(f"{self.padding_seed}:{room_name}")
        adj = rng.choice(_ROOM_ADJECTIVES)
        features = rng.sample(_ROOM_FEATURES, k=3)
        return (
            f"[Environment: {room_name}] This is a {adj} room. "
            + " ".join(features)
        )

    def _pad_memory_context(self, memory_context: str) -> str:
        """Interleave room descriptions between history steps."""
        if not memory_context or memory_context == "[no prior executed steps]":
            return memory_context
        blocks = memory_context.split("\n\n")
        padded: list[str] = []
        for block in blocks:
            padded.append(block)
            rooms = re.findall(r"room_\d+", block)
            if rooms:
                rng = random.Random(f"{self.padding_seed}:{rooms[-1]}:{len(padded)}")
                room = rng.choice(rooms)
                padded.append(self._generate_room_description(room))
        return "\n\n".join(padded)

    def system_prompt(self, mode: str = "direct_action") -> str:
        """Return mode-appropriate system prompt.

        Args:
            mode: Either "direct_action" or "cot".

        Returns:
            System prompt string.
        """
        if mode == "cot":
            return (
                "You are a precise planning assistant. "
                "Think step by step about the current state and which action to take next. "
                "Keep the reasoning concise and avoid repeating prior facts. "
                "After your reasoning, output your chosen action as a single JSON object "
                "on the final line in this exact format:\n"
                '{"action_name": "<action>", "arguments": ["<arg>", "..."]}\n'
                "Always finish with the JSON object, even if you are uncertain. "
                "Do not use markdown or code fences around the JSON."
            )
        return (
            "You are a precise planning assistant. "
            "Return exactly one bare JSON object and nothing else. "
            'The object must match {"action_name": "<action>", "arguments": ["<arg>", "..."]}. '
            "Do not use markdown, code fences, explanations, or extra keys."
        )

    def user_prompt(
        self,
        task: PlanningTaskInstance,
        memory_context: str,
        prompt_mode: str,
        current_state_facts: List[str],
        incident_context: str = "",
        repair_context: str | None = None,
    ) -> str:
        """Build the user prompt with state, goal, history, and instruction.

        Args:
            task: The current planning task.
            memory_context: Formatted memory context string.
            prompt_mode: Either "direct_action" or "cot".
            current_state_facts: Current world state as a list of fact strings.
            incident_context: Deterministic background incident records.
            repair_context: Optional symbolic feedback for a rejected action.

        Returns:
            Formatted user prompt string.
        """
        action_lines = self._action_dependency_lines(task)
        memory_block = memory_context if memory_context else "[no prior executed steps]"
        incident_block = incident_context if incident_context else "[none]"

        if self.context_padding:
            memory_block = self._pad_memory_context(memory_block)

        if prompt_mode == "cot":
            instruction = (
                "Reason step by step briefly about the current state, "
                "then output your action as bare JSON on the final line. "
                "Do not stop before writing the final JSON object."
            )
        else:
            instruction = "Return the single next action as bare JSON only."
        if repair_context:
            instruction = (
                "The previous action was rejected by symbolic validation. "
                "Return one legal replacement action as bare JSON only."
            )

        sections = [
            f"Task: {task.goal_text}",
            f"Goal facts: {'; '.join(task.goal_facts)}",
            f"Valid objects: {', '.join(task.valid_objects)}",
            "[Available Actions]",
            "\n".join(action_lines),
            "Choose only actions whose listed requirements are satisfied by Current State.",
            "[Current State]",
            "; ".join(current_state_facts),
        ]

        if self.context_padding:
            rooms = [obj for obj in task.valid_objects if obj.startswith("room_")]
            room_descriptions = [self._generate_room_description(r) for r in rooms]
            sections.append("[Environmental Details]")
            sections.append("\n".join(room_descriptions))

        sections.extend([
            "[Background Incident Records]",
            incident_block,
            "[Executed History]",
            memory_block,
        ])

        if repair_context:
            sections.extend([
                "[Symbolic Feedback]",
                repair_context,
            ])

        sections.extend([
            "[Instruction]",
            instruction,
        ])

        return "\n\n".join(sections)

    def _action_dependency_lines(self, task: PlanningTaskInstance) -> List[str]:
        """Format action schemas with symbolic requirements and effects."""
        action_schemas = task.metadata.get("action_schemas", {})
        lines: List[str] = []
        for action_name, argument_types in sorted(action_schemas.items()):
            arguments = _template_arguments(action_name, argument_types)
            step = self.executor.annotate_step(
                task,
                PlanStep(index=0, action_name=action_name, arguments=arguments),
            )
            requires = "; ".join(step.preconditions) if step.preconditions else "none"
            effects = "; ".join(step.effects) if step.effects else "none"
            lines.append(
                f"- {action_name}({', '.join(argument_types)}) "
                f"requires: {requires}; effects: {effects}"
            )
        return lines


def _template_arguments(action_name: str, argument_types: List[str]) -> List[str]:
    """Return readable placeholder arguments for action dependency templates."""
    named_templates = {
        "move": ["?robot", "?from_room", "?to_room"],
        "inspect_gripper": ["?robot", "?room"],
        "clean_gripper": ["?robot", "?room"],
        "calibrate_gripper": ["?robot", "?room"],
        "inspect_package": ["?robot", "?package", "?room"],
        "clear_obstruction": ["?robot", "?package", "?room"],
        "pickup": ["?robot", "?package", "?room"],
        "verify_grasp": ["?robot", "?package", "?room"],
        "regrasp": ["?robot", "?package", "?room"],
        "drop": ["?robot", "?package", "?room"],
        "verify_delivery": ["?robot", "?package", "?room"],
    }
    if action_name in named_templates:
        return named_templates[action_name]

    counts: dict[str, int] = {}
    arguments: List[str] = []
    for argument_type in argument_types:
        counts[argument_type] = counts.get(argument_type, 0) + 1
        suffix = "" if argument_types.count(argument_type) == 1 else f"_{counts[argument_type]}"
        arguments.append(f"?{argument_type}{suffix}")
    return arguments


class StrictActionParser:
    """Parse a single bare-JSON action object from model output.

    Supports two parsing strategies:
      - direct_action: entire output must be a single JSON object
      - cot: scan from the end for the last valid JSON line with
        action_name + arguments keys
    """

    def __init__(self, valid_action_names: List[str]) -> None:
        self._valid_action_names = set(valid_action_names)

    def parse(self, raw_text: str, mode: str = "direct_action") -> PlanStep:
        """Parse model output into a PlanStep.

        Args:
            raw_text: Raw model output text.
            mode: "direct_action" or "cot".

        Returns:
            Parsed PlanStep with index=0 (caller should set the real index).

        Raises:
            ValueError: If parsing fails or action is invalid.
        """
        if mode == "cot":
            return self._parse_cot(raw_text)
        return self._parse_direct(raw_text)

    def _parse_direct(self, raw_text: str) -> PlanStep:
        """Parse output as JSON, falling back to extraction if needed.

        Tries the full output first. If that fails, scans for an embedded
        JSON object with ``action_name`` and ``arguments`` keys (handles
        models that prepend reasoning text before the JSON).
        """
        stripped = raw_text.strip()
        # Fast path: entire output is valid JSON
        try:
            payload = json.loads(stripped)
            return self._validate_payload(payload)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract JSON object from anywhere in the text
        return self._extract_json_action(stripped)

    def _parse_cot(self, raw_text: str) -> PlanStep:
        """Scan from the end for the last valid JSON action line."""
        return self._extract_json_action(raw_text.strip(), label="CoT")

    def _extract_json_action(self, text: str, label: str = "model") -> PlanStep:
        """Find the last valid ``{action_name, arguments}`` JSON in *text*."""
        lines = text.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "action_name" in payload and "arguments" in payload:
                return self._validate_payload(payload)
        raise ValueError(
            f"No valid JSON action found in {label} output. "
            f"Last 500 chars: {text[-500:]}"
        )

    def _validate_payload(self, payload: dict) -> PlanStep:
        """Validate a parsed JSON payload and convert to PlanStep."""
        if not isinstance(payload, dict):
            raise ValueError("Model output must be a JSON object.")
        if set(payload.keys()) != {"action_name", "arguments"}:
            raise ValueError(
                "Model output must contain only `action_name` and `arguments` keys. "
                f"Got: {set(payload.keys())}"
            )

        action_name = payload["action_name"]
        arguments = payload["arguments"]
        if not isinstance(action_name, str) or not action_name.strip():
            raise ValueError("`action_name` must be a non-empty string.")
        if action_name not in self._valid_action_names:
            raise ValueError(f"Unknown action `{action_name}`.")
        if not isinstance(arguments, list) or any(
            not isinstance(arg, str) or not arg for arg in arguments
        ):
            raise ValueError("`arguments` must be a list of non-empty strings.")

        return PlanStep(index=0, action_name=action_name, arguments=list(arguments))
