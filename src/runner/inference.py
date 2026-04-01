"""Prompting, parsing, and prompt construction for the memory compression study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from src.common.contracts import ModelResponse, PlanStep, PlanningTaskInstance


@dataclass
class PromptBuilder:
    """Build prompts for both direct-action and chain-of-thought modes."""

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
                "After your reasoning, output your chosen action as a single JSON object "
                "on its own line in this exact format:\n"
                '{"action_name": "<action>", "arguments": ["<arg>", "..."]}\n'
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
    ) -> str:
        """Build the user prompt with state, goal, history, and instruction.

        Args:
            task: The current planning task.
            memory_context: Formatted memory context string.
            prompt_mode: Either "direct_action" or "cot".
            current_state_facts: Current world state as a list of fact strings.

        Returns:
            Formatted user prompt string.
        """
        action_schemas = task.metadata.get("action_schemas", {})
        action_lines = [
            f"- {action_name}({', '.join(argument_types)})"
            for action_name, argument_types in sorted(action_schemas.items())
        ]
        memory_block = memory_context if memory_context else "[no prior executed steps]"

        if prompt_mode == "cot":
            instruction = (
                "Reason step by step about the current state, "
                "then output your action as bare JSON on the final line."
            )
        else:
            instruction = "Return the single next action as bare JSON only."

        return "\n\n".join(
            [
                f"Task: {task.goal_text}",
                f"Goal facts: {'; '.join(task.goal_facts)}",
                f"Valid objects: {', '.join(task.valid_objects)}",
                "[Available Actions]",
                "\n".join(action_lines),
                "[Current State]",
                "; ".join(current_state_facts),
                "[Executed History]",
                memory_block,
                "[Instruction]",
                instruction,
            ]
        )


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
        """Parse entire output as a single JSON object (fix Q2: strip first)."""
        stripped = raw_text.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model output is not valid JSON: {stripped[:200]}") from exc
        return self._validate_payload(payload)

    def _parse_cot(self, raw_text: str) -> PlanStep:
        """Scan from the end for the last valid JSON action line."""
        lines = raw_text.strip().split("\n")
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
            "No valid JSON action found in CoT output. "
            f"Last 500 chars: {raw_text[-500:]}"
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
