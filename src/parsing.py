"""Robust extraction of the <think> block and trailing JSON object from model output.

Handles: markdown code fences, pretty-printed JSON, and extra prose/fences after
the object (extracts by brace-matching the first balanced {...}).
"""
from __future__ import annotations

import json
import re

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
REQUIRED = {"legal_basis", "suggested_action", "tone_guidance",
            "escalation_risk", "rights_reminder", "next_steps"}

# Map common escalation_risk synonyms onto the required {low, medium, high} enum.
_RISK_MAP = {
    "low": "low", "minimal": "low", "none": "low",
    "medium": "medium", "moderate": "medium", "elevated": "medium",
    "high": "high", "severe": "high", "critical": "high", "very high": "high",
}


def _normalise_risk(obj: dict) -> None:
    val = str(obj.get("escalation_risk", "")).strip().lower()
    if val in _RISK_MAP:
        obj["escalation_risk"] = _RISK_MAP[val]


def _first_json_object(text: str) -> dict | None:
    """Return the first balanced {...} parsed as JSON, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break  # try next '{'
        start = text.find("{", start + 1)
    return None


def parse_output(text: str) -> tuple[str, dict] | None:
    """Return (think_block, parsed_json) if the output is well-formed, else None."""
    text = (text or "").strip()
    m = THINK_RE.search(text)
    if not m:
        return None
    think = m.group(0)
    rest = text[m.end():]
    obj = _first_json_object(rest)
    if obj is None or not REQUIRED.issubset(obj.keys()):
        return None
    _normalise_risk(obj)
    return think, obj


def normalise_assistant(text: str) -> str | None:
    """Validate and re-serialise as '<think>...</think>\\n{compact json}'."""
    parsed = parse_output(text)
    if parsed is None:
        return None
    think, obj = parsed
    return f"{think}\n{json.dumps(obj, ensure_ascii=False)}"


def get_json(assistant: str) -> dict | None:
    """Parse the JSON object from an already-normalised assistant turn."""
    parsed = parse_output(assistant)
    return parsed[1] if parsed else None
