"""PoliceAI prompt construction + response parsing.

The fine-tuned reasoner (served by police-llm/) was trained on a fixed chat shape:

  system   : the PoliceAI system prompt (verbatim, below)
  user     : a SCENE CARD describing the incident + a QUERY + OFFICER CONTEXT
  assistant: a <think>...</think> reasoning block, then a single JSON object with
             keys legal_basis / suggested_action / tone_guidance / escalation_risk /
             rights_reminder / next_steps.

To get good output the live prompt must match that training format, so this module
rebuilds the SCENE CARD from the realtime context and parses the structured reply back
into a GuidanceEvent. Keeping it isolated from the HTTP client makes both halves testable
without a model server.
"""

from __future__ import annotations

import json
import re

from app.ai.base import ReasoningInput
from app.models.events import GuidanceEvent, LegalCitation, Severity

# Verbatim from prompts/generation_system.txt — the system prompt the model was trained
# against. Changing this drifts the live prompt away from the fine-tune, so keep it in sync.
SYSTEM_PROMPT = (
    "You are PoliceAI, a real-time legal advisor for on-duty law enforcement officers. "
    "Your role is to provide legally-grounded, de-escalation-focused guidance during active "
    "incidents. You NEVER give an opinion on guilt or innocence. All advice must be phrased as "
    "guidance, not commands. Always cite specific case law or statutes. You must reason step by "
    "step through the legal situation before producing your final structured JSON response."
)

# Maps the model's escalation_risk to the UI severity scale.
_RISK_TO_SEVERITY = {
    "low": Severity.INFO,
    "medium": Severity.CAUTION,
    "high": Severity.CRITICAL,
}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_scene_card(context: ReasoningInput) -> str:
    """Render the realtime context as a SCENE CARD in the model's training format.

    Only fields we actually have at runtime are filled; the scene/detection summary and
    the rolling transcript stand in for the richer synthetic scene cards used in training.
    """
    detections = ", ".join(box.label for box in context.detections) or "none reported"
    lines = [
        "SCENE CARD (live patrol)",
        f"Location: {context.location or 'Unknown'}",
        f"Scene: {context.scene_summary or 'No visual summary available'}",
        f"Visible objects: {detections}",
        "Jurisdiction: England & Wales",
        "",
        "RECENT DIALOGUE:",
        context.transcript.strip() or "(no speech transcribed yet)",
        "",
        "QUERY: Based on the scene and dialogue so far, what is the lawful, "
        "de-escalation-focused next step, and what must I tell the subject?",
    ]
    return "\n".join(lines)


def build_messages(context: ReasoningInput) -> list[dict[str, str]]:
    """The full chat payload for the fine-tuned model: system + SCENE CARD user turn."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_scene_card(context)},
    ]


def _extract_json(content: str) -> dict[str, object] | None:
    """Pull the JSON object that follows the <think> block (fences tolerated)."""
    body = _THINK_BLOCK.sub("", content).strip()
    body = re.sub(r"^```(?:json)?|```$", "", body, flags=re.MULTILINE).strip()
    start = body.find("{")
    end = body.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_guidance(content: str, ts: float) -> GuidanceEvent | None:
    """Parse a raw model completion into a GuidanceEvent, or None if unusable.

    The trained output is a <think> block followed by JSON. We surface the structured
    fields (action + tone) as the suggestion/rationale and the cited legal_basis as a
    citation, so guidance stays auditable. Malformed output yields None rather than a guess.
    """
    parsed = _extract_json(content)
    if not parsed:
        return None

    action = str(parsed.get("suggested_action", "")).strip()
    if not action:
        return None

    tone = str(parsed.get("tone_guidance", "")).strip()
    legal_basis = str(parsed.get("legal_basis", "")).strip()
    rights = str(parsed.get("rights_reminder", "")).strip()
    risk = str(parsed.get("escalation_risk", "")).strip().lower()

    rationale_parts = [part for part in (tone, rights) if part]
    citations = (
        [LegalCitation(title="Legal basis", reference=legal_basis)] if legal_basis else []
    )

    return GuidanceEvent(
        ts=ts,
        suggestion=action,
        rationale=" ".join(rationale_parts) or None,
        citations=citations,
        severity=_RISK_TO_SEVERITY.get(risk, Severity.INFO),
    )
