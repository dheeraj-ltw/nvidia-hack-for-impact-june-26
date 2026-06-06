import pathlib

import pytest

from app.ai import policeai
from app.ai.base import ReasoningInput
from app.models.events import BoundingBox, Severity

# The live system prompt must stay byte-identical to the prompt the model was fine-tuned
# against; drift silently degrades guidance. The training prompt lives at the repo root, which
# isn't in the backend container build context — skip there rather than fail.
_TRAINING_PROMPT = pathlib.Path(__file__).resolve().parents[2] / "prompts" / "generation_system.txt"


@pytest.mark.skipif(not _TRAINING_PROMPT.exists(), reason="training prompt not in build context")
def test_system_prompt_matches_training_source() -> None:
    expected = _TRAINING_PROMPT.read_text(encoding="utf-8").strip()
    assert policeai.SYSTEM_PROMPT == expected  # noqa: SIM300 - assert actual == expected reads clearer


def _context() -> ReasoningInput:
    return ReasoningInput(
        ts=12.0,
        transcript="officer: stop there. subject: why am I being stopped?",
        scene_summary="one person near a parked car",
        detections=[BoundingBox(label="person", confidence=0.9, x=0.1, y=0.1, w=0.2, h=0.3)],
        location="High Street, Camden",
    )


def test_build_messages_uses_trained_system_prompt_and_scene_card() -> None:
    messages = policeai.build_messages(_context())
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == policeai.SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "SCENE CARD" in user
    assert "High Street, Camden" in user
    assert "person" in user  # detection label rendered into the card


def test_parse_guidance_extracts_structured_fields() -> None:
    content = (
        "<think>reasoning about s.23 MDA 1971</think>\n"
        '{"legal_basis": "Misuse of Drugs Act 1971, s.23", '
        '"suggested_action": "You may search the subject; explain the grounds first.", '
        '"tone_guidance": "Stay calm and explain each step.", '
        '"escalation_risk": "medium", '
        '"rights_reminder": "The subject may request the search record.", '
        '"next_steps": ["State grounds", "Conduct search", "Record outcome"]}'
    )
    guidance = policeai.parse_guidance(content, ts=12.0)
    assert guidance is not None
    assert guidance.ts == 12.0
    assert "search the subject" in guidance.suggestion
    assert guidance.severity is Severity.CAUTION  # medium -> caution
    assert guidance.citations[0].reference == "Misuse of Drugs Act 1971, s.23"
    assert guidance.rationale and "calm" in guidance.rationale


def test_parse_guidance_tolerates_code_fences() -> None:
    content = (
        "<think>x</think>\n```json\n"
        '{"suggested_action": "Advise the subject of their rights.", '
        '"escalation_risk": "high"}\n```'
    )
    guidance = policeai.parse_guidance(content, ts=1.0)
    assert guidance is not None
    assert guidance.severity is Severity.CRITICAL


def test_parse_guidance_returns_none_on_garbage() -> None:
    assert policeai.parse_guidance("no json here at all", ts=0.0) is None
    assert policeai.parse_guidance('{"escalation_risk": "low"}', ts=0.0) is None  # no action
