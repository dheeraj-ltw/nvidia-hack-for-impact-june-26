"""Wire contract for realtime events.

These models are the single source of truth for what flows over the WebSocket between
the pipeline and the patrol UI. The frontend mirrors these shapes in TypeScript.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class EventType(StrEnum):
    TRANSCRIPT = "transcript"
    DETECTION = "detection"
    GUIDANCE = "guidance"
    ALERT = "alert"
    SPEECH = "speech"
    STATUS = "status"


class Severity(StrEnum):
    INFO = "info"
    CAUTION = "caution"
    CRITICAL = "critical"


class BoundingBox(BaseModel):
    """Normalized [0,1] coordinates so the UI can scale to any video size."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)


class LegalCitation(BaseModel):
    """A reference that backs a piece of guidance — core to law-alignment + auditability."""

    title: str
    reference: str  # e.g. statute / code section
    snippet: str | None = None


class TranscriptEvent(BaseModel):
    type: Literal[EventType.TRANSCRIPT] = EventType.TRANSCRIPT
    ts: float
    text: str
    speaker: Literal["officer", "subject", "unknown"] = "unknown"
    is_final: bool = True


class DetectionEvent(BaseModel):
    type: Literal[EventType.DETECTION] = EventType.DETECTION
    ts: float
    boxes: list[BoundingBox] = Field(default_factory=list)
    summary: str | None = None


class GuidanceEvent(BaseModel):
    """Assistive, cited suggestion. Never a directive — always officer-acknowledged."""

    type: Literal[EventType.GUIDANCE] = EventType.GUIDANCE
    ts: float
    suggestion: str
    rationale: str | None = None
    citations: list[LegalCitation] = Field(default_factory=list)
    severity: Severity = Severity.INFO


class AlertEvent(BaseModel):
    type: Literal[EventType.ALERT] = EventType.ALERT
    ts: float
    message: str
    severity: Severity = Severity.CAUTION


class SpeechEvent(BaseModel):
    """TTS audio for hands-free guidance, base64-encoded (mp3)."""

    type: Literal[EventType.SPEECH] = EventType.SPEECH
    ts: float
    audio_b64: str
    text: str


class StatusEvent(BaseModel):
    type: Literal[EventType.STATUS] = EventType.STATUS
    ts: float
    state: str  # e.g. "connected", "ai_backend:null"
    detail: str | None = None
    session_id: str | None = None


PatrolEvent = (
    TranscriptEvent | DetectionEvent | GuidanceEvent | AlertEvent | SpeechEvent | StatusEvent
)
