"""The pluggable AI boundary.

Everything model-related lives behind `AIService`. Selecting `AI_BACKEND` swaps the
implementation (stub / live NVIDIA NIM + ElevenLabs) with no changes to the pipeline or UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.models.events import BoundingBox, GuidanceEvent


@dataclass(slots=True)
class Frame:
    """A single decoded video frame plus capture timestamp."""

    ts: float
    jpeg: bytes
    width: int
    height: int


@dataclass(slots=True)
class Transcription:
    """A transcribed audio clip: the text plus optional per-word diarization.

    `words` carries ElevenLabs' word-level speaker split for *this clip* — used to label the
    speaker by isolating the dominant voice's audio before matching it to the officer. Empty
    when the backend doesn't diarize (e.g. the stub); the speaker then defaults to the officer.
    """

    text: str
    is_final: bool
    words: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class ReasoningInput:
    """Aggregated context handed to the reasoner to produce law-aligned guidance."""

    ts: float
    transcript: str
    scene_summary: str
    detections: list[BoundingBox] = field(default_factory=list)
    location: str | None = None


@runtime_checkable
class AIService(Protocol):
    """Contract for all AI capabilities. Implementations must be async and stateless."""

    async def transcribe(self, audio_chunk: bytes) -> Transcription:
        """Transcribe an audio chunk. Empty text => nothing recognized."""
        ...

    async def analyze_frame(self, frame: Frame) -> tuple[list[BoundingBox], str]:
        """Return (detections, scene_summary) for a frame."""
        ...

    async def reason(self, context: ReasoningInput) -> GuidanceEvent | None:
        """Produce assistive, cited guidance from aggregated context, or None if nothing useful."""
        ...

    async def speak(self, text: str) -> bytes:
        """Return mp3 audio bytes for the given text (TTS)."""
        ...
