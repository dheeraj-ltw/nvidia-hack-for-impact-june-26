"""No-op AI backend.

The default when no model is connected. It recognizes nothing, detects nothing, and
produces no guidance — so the console shows the real camera feed and the session is
recorded for later analysis, without inventing fake output.
"""

from __future__ import annotations

from app.ai.base import Frame, ReasoningInput
from app.models.events import BoundingBox, GuidanceEvent


class NullAIService:
    """Implements the AIService protocol by returning empty results."""

    async def transcribe(self, audio_chunk: bytes) -> tuple[str, bool]:
        return "", False

    async def analyze_frame(self, frame: Frame) -> tuple[list[BoundingBox], str]:
        return [], ""

    async def reason(self, context: ReasoningInput) -> GuidanceEvent | None:
        return None

    async def speak(self, text: str) -> bytes:
        return b""
