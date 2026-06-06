"""Per-session orchestrator.

Owns one patrol session: receives frames/audio, fans them through the AIService, and
emits typed PatrolEvents. Frame analysis and reasoning are throttled to a sane cadence so we
never queue faster than the AI can respond — the realtime loop, not transport, is the budget.
"""

from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable

from app.ai.base import AIService, Frame, ReasoningInput
from app.models.events import (
    DetectionEvent,
    GuidanceEvent,
    PatrolEvent,
    SpeechEvent,
    TranscriptEvent,
)

EmitCallback = Callable[[PatrolEvent], Awaitable[None]]

# Analyze at most this often, regardless of how fast frames arrive.
_ANALYZE_INTERVAL_SECONDS = 0.7
# Bound how much transcript history we feed the reasoner, to cap prompt size.
_TRANSCRIPT_CONTEXT_CHARS = 2000


class SessionPipeline:
    def __init__(self, ai_service: AIService, emit: EmitCallback) -> None:
        self._ai_service = ai_service
        self._emit = emit
        self._is_analyzing = False
        self._last_analyze_timestamp = 0.0
        self._transcript = ""

    async def handle_frame(self, frame: Frame) -> None:
        """Throttled, drop-if-busy frame handling — keeps latency bounded under load."""
        timestamp = frame.ts
        too_soon = (timestamp - self._last_analyze_timestamp) < _ANALYZE_INTERVAL_SECONDS
        if self._is_analyzing or too_soon:
            return
        self._is_analyzing = True
        self._last_analyze_timestamp = timestamp
        try:
            boxes, scene_summary = await self._ai_service.analyze_frame(frame)
            await self._emit(DetectionEvent(ts=timestamp, boxes=boxes, summary=scene_summary))

            guidance = await self._ai_service.reason(
                ReasoningInput(
                    ts=timestamp,
                    transcript=self._transcript[-_TRANSCRIPT_CONTEXT_CHARS:],
                    scene_summary=scene_summary,
                    detections=boxes,
                )
            )
            if guidance is not None:
                await self._emit(guidance)
                await self._emit_speech(guidance)
        finally:
            self._is_analyzing = False

    async def handle_audio(self, audio_chunk: bytes, timestamp: float) -> None:
        text, is_final = await self._ai_service.transcribe(audio_chunk)
        if not text:
            return
        if is_final:
            self._transcript += " " + text
        await self._emit(
            TranscriptEvent(ts=timestamp, text=text, speaker="officer", is_final=is_final)
        )

    async def _emit_speech(self, guidance: GuidanceEvent) -> None:
        audio_bytes = await self._ai_service.speak(guidance.suggestion)
        await self._emit(
            SpeechEvent(
                ts=guidance.ts,
                audio_b64=base64.b64encode(audio_bytes).decode("ascii"),
                text=guidance.suggestion,
            )
        )
