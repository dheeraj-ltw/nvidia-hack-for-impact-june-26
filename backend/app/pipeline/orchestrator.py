"""Per-session orchestrator.

Owns one patrol session: receives frames/audio, fans them through the AIService, and
emits typed PatrolEvents. Frame analysis and reasoning are throttled to a sane cadence so we
never queue faster than the AI can respond — the realtime loop, not transport, is the budget.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable

from app.ai.base import AIService, Frame, ReasoningInput
from app.ai.speaker_id import match_clip
from app.models.events import (
    DetectionEvent,
    GuidanceEvent,
    PatrolEvent,
    SpeechEvent,
    TranscriptEvent,
)

logger = logging.getLogger(__name__)

EmitCallback = Callable[[PatrolEvent], Awaitable[None]]

# Analyze at most this often, regardless of how fast frames arrive.
_ANALYZE_INTERVAL_SECONDS = 0.7
# Bound how much transcript history we feed the reasoner, to cap prompt size.
_TRANSCRIPT_CONTEXT_CHARS = 2000


class SessionPipeline:
    def __init__(
        self,
        ai_service: AIService,
        emit: EmitCallback,
        *,
        officer_embedding: list[float] | None = None,
    ) -> None:
        self._ai_service = ai_service
        self._emit = emit
        self._is_analyzing = False
        self._last_analyze_timestamp = 0.0
        self._transcript = ""
        # When set, each transcribed clip is matched against the officer's enrolled voice to
        # label the speaker live (officer vs subject). The post-session pass refines this into
        # consistent officer/person1/person2 numbering across the whole conversation.
        self._officer_embedding = officer_embedding

    async def handle_frame(self, frame: Frame) -> None:
        """Throttled, drop-if-busy frame handling — keeps latency bounded under load."""
        timestamp = frame.ts
        too_soon = (timestamp - self._last_analyze_timestamp) < _ANALYZE_INTERVAL_SECONDS
        if self._is_analyzing or too_soon:
            return
        self._is_analyzing = True
        self._last_analyze_timestamp = timestamp
        try:
            logger.info(
                "← frame ts=%.2f %dx%d (%d bytes)",
                timestamp,
                frame.width,
                frame.height,
                len(frame.jpeg),
            )
            boxes, scene_summary = await self._ai_service.analyze_frame(frame)
            logger.info(
                "  analyze_frame → %d detection(s), scene=%r", len(boxes), scene_summary or ""
            )
            await self._emit(DetectionEvent(ts=timestamp, boxes=boxes, summary=scene_summary))

            transcript_context = self._transcript[-_TRANSCRIPT_CONTEXT_CHARS:]
            logger.info(
                "  reason ← scene=%r transcript=%d chars",
                scene_summary or "",
                len(transcript_context),
            )
            guidance = await self._ai_service.reason(
                ReasoningInput(
                    ts=timestamp,
                    transcript=transcript_context,
                    scene_summary=scene_summary,
                    detections=boxes,
                )
            )
            if guidance is not None:
                logger.info(
                    "  reason → guidance [%s] %r (%d citation(s))",
                    guidance.severity.value,
                    guidance.suggestion,
                    len(guidance.citations),
                )
                await self._emit(guidance)
                await self._emit_speech(guidance)
            else:
                logger.info("  reason → no guidance")
        finally:
            self._is_analyzing = False

    async def handle_audio(self, audio_chunk: bytes, timestamp: float) -> None:
        logger.info("← audio clip ts=%.2f (%d bytes)", timestamp, len(audio_chunk))
        text, is_final = await self._ai_service.transcribe(audio_chunk)
        if not text:
            logger.info("  transcribe → (nothing recognized)")
            return
        if is_final:
            self._transcript += " " + text
        speaker = await self._label_speaker(audio_chunk)
        logger.info("  transcribe → [%s] %r (final=%s)", speaker, text, is_final)
        await self._emit(
            TranscriptEvent(ts=timestamp, text=text, speaker=speaker, is_final=is_final)
        )

    async def _label_speaker(self, clip: bytes) -> str:
        """Live speaker label for a clip: 'officer'/'subject' if enrolled, else 'officer'."""
        if self._officer_embedding is None:
            return "officer"  # no enrollment to compare against
        try:
            label, _ = await asyncio.to_thread(match_clip, clip, self._officer_embedding)
        except Exception as error:  # noqa: BLE001 - never let matching break the transcript
            logger.warning("Live speaker match failed: %s", error)
            return "unknown"
        return label

    async def _emit_speech(self, guidance: GuidanceEvent) -> None:
        audio_bytes = await self._ai_service.speak(guidance.suggestion)
        logger.info("  speak → %d bytes of mp3 audio", len(audio_bytes))
        await self._emit(
            SpeechEvent(
                ts=guidance.ts,
                audio_b64=base64.b64encode(audio_bytes).decode("ascii"),
                text=guidance.suggestion,
            )
        )
