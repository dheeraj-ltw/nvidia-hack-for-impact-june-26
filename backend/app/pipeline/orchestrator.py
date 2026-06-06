"""Per-session orchestrator.

Owns one patrol session: receives frames/audio, fans them through the AIService, and
emits typed PatrolEvents. Frame analysis and reasoning are throttled to a sane cadence so we
never queue faster than the AI can respond — the realtime loop, not transport, is the budget.

Frame analysis runs *off* the receive loop (schedule_frame), so a slow reasoning call can
never head-of-line-block audio transcription. Vision (VLM) is not on the live path at all —
it runs post-session over the recorded frames (see app.api.sessions.run_scene_analysis).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable

from app.ai.base import AIService, Frame, ReasoningInput
from app.ai.speaker_id import label_dominant_speaker
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
        # Frame analysis (reasoning) runs as a detached task so it never blocks the receive
        # loop — and thus never delays audio transcription. We keep a handle so the session
        # can await the in-flight one on shutdown.
        self._analyze_task: asyncio.Task[None] | None = None
        # When an officer is enrolled, label each transcribed clip officer-vs-subject live by
        # matching the clip's dominant diarized voice against this embedding (see
        # label_dominant_speaker). The post-session pass refines this into consistent
        # officer/person1/person2 numbering. No enrollment => every clip is the officer.
        self._officer_embedding = officer_embedding

    def schedule_frame(self, frame: Frame) -> None:
        """Throttled, drop-if-busy frame analysis, dispatched off the caller's loop.

        Reasoning can take seconds; running it inline on the WebSocket receive loop would
        delay reading the next audio clip and stall transcription. So we fire it as a detached
        task and return immediately. Drop-if-busy keeps at most one analysis in flight, which
        also bounds how stale the reasoning context can get.
        """
        timestamp = frame.ts
        too_soon = (timestamp - self._last_analyze_timestamp) < _ANALYZE_INTERVAL_SECONDS
        if self._is_analyzing or too_soon:
            return
        self._is_analyzing = True
        self._last_analyze_timestamp = timestamp
        self._analyze_task = asyncio.create_task(self._analyze_frame(frame))

    async def _analyze_frame(self, frame: Frame) -> None:
        timestamp = frame.ts
        try:
            logger.info(
                "← frame ts=%.2f %dx%d (%d bytes)",
                timestamp,
                frame.width,
                frame.height,
                len(frame.jpeg),
            )
            # Vision runs post-session, not here: detections/scene are empty on the live path.
            boxes, scene_summary = await self._ai_service.analyze_frame(frame)
            if boxes or scene_summary:
                await self._emit(DetectionEvent(ts=timestamp, boxes=boxes, summary=scene_summary))

            transcript_context = self._transcript[-_TRANSCRIPT_CONTEXT_CHARS:]
            # Skip the reasoning call entirely until there's dialogue to reason about.
            if not transcript_context.strip():
                logger.info("  reason ← (no transcript yet) — skipping")
                return
            logger.info("  reason ← transcript=%d chars", len(transcript_context))
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
        except Exception as error:  # noqa: BLE001 - detached task; must not crash the session
            logger.warning("Frame analysis failed: %s", error)
        finally:
            self._is_analyzing = False

    async def aclose(self) -> None:
        """Await any in-flight frame analysis so it isn't cancelled mid-emit on shutdown."""
        if self._analyze_task is not None:
            await asyncio.gather(self._analyze_task, return_exceptions=True)

    async def handle_audio(self, audio_chunk: bytes, timestamp: float) -> None:
        logger.info("← audio clip ts=%.2f (%d bytes)", timestamp, len(audio_chunk))
        result = await self._ai_service.transcribe(audio_chunk)
        if not result.text:
            logger.info("  transcribe → (nothing recognized)")
            return
        if result.is_final:
            self._transcript += " " + result.text
        speaker = await self._label_speaker(audio_chunk, result.words)
        logger.info("  transcribe → [%s] %r (final=%s)", speaker, result.text, result.is_final)
        await self._emit(
            TranscriptEvent(
                ts=timestamp, text=result.text, speaker=speaker, is_final=result.is_final
            )
        )

    async def _label_speaker(self, clip: bytes, words: list[dict]) -> str:
        """Live speaker label for a clip: 'officer'/'subject' if enrolled, else 'officer'."""
        if self._officer_embedding is None:
            return "officer"  # no enrollment to compare against
        try:
            label, _ = await asyncio.to_thread(
                label_dominant_speaker, clip, words, self._officer_embedding
            )
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
