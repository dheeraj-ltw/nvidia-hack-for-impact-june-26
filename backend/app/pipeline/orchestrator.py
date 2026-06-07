"""Per-session orchestrator (live audio path).

Owns one patrol session's live audio: transcribes each audio clip, labels the speaker, and
emits transcript events. Video understanding is deferred to a session-end summarization pass
(see app.api.sessions.summarize_session_video) rather than running live per frame.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.ai.base import AIService
from app.ai.speaker_id import label_dominant_speaker
from app.models.events import PatrolEvent, TranscriptEvent

logger = logging.getLogger(__name__)

EmitCallback = Callable[[PatrolEvent], Awaitable[None]]


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
        # When an officer is enrolled, label each transcribed clip officer-vs-subject live by
        # matching the clip's dominant diarized voice against this embedding (see
        # label_dominant_speaker). The post-session pass refines this into consistent
        # officer/person1/person2 numbering. No enrollment => every clip is the officer.
        self._officer_embedding = officer_embedding

    async def handle_audio(self, audio_chunk: bytes, timestamp: float) -> None:
        logger.info("← audio clip ts=%.2f (%d bytes)", timestamp, len(audio_chunk))
        result = await self._ai_service.transcribe(audio_chunk)
        if not result.text:
            logger.info("  transcribe → (nothing recognized)")
            return
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
