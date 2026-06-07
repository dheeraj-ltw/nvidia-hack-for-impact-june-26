"""Per-session recorder.

Streams captured frames and audio straight to object storage as they arrive, records the
transcript/guidance events that occur, and writes a manifest when the session ends. Frames
are stored individually (so a model can sample them, and so they can be encoded into a video);
audio is appended into a single track.
"""

from __future__ import annotations

import re

from app.models.events import DetectionEvent, GuidanceEvent, TranscriptEvent
from app.models.session import RecordedEvent, SessionManifest
from app.storage import ObjectStore

# Minimum seconds between recorded VLM scene snapshots. Detections arrive ~1.4/s; keeping
# every one would flood the manifest and the playback log, so the scene stream is sampled.
# This is the primary control on how dense the scene stream is — raise it for fewer lines.
_SCENE_MIN_INTERVAL = 10.0
# A static view makes the VLM re-describe the same scene with slightly different wording each
# call. Skip a new scene whose word-overlap with the last recorded one is at least this high,
# so the stream only advances when the scene meaningfully changes (new person/object/place).
# Tuned low because the model rewords heavily — same-scene rewordings score ~0.3-0.5, while a
# genuinely different scene scores well under 0.1.
_SCENE_SIMILARITY = 0.45


def _scene_similarity(a: str, b: str) -> float:
    """Jaccard word-overlap of two scene descriptions, in [0, 1]."""
    tokens_a = set(re.findall(r"[a-z0-9]+", a.lower()))
    tokens_b = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def session_prefix(session_id: str) -> str:
    return f"sessions/{session_id}"


class SessionRecorder:
    def __init__(
        self,
        object_store: ObjectStore,
        session_id: str,
        started_at: float,
        *,
        officer_id: str | None = None,
        officer_name: str | None = None,
        location: str | None = None,
    ) -> None:
        self._store = object_store
        self._started_at = started_at
        self._manifest = SessionManifest(
            session_id=session_id,
            started_at=started_at,
            officer_id=officer_id,
            officer_name=officer_name,
            location=location,
        )
        self._audio_buffer = bytearray()
        # Dedup/throttle state for the recorded VLM scene stream.
        self._last_scene: str | None = None
        self._last_scene_offset: float | None = None

    @property
    def session_id(self) -> str:
        return self._manifest.session_id

    @property
    def frame_count(self) -> int:
        return self._manifest.frame_count

    @property
    def has_audio(self) -> bool:
        return self._manifest.has_audio

    @property
    def has_content(self) -> bool:
        """Did the session capture anything worth reporting (frames, audio, or events)?"""
        return bool(
            self._manifest.frame_count or self._manifest.has_audio or self._manifest.events
        )

    def _offset(self, timestamp: float) -> float:
        return max(0.0, timestamp - self._started_at)

    async def add_frame(self, jpeg: bytes, timestamp: float) -> None:
        index = self._manifest.frame_count
        key = f"{session_prefix(self.session_id)}/frames/{index:06d}.jpg"
        await self._store.put(key, jpeg, "image/jpeg")
        self._manifest.frame_keys.append(key)
        self._manifest.frame_offsets.append(self._offset(timestamp))
        self._manifest.frame_count += 1

    def add_audio_chunk(self, chunk: bytes) -> None:
        # Audio chunks are contiguous parts of one WebM stream; buffer then flush on close.
        self._audio_buffer.extend(chunk)
        self._manifest.has_audio = True

    def record_event(
        self, event: TranscriptEvent | GuidanceEvent | DetectionEvent
    ) -> None:
        """Capture an event so playback can replay it in sync.

        Transcript/guidance events are kept verbatim. Detection (VLM scene) events are
        sampled into a "what the camera saw" stream: only a non-empty summary that has
        changed and is at least `_SCENE_MIN_INTERVAL` seconds after the last recorded scene
        is kept, so the stream stays readable and the manifest small.
        """
        offset = self._offset(event.ts)
        if isinstance(event, DetectionEvent):
            summary = (event.summary or "").strip()
            if not summary:
                return
            # Skip near-duplicates of the last recorded scene (same view, reworded).
            if (
                self._last_scene is not None
                and _scene_similarity(summary, self._last_scene) >= _SCENE_SIMILARITY
            ):
                return
            # Even for a changed scene, don't record faster than the sampling floor.
            if (
                self._last_scene_offset is not None
                and (offset - self._last_scene_offset) < _SCENE_MIN_INTERVAL
            ):
                return
            self._last_scene = summary
            self._last_scene_offset = offset
        self._manifest.events.append(
            RecordedEvent(
                offset_seconds=offset,
                kind=event.type.value,
                payload=event.model_dump(mode="json"),
            )
        )

    async def finalize(self, ended_at: float) -> SessionManifest:
        if self._audio_buffer:
            audio_key = f"{session_prefix(self.session_id)}/audio.webm"
            await self._store.put(audio_key, bytes(self._audio_buffer), "audio/webm")
            self._manifest.audio_key = audio_key

        self._manifest.ended_at = ended_at
        # The session-end pipeline (diarize → video summary → report → scene card) runs in the
        # background; flag a session with content as "processing" until that pass marks it ready.
        self._manifest.status = "processing" if self.has_content else "ready"
        await self._write_manifest()
        return self._manifest

    async def _write_manifest(self) -> None:
        manifest_key = f"{session_prefix(self.session_id)}/manifest.json"
        await self._store.put(
            manifest_key,
            self._manifest.model_dump_json(indent=2).encode("utf-8"),
            "application/json",
        )
