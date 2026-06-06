"""Per-session recorder.

Streams captured frames and audio straight to object storage as they arrive, records the
transcript/guidance events that occur, and writes a manifest when the session ends. Frames
are stored individually (so a model can sample them, and so they can be encoded into a video);
audio is appended into a single track.
"""

from __future__ import annotations

from app.models.events import GuidanceEvent, TranscriptEvent
from app.models.session import RecordedEvent, SessionManifest
from app.storage import ObjectStore


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
    ) -> None:
        self._store = object_store
        self._started_at = started_at
        self._manifest = SessionManifest(
            session_id=session_id,
            started_at=started_at,
            officer_id=officer_id,
            officer_name=officer_name,
        )
        self._audio_buffer = bytearray()

    @property
    def session_id(self) -> str:
        return self._manifest.session_id

    @property
    def frame_count(self) -> int:
        return self._manifest.frame_count

    @property
    def has_audio(self) -> bool:
        return self._manifest.has_audio

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

    def record_event(self, event: TranscriptEvent | GuidanceEvent) -> None:
        """Capture a transcript/guidance event so playback can replay it in sync."""
        self._manifest.events.append(
            RecordedEvent(
                offset_seconds=self._offset(event.ts),
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
        await self._write_manifest()
        return self._manifest

    async def _write_manifest(self) -> None:
        manifest_key = f"{session_prefix(self.session_id)}/manifest.json"
        await self._store.put(
            manifest_key,
            self._manifest.model_dump_json(indent=2).encode("utf-8"),
            "application/json",
        )
