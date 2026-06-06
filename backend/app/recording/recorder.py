"""Per-session recorder.

Streams captured frames and audio straight to object storage as they arrive, and writes
a manifest when the session ends. Frames are stored individually (so a model can sample
them); audio is appended into a single track.
"""

from __future__ import annotations

from app.models.session import SessionManifest
from app.storage import ObjectStore


def _session_prefix(session_id: str) -> str:
    return f"sessions/{session_id}"


class SessionRecorder:
    def __init__(self, object_store: ObjectStore, session_id: str, started_at: float) -> None:
        self._store = object_store
        self._manifest = SessionManifest(session_id=session_id, started_at=started_at)
        self._audio_buffer = bytearray()

    @property
    def session_id(self) -> str:
        return self._manifest.session_id

    @property
    def frame_count(self) -> int:
        return self._manifest.frame_count

    async def add_frame(self, jpeg: bytes) -> None:
        index = self._manifest.frame_count
        key = f"{_session_prefix(self.session_id)}/frames/{index:06d}.jpg"
        await self._store.put(key, jpeg, "image/jpeg")
        self._manifest.frame_keys.append(key)
        self._manifest.frame_count += 1

    def add_audio_chunk(self, chunk: bytes) -> None:
        # Audio chunks are contiguous parts of one WebM stream; buffer then flush on close.
        self._audio_buffer.extend(chunk)
        self._manifest.has_audio = True

    async def finalize(self, ended_at: float) -> SessionManifest:
        if self._audio_buffer:
            audio_key = f"{_session_prefix(self.session_id)}/audio.webm"
            await self._store.put(audio_key, bytes(self._audio_buffer), "audio/webm")
            self._manifest.audio_key = audio_key

        self._manifest.ended_at = ended_at
        manifest_key = f"{_session_prefix(self.session_id)}/manifest.json"
        await self._store.put(
            manifest_key,
            self._manifest.model_dump_json(indent=2).encode("utf-8"),
            "application/json",
        )
        return self._manifest
