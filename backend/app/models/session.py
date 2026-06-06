"""Recorded patrol session metadata.

A session manifest is stored alongside its media in object storage so a model — or an
operator reviewing later — can consume the stored frames, audio, encoded video, and the
transcript/guidance events that occurred during the session.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecordedEvent(BaseModel):
    """A transcript or guidance event captured during the session, for synced playback."""

    offset_seconds: float  # seconds from session start, so playback can align it
    kind: str  # "transcript" | "guidance"
    payload: dict[str, Any]  # the original event, serialized


class SessionManifest(BaseModel):
    session_id: str
    label: str | None = None
    started_at: float  # client capture clock (seconds)
    ended_at: float | None = None
    frame_count: int = 0
    has_audio: bool = False
    frame_keys: list[str] = Field(default_factory=list)
    frame_offsets: list[float] = Field(default_factory=list)  # per-frame seconds from start
    audio_key: str | None = None
    video_key: str | None = None  # encoded MP4, produced on demand
    events: list[RecordedEvent] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if self.ended_at is None:
            return 0.0
        return max(0.0, self.ended_at - self.started_at)


class SessionSummary(BaseModel):
    """Lightweight listing entry — what the sessions list endpoint returns."""

    session_id: str
    label: str | None
    started_at: float
    ended_at: float | None
    frame_count: int
    has_audio: bool
    has_video: bool
    event_count: int


class SessionLabelUpdate(BaseModel):
    label: str = Field(min_length=1, max_length=120)
