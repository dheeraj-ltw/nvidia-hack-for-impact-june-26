"""Recorded patrol session metadata.

A session manifest is stored alongside its media in object storage so a model can later
consume the stored frames + audio without any live connection.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionManifest(BaseModel):
    session_id: str
    started_at: float  # client capture clock (seconds)
    ended_at: float | None = None
    frame_count: int = 0
    has_audio: bool = False
    frame_keys: list[str] = Field(default_factory=list)
    audio_key: str | None = None


class SessionSummary(BaseModel):
    """Lightweight listing entry — what the sessions list endpoint returns."""

    session_id: str
    started_at: float
    ended_at: float | None
    frame_count: int
    has_audio: bool
