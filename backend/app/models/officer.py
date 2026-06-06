"""Officer roster: a lightweight, no-auth profile so we can identify who is on patrol.

An officer enrolls once with a name and a short voice sample. The sample is embedded into a
reusable speaker d-vector (resemblyzer) and stored on the profile, so future patrols can match
the officer's voice against the conversation without re-embedding the reference each time.

Stored in object storage alongside session media:
  officers/<officer_id>/profile.json     this model, serialized
  officers/<officer_id>/reference.webm   the raw voice sample
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OfficerProfile(BaseModel):
    """Full officer record, including the voice embedding used for speaker matching."""

    officer_id: str
    name: str
    created_at: float  # unix seconds
    reference_audio_key: str | None = None
    embedding: list[float] = Field(default_factory=list)  # resemblyzer d-vector

    @property
    def has_audio(self) -> bool:
        return self.reference_audio_key is not None


class OfficerSummary(BaseModel):
    """Lightweight listing entry — what the officers list endpoint returns (no embedding)."""

    officer_id: str
    name: str
    created_at: float
    has_audio: bool


class OfficerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
