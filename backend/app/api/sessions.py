"""Sessions API — browse recorded patrol sessions and play back their media."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.session import SessionManifest, SessionSummary
from app.storage import get_object_store

router = APIRouter(prefix="/sessions", tags=["sessions"])


async def _load_manifest(session_id: str) -> SessionManifest:
    store = get_object_store()
    try:
        body, _ = await store.get(f"sessions/{session_id}/manifest.json")
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Session not found") from error
    return SessionManifest.model_validate(json.loads(body))


@router.get("", response_model=list[SessionSummary])
async def list_sessions() -> list[SessionSummary]:
    """List recorded sessions, newest first."""
    store = get_object_store()
    prefixes = await store.list_prefixes("sessions/")
    summaries: list[SessionSummary] = []
    for prefix in prefixes:
        session_id = prefix.removeprefix("sessions/").rstrip("/")
        try:
            manifest = await _load_manifest(session_id)
        except HTTPException:
            continue  # session still recording or manifest not yet written
        summaries.append(
            SessionSummary(
                session_id=manifest.session_id,
                started_at=manifest.started_at,
                ended_at=manifest.ended_at,
                frame_count=manifest.frame_count,
                has_audio=manifest.has_audio,
            )
        )
    summaries.sort(key=lambda summary: summary.started_at, reverse=True)
    return summaries


@router.get("/{session_id}", response_model=SessionManifest)
async def get_session(session_id: str) -> SessionManifest:
    return await _load_manifest(session_id)


@router.get("/{session_id}/frames/{index}")
async def get_frame(session_id: str, index: int) -> StreamingResponse:
    store = get_object_store()
    key = f"sessions/{session_id}/frames/{index:06d}.jpg"
    try:
        body, content_type = await store.get(key)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Frame not found") from error
    return StreamingResponse(iter([body]), media_type=content_type)


@router.get("/{session_id}/audio")
async def get_audio(session_id: str) -> StreamingResponse:
    manifest = await _load_manifest(session_id)
    if not manifest.audio_key:
        raise HTTPException(status_code=404, detail="Session has no audio")
    store = get_object_store()
    return StreamingResponse(store.stream(manifest.audio_key), media_type="audio/webm")
