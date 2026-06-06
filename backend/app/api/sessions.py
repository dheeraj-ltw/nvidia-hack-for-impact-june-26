"""Sessions API — browse, play back, and manage recorded patrol sessions."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from app.models.session import (
    SessionLabelUpdate,
    SessionManifest,
    SessionSummary,
)
from app.recording import EncodingError, encode_session_video, session_prefix
from app.storage import get_object_store

router = APIRouter(prefix="/sessions", tags=["sessions"])


async def _load_manifest(session_id: str) -> SessionManifest:
    store = get_object_store()
    try:
        body, _ = await store.get(f"{session_prefix(session_id)}/manifest.json")
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Session not found") from error
    return SessionManifest.model_validate(json.loads(body))


async def _save_manifest(manifest: SessionManifest) -> None:
    store = get_object_store()
    await store.put(
        f"{session_prefix(manifest.session_id)}/manifest.json",
        manifest.model_dump_json(indent=2).encode("utf-8"),
        "application/json",
    )


def _to_summary(manifest: SessionManifest) -> SessionSummary:
    return SessionSummary(
        session_id=manifest.session_id,
        label=manifest.label,
        started_at=manifest.started_at,
        ended_at=manifest.ended_at,
        frame_count=manifest.frame_count,
        has_audio=manifest.has_audio,
        has_video=manifest.video_key is not None,
        event_count=len(manifest.events),
    )


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
        summaries.append(_to_summary(manifest))
    summaries.sort(key=lambda summary: summary.started_at, reverse=True)
    return summaries


@router.get("/{session_id}", response_model=SessionManifest)
async def get_session(session_id: str) -> SessionManifest:
    """Full manifest, including the recorded transcript/guidance events for playback."""
    return await _load_manifest(session_id)


@router.patch("/{session_id}", response_model=SessionSummary)
async def rename_session(session_id: str, update: SessionLabelUpdate) -> SessionSummary:
    manifest = await _load_manifest(session_id)
    manifest.label = update.label
    await _save_manifest(manifest)
    return _to_summary(manifest)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str) -> Response:
    await _load_manifest(session_id)  # 404 if it does not exist
    store = get_object_store()
    await store.delete_prefix(f"{session_prefix(session_id)}/")
    return Response(status_code=204)


@router.get("/{session_id}/frames/{index}")
async def get_frame(session_id: str, index: int) -> StreamingResponse:
    store = get_object_store()
    key = f"{session_prefix(session_id)}/frames/{index:06d}.jpg"
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


@router.get("/{session_id}/video")
async def get_video(session_id: str) -> StreamingResponse:
    """Return the encoded MP4, encoding it on first request and caching the result."""
    manifest = await _load_manifest(session_id)
    store = get_object_store()

    if manifest.video_key is None:
        try:
            manifest.video_key = await encode_session_video(store, manifest)
        except EncodingError as error:
            raise HTTPException(status_code=422, detail=f"Encoding failed: {error}") from error
        await _save_manifest(manifest)

    headers = {"Content-Disposition": f'inline; filename="{session_id}.mp4"'}
    return StreamingResponse(
        store.stream(manifest.video_key), media_type="video/mp4", headers=headers
    )
