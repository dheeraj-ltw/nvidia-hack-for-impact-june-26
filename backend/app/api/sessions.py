"""Sessions API — browse, play back, and manage recorded patrol sessions."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from app.ai.speaker_id import SpeakerIdError, identify_officer
from app.ai.vlm import VlmCaptioner
from app.api.officers import load_profile
from app.config import get_settings
from app.models.events import EventType, TranscriptEvent
from app.models.session import (
    IncidentReport,
    RecordedEvent,
    ReportResult,
    SessionLabelUpdate,
    SessionManifest,
    SessionSummary,
)
from app.recording import EncodingError, encode_session_video, session_prefix
from app.reporting import generate_and_dispatch
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
        has_report=manifest.report_key is not None,
        event_count=len(manifest.events),
        officer_name=manifest.officer_name,
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


def _diarized_events(segments: list[dict[str, Any]]) -> list[RecordedEvent]:
    """Turn diarized officer/personN segments into replayable transcript events."""
    events: list[RecordedEvent] = []
    for seg in segments:
        if not seg.get("text"):
            continue
        transcript = TranscriptEvent(
            ts=float(seg["start"]),
            text=seg["text"],
            speaker=seg["speaker_label"],
            is_final=True,
        )
        events.append(
            RecordedEvent(
                offset_seconds=float(seg["start"]),
                kind=EventType.TRANSCRIPT.value,
                payload=transcript.model_dump(mode="json"),
            )
        )
    return events


async def run_session_identification(session_id: str) -> SessionManifest:
    """Re-label a recorded session's transcript by officer vs person1/person2/...

    Loads the recorded audio and the session's enrolled officer, runs diarized speaker-ID,
    then *replaces* the transcript events with the diarized turns (guidance events are kept)
    and records the diarization summary. Returns the updated manifest.

    Raises SpeakerIdError if it can't run (no officer, no audio, or no STT key) — callers
    decide whether that's a hard error (manual endpoint) or a soft skip (background task).
    """
    manifest = await _load_manifest(session_id)
    if not manifest.officer_id:
        raise SpeakerIdError("Session has no enrolled officer to identify.")
    if not manifest.audio_key:
        raise SpeakerIdError("Session has no recorded audio.")

    officer = await load_profile(manifest.officer_id)
    if not officer.embedding:
        raise SpeakerIdError("Enrolled officer has no voice embedding.")

    store = get_object_store()
    audio, _ = await store.get(manifest.audio_key)
    result = await asyncio.to_thread(identify_officer, audio, officer.embedding)

    # Replace transcript events with the diarized segments; preserve guidance events in order.
    guidance_events = [e for e in manifest.events if e.kind != EventType.TRANSCRIPT.value]
    transcript_events = _diarized_events(result.get("segments", []))
    merged = transcript_events + guidance_events
    merged.sort(key=lambda event: event.offset_seconds)
    manifest.events = merged

    manifest.diarization = {
        "matched": result.get("matched", False),
        "officer_speaker_id": result.get("officer_speaker_id"),
        "similarities": result.get("similarities", {}),
        "labels": result.get("labels", {}),
    }
    await _save_manifest(manifest)
    return manifest


@router.post("/{session_id}/identify", response_model=SessionManifest)
async def identify_session_speakers(session_id: str) -> SessionManifest:
    """Run (or re-run) the post-session speaker-ID pass and return the updated manifest."""
    try:
        return await run_session_identification(session_id)
    except SpeakerIdError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


# Evenly sample at most this many recorded frames to caption — bounds VLM cost/latency while
# still spanning the whole session.
_SCENE_SAMPLE_FRAMES = 6


def _sample_indices(total: int, sample: int) -> list[int]:
    """Evenly spaced frame indices spanning [0, total), at most `sample` of them."""
    if total <= 0:
        return []
    if total <= sample:
        return list(range(total))
    step = total / sample
    return [min(total - 1, int(i * step)) for i in range(sample)]


async def run_scene_analysis(session_id: str) -> SessionManifest:
    """Caption the session's recorded frames with the VLM and store a scene summary.

    Runs post-session (never on the live path) so vision can't block transcription. Samples a
    handful of frames across the session, captions each with Nebius Qwen2.5-VL, and joins them
    into a single timestamped scene description on the manifest. A no-op (returns the manifest
    unchanged) when no Nebius key is configured or the session has no frames.
    """
    settings = get_settings()
    manifest = await _load_manifest(session_id)
    if not settings.nebius_api_key or not manifest.frame_keys:
        return manifest

    store = get_object_store()
    captioner = VlmCaptioner(
        base_url=settings.nebius_base_url,
        model=settings.nebius_vlm_model,
        api_key=settings.nebius_api_key,
    )

    indices = _sample_indices(len(manifest.frame_keys), _SCENE_SAMPLE_FRAMES)
    lines: list[str] = []
    for index in indices:
        try:
            jpeg, _ = await store.get(manifest.frame_keys[index])
        except KeyError:
            continue
        caption = await captioner.describe_frame(jpeg)
        if caption:
            offset = manifest.frame_offsets[index] if index < len(manifest.frame_offsets) else 0.0
            lines.append(f"[{offset:.0f}s] {caption}")

    # Re-load before saving so we don't clobber a diarization pass that ran alongside us.
    fresh = await _load_manifest(session_id)
    fresh.scene_summary = "\n".join(lines)
    await _save_manifest(fresh)
    return fresh


@router.post("/{session_id}/scene", response_model=SessionManifest)
async def analyze_session_scene(session_id: str) -> SessionManifest:
    """Run (or re-run) the post-session VLM scene analysis and return the updated manifest."""
    return await run_scene_analysis(session_id)


async def generate_session_report(session_id: str) -> ReportResult:
    """Build the incident report for a session, store it, and dispatch the webhooks.

    Used both by the manual endpoint and by the session-end background task. Records the
    report key back onto the manifest so the session list can show that a report exists.
    """
    manifest = await _load_manifest(session_id)
    store = get_object_store()
    result = await generate_and_dispatch(store, manifest)
    # Re-load before stamping report_key: the post-session identify pass may have written
    # diarization onto the manifest concurrently, and we must not clobber it (last-write-wins).
    fresh = await _load_manifest(session_id)
    fresh.report_key = f"{session_prefix(session_id)}/report.json"
    await _save_manifest(fresh)
    return result


@router.get("/{session_id}/report", response_model=IncidentReport)
async def get_session_report(session_id: str) -> IncidentReport:
    """Return the stored incident report, 404 if one hasn't been generated yet."""
    manifest = await _load_manifest(session_id)
    if not manifest.report_key:
        raise HTTPException(status_code=404, detail="No report generated for this session")
    store = get_object_store()
    try:
        body, _ = await store.get(manifest.report_key)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Report not found") from error
    return IncidentReport.model_validate(json.loads(body))


@router.post("/{session_id}/report", response_model=ReportResult)
async def create_session_report(session_id: str) -> ReportResult:
    """Generate (or regenerate) the incident report and redispatch it to the webhooks."""
    return await generate_session_report(session_id)
