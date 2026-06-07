"""Sessions API — browse, play back, and manage recorded patrol sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import tempfile
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.ai import policeai
from app.ai.scene_compiler import SceneCardComposer
from app.ai.speaker_id import (
    SpeakerIdError,
    _merge_segments,
    identify_officer,
    transcribe_with_speakers,
)
from app.ai.vlm import VlmCaptioner
from app.api.officers import load_profile
from app.config import get_settings
from app.models.events import DetectionEvent, EventType, TranscriptEvent
from app.models.session import (
    IncidentReport,
    RecordedEvent,
    ReportResult,
    SessionLabelUpdate,
    SessionManifest,
    SessionSummary,
)
from app.recording import (
    EncodingError,
    SessionRecorder,
    encode_session_video,
    session_prefix,
)
from app.reporting import generate_and_dispatch
from app.storage import get_object_store

logger = logging.getLogger(__name__)

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
        status=manifest.status,
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


async def mark_session_ready(session_id: str) -> None:
    """Flip a session's status to "ready" once the session-end pipeline has finished."""
    manifest = await _load_manifest(session_id)
    if manifest.status != "ready":
        manifest.status = "ready"
        await _save_manifest(manifest)


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


async def transcribe_session_audio(session_id: str) -> None:
    """Diarized speech-to-text for a session that has audio but no transcript yet (uploads).

    Officer-aware: if an officer is enrolled, their turns are labelled `officer` and the rest
    `person1`/`person2`…; otherwise speakers are labelled `person1`/`person2`… by first
    appearance. No-op when there's no audio, or transcript events already exist (a live session
    already produced them live / via identification). Best-effort — failures are logged.
    """
    manifest = await _load_manifest(session_id)
    if not manifest.audio_key:
        return
    if any(event.kind == EventType.TRANSCRIPT.value for event in manifest.events):
        return

    store = get_object_store()
    audio, _ = await store.get(manifest.audio_key)

    embedding: list[float] | None = None
    if manifest.officer_id:
        try:
            officer = await load_profile(manifest.officer_id)
            embedding = officer.embedding
        except HTTPException:
            embedding = None

    try:
        if embedding:
            result = await asyncio.to_thread(identify_officer, audio, embedding)
            segments = result.get("segments", [])
            diarization: dict[str, Any] = {
                "matched": result.get("matched", False),
                "officer_speaker_id": result.get("officer_speaker_id"),
                "similarities": result.get("similarities", {}),
                "labels": result.get("labels", {}),
            }
        else:
            diar = await asyncio.to_thread(transcribe_with_speakers, audio)
            labels = {spk: f"person{i + 1}" for i, spk in enumerate(diar["speakers"])}
            segments = _merge_segments(diar["words"], labels)
            diarization = {"labels": labels}
    except SpeakerIdError as error:
        logger.warning("Transcription failed for %s: %s", session_id, error)
        return

    transcript_events = _diarized_events(segments)
    if not transcript_events:
        return

    fresh = await _load_manifest(session_id)
    others = [e for e in fresh.events if e.kind != EventType.TRANSCRIPT.value]
    fresh.events = transcript_events + others
    fresh.events.sort(key=lambda event: event.offset_seconds)
    fresh.diarization = diarization
    await _save_manifest(fresh)


@router.post("/{session_id}/identify", response_model=SessionManifest)
async def identify_session_speakers(session_id: str) -> SessionManifest:
    """Run (or re-run) the post-session speaker-ID pass and return the updated manifest."""
    try:
        return await run_session_identification(session_id)
    except SpeakerIdError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


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


# Bound the reasoner prompt: keep the most recent dialogue and the recent scene stream.
_TRANSCRIPT_MAX_CHARS = 6000
_SCENE_MAX_CHARS = 2000


def _aggregate_session_context(manifest: SessionManifest) -> tuple[str, str]:
    """Collect the recorded transcript and VLM scene streams from a manifest.

    One ordered pass (mirrors report._extract_transcript_and_guidance): transcript lines from
    transcript events, scene descriptions from detection events. Each is truncated to its tail
    (most recent) to bound the reasoner prompt.
    """
    transcript_lines: list[str] = []
    scene_lines: list[str] = []
    for event in sorted(manifest.events, key=lambda e: e.offset_seconds):
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.kind == EventType.TRANSCRIPT.value:
            text = str(payload.get("text", "")).strip()
            if text:
                speaker = str(payload.get("speaker", "unknown"))
                transcript_lines.append(f"[{speaker}] {text}")
        elif event.kind == EventType.DETECTION.value:
            summary = str(payload.get("summary", "")).strip()
            if summary:
                scene_lines.append(summary)
    transcript = "\n".join(transcript_lines)[-_TRANSCRIPT_MAX_CHARS:]
    scene_text = "\n".join(scene_lines)[-_SCENE_MAX_CHARS:]
    return transcript, scene_text


_UPLOAD_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".m4v")


async def _extract_frames_with_ffmpeg(
    video_bytes: bytes, fps: float
) -> list[tuple[bytes, float]]:
    """Extract JPEG frames from an uploaded video at `fps` frames/second (via ffmpeg).

    Returns [(jpeg, offset_seconds), ...] in chronological order. Raises HTTPException(422) if
    ffmpeg cannot read the file. Mirrors the subprocess pattern in recording/encoder.py.
    """
    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)
        source = work / "upload"
        source.write_bytes(video_bytes)
        frames_dir = work / "frames"
        frames_dir.mkdir()

        command = [
            "ffmpeg", "-y", "-i", str(source),
            "-vf", f"fps={fps}",
            "-q:v", "3",
            str(frames_dir / "frame_%06d.jpg"),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            detail = stderr.decode(errors="replace")[-300:]
            raise HTTPException(status_code=422, detail=f"Could not read the video: {detail}")

        frames: list[tuple[bytes, float]] = []
        for path in sorted(frames_dir.glob("frame_*.jpg")):
            index = int(path.stem.split("_")[1])  # ffmpeg numbers frames from 1
            frames.append((path.read_bytes(), (index - 1) / fps))
        return frames


async def _extract_audio_with_ffmpeg(video_bytes: bytes) -> bytes:
    """Extract a video's audio track as mono WebM/Opus, or b"" if there is no audio.

    Matches the recorded-audio convention (audio.webm) so STT, diarization and the /audio
    endpoint all treat it identically. Best-effort — returns b"" on any failure / no track.
    """
    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)
        source = work / "upload"
        source.write_bytes(video_bytes)
        out = work / "audio.webm"
        command = [
            "ffmpeg", "-y", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "48000", "-c:a", "libopus",
            str(out),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        if process.returncode != 0 or not out.exists():
            return b""  # no audio track, or the encoder is unavailable
        return out.read_bytes()


@router.post("/upload", response_model=SessionSummary)
async def upload_session_video(
    file: Annotated[UploadFile, File()],
    officer_id: Annotated[str | None, Form()] = None,
    fps: Annotated[float, Form()] = 2.0,
) -> SessionSummary:
    """Upload a recorded video → extract frames → run the same post-session pipeline as a live
    patrol (video summary → scene card). The original file is kept as the playable session
    video. Returns the new session (status 'processing')."""
    name = (file.filename or "").lower()
    if not name.endswith(_UPLOAD_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Upload an .mp4, .mov, .mkv or .webm file.")
    fps = min(max(fps, 0.5), 5.0)

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    frames = await _extract_frames_with_ffmpeg(video_bytes, fps)
    if not frames:
        raise HTTPException(status_code=422, detail="No frames could be extracted from the video.")
    audio_bytes = await _extract_audio_with_ffmpeg(video_bytes)

    officer = None
    if officer_id:
        try:
            officer = await load_profile(officer_id)
        except HTTPException:
            officer = None  # bad/missing id must not block the upload

    session_id = uuid.uuid4().hex
    started_at = time.time()
    store = get_object_store()
    recorder = SessionRecorder(
        store,
        session_id,
        started_at,
        officer_id=officer.officer_id if officer else None,
        officer_name=officer.name if officer else None,
    )
    for jpeg, offset in frames:
        await recorder.add_frame(jpeg, started_at + offset)
    if audio_bytes:
        recorder.add_audio_chunk(audio_bytes)  # finalize writes it as audio.webm + has_audio
    await recorder.finalize(started_at + frames[-1][1])

    # Serve the uploaded file itself as the session video (real playback + audio), bypassing
    # the frame-concat encoder.
    video_key = f"{session_prefix(session_id)}/session.mp4"
    await store.put(video_key, video_bytes, "video/mp4")
    fresh = await _load_manifest(session_id)
    fresh.video_key = video_key
    await _save_manifest(fresh)

    # Same post-session pipeline as a live session, in the background (no audio → no diarization).
    from app.realtime.session import _background_tasks, _finalize_session

    task = asyncio.create_task(_finalize_session(session_id, run_identify=False))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return _to_summary(fresh)


_VIDEO_SUMMARY_FRAMES = 12  # frames sampled across the session for the post-session summary


async def summarize_session_video(session_id: str) -> str:
    """Post-session video understanding: sample frames across the recording, caption the clip
    with the VLM in one call, and record the scene description on the manifest (as a detection
    event) so the scene card and playback log can use it.

    Returns the summary, or "" if there are no frames / the VLM is unavailable. Best-effort —
    replaces any prior detection events so re-running is idempotent.
    """
    manifest = await _load_manifest(session_id)
    keys = manifest.frame_keys
    if not keys:
        return ""

    # Evenly sample up to N frames across the session.
    count = min(_VIDEO_SUMMARY_FRAMES, len(keys))
    if count <= 1:
        indices = [0]
    else:
        indices = sorted({round(i * (len(keys) - 1) / (count - 1)) for i in range(count)})
    store = get_object_store()
    frames: list[bytes] = []
    for index in indices:
        try:
            body, _ = await store.get(keys[index])
            frames.append(body)
        except KeyError:
            continue
    if not frames:
        return ""

    settings = get_settings()
    vlm = VlmCaptioner(
        base_url=settings.nebius_base_url,
        model=settings.nebius_vlm_model,
        api_key=settings.nebius_api_key,
    )
    summary = await vlm.describe_video(frames)
    if not summary:
        return ""

    # Record the whole-session scene as a single detection event at the start, replacing any
    # earlier scene events (none in the post-session-only flow, but keep it idempotent).
    fresh = await _load_manifest(session_id)
    fresh.events = [e for e in fresh.events if e.kind != EventType.DETECTION.value]
    event = DetectionEvent(ts=0.0, boxes=[], summary=summary)
    fresh.events.append(
        RecordedEvent(
            offset_seconds=0.0,
            kind=EventType.DETECTION.value,
            payload=event.model_dump(mode="json"),
        )
    )
    fresh.events.sort(key=lambda event: event.offset_seconds)
    await _save_manifest(fresh)
    return summary


async def _compose_scene_card(manifest: SessionManifest) -> str:
    """Compile the SCENE CARD for a session from its two recorded streams.

    The Nebius-hosted model composes a factual card from the transcript + VLM scene (no legal
    reasoning). If the model is unavailable (no key / error / empty), fall back to the
    deterministic template so a card is always produced.
    """
    transcript, scene_text = _aggregate_session_context(manifest)
    updated = time.strftime("%H:%M:%S", time.localtime(manifest.ended_at or manifest.started_at))
    duration = policeai._format_duration(manifest.duration_seconds)

    settings = get_settings()
    composer = SceneCardComposer(
        base_url=settings.nebius_scene_base_url,
        model=settings.nebius_scene_model,
        api_key=settings.nebius_api_key,
    )
    card = await composer.compose(
        transcript=transcript,
        scene_text=scene_text,
        officer_name=manifest.officer_name,
        duration=duration,
        updated_clock=updated,
    )
    if card:
        return card
    # Deterministic fallback (template + light derivation) when the model can't be reached.
    logger.info("Scene compiler unavailable for %s — deterministic card.", manifest.session_id)
    return policeai.build_session_scene_card(
        updated_clock=updated,
        officer_name=manifest.officer_name,
        duration_seconds=manifest.duration_seconds,
        scene_text=scene_text,
        transcript=transcript,
    )


async def _scene_card_record(manifest: SessionManifest) -> dict[str, Any]:
    """The session's SCENE CARD wrapped in the chat request format (system + user message).

    Same envelope as data/output/sample.jsonl but without the assistant turn — the user
    message is the model-composed SCENE CARD; the PoliceAI system prompt is attached for the
    (separate, not-yet-connected) downstream legal model that would reason over the card.
    """
    card = await _compose_scene_card(manifest)
    return {
        "messages": [
            {"role": "system", "content": policeai.SYSTEM_PROMPT},
            {"role": "user", "content": card},
        ]
    }


def _scene_card_dir() -> pathlib.Path:
    """Local dataset dir for the mirrored SCENE CARD files (default: repo data/output)."""
    configured = get_settings().scene_card_dir
    if configured:
        return pathlib.Path(configured)
    return pathlib.Path(__file__).resolve().parents[3] / "data" / "output"


async def write_session_scene_card(session_id: str) -> dict[str, Any]:
    """Compose the session's SCENE CARD (request-format JSONL) and persist it.

    Canonical copy → object storage (sessions/{id}/scene_card.jsonl); a convenience copy is
    mirrored to the local dataset dir (scene_card_<short>.jsonl) for collection. Records the
    object key on the manifest and returns the record. Called from the session-end task and
    the regenerate endpoint.
    """
    manifest = await _load_manifest(session_id)
    record = await _scene_card_record(manifest)
    line = json.dumps(record, ensure_ascii=False) + "\n"

    key = f"{session_prefix(session_id)}/scene_card.jsonl"
    await get_object_store().put(key, line.encode("utf-8"), "application/jsonl")

    # Best-effort local mirror for dataset collection; never fail the session on this.
    try:
        directory = _scene_card_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"scene_card_{session_id[:12]}.jsonl").write_text(line, encoding="utf-8")
    except OSError as error:  # noqa: BLE001 - mirror is a convenience, not the source of truth
        logger.warning("Could not mirror scene card locally for %s: %s", session_id, error)

    # Re-load before stamping so a concurrent identify/report write isn't clobbered.
    fresh = await _load_manifest(session_id)
    fresh.scene_card_key = key
    await _save_manifest(fresh)
    return record


@router.post("/{session_id}/scenecard")
async def create_session_scene_card(session_id: str) -> dict[str, Any]:
    """(Re)compile the session's SCENE CARD from its transcript + scene and persist it."""
    return await write_session_scene_card(session_id)


@router.get("/{session_id}/scenecard")
async def get_session_scene_card(session_id: str) -> Response:
    """Return the session's stored SCENE CARD as a request-format JSONL line."""
    manifest = await _load_manifest(session_id)
    key = manifest.scene_card_key or f"{session_prefix(session_id)}/scene_card.jsonl"
    store = get_object_store()
    try:
        body, _ = await store.get(key)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail="No scene card for this session"
        ) from error
    return Response(content=body, media_type="application/jsonl")
