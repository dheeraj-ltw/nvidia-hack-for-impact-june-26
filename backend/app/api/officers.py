"""Officers API — enroll and manage the (no-auth) roster used for speaker identification.

Onboarding is deliberately lightweight: an officer provides a name and a short voice sample.
The sample is embedded once into a reusable speaker d-vector so patrols can later tell the
officer's voice apart from everyone else in the conversation.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.ai.speaker_id import SpeakerIdError, embed_reference
from app.models.officer import OfficerProfile, OfficerSummary
from app.storage import get_object_store

router = APIRouter(prefix="/officers", tags=["officers"])

# A voice sample is a short clip (~6-8 s). Cap the upload so a large/crafted body can't
# exhaust memory — the whole clip is read into RAM to embed it.
_MIN_SAMPLE_BYTES = 1024
_MAX_SAMPLE_BYTES = 25 * 1024 * 1024  # 25 MB


def officer_prefix(officer_id: str) -> str:
    return f"officers/{officer_id}"


async def load_profile(officer_id: str) -> OfficerProfile:
    store = get_object_store()
    try:
        body, _ = await store.get(f"{officer_prefix(officer_id)}/profile.json")
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Officer not found") from error
    return OfficerProfile.model_validate(json.loads(body))


def _to_summary(profile: OfficerProfile) -> OfficerSummary:
    return OfficerSummary(
        officer_id=profile.officer_id,
        name=profile.name,
        created_at=profile.created_at,
        has_audio=profile.has_audio,
    )


@router.post("", response_model=OfficerSummary, status_code=201)
async def enroll_officer(
    name: Annotated[str, Form(min_length=1, max_length=120)],
    audio: Annotated[UploadFile, File()],
) -> OfficerSummary:
    """Enroll an officer: store the voice sample and its embedding under a new officer id."""
    clip = await audio.read()
    if len(clip) < _MIN_SAMPLE_BYTES:
        raise HTTPException(status_code=422, detail="Voice sample is too short or empty.")
    if len(clip) > _MAX_SAMPLE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Voice sample too large (max {_MAX_SAMPLE_BYTES // (1024 * 1024)} MB).",
        )

    # Embedding is CPU-bound (and loads the model on first call) — keep the loop responsive.
    try:
        embedding = await asyncio.to_thread(embed_reference, clip)
    except SpeakerIdError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    officer_id = uuid.uuid4().hex
    store = get_object_store()
    reference_key = f"{officer_prefix(officer_id)}/reference.webm"
    await store.put(reference_key, clip, audio.content_type or "audio/webm")

    profile = OfficerProfile(
        officer_id=officer_id,
        name=name.strip(),
        created_at=time.time(),
        reference_audio_key=reference_key,
        embedding=embedding,
    )
    await store.put(
        f"{officer_prefix(officer_id)}/profile.json",
        profile.model_dump_json(indent=2).encode("utf-8"),
        "application/json",
    )
    return _to_summary(profile)


@router.get("", response_model=list[OfficerSummary])
async def list_officers() -> list[OfficerSummary]:
    """List enrolled officers, newest first."""
    store = get_object_store()
    prefixes = await store.list_prefixes("officers/")
    summaries: list[OfficerSummary] = []
    for prefix in prefixes:
        officer_id = prefix.removeprefix("officers/").rstrip("/")
        try:
            summaries.append(_to_summary(await load_profile(officer_id)))
        except HTTPException:
            continue  # incomplete/partial profile — skip
    summaries.sort(key=lambda summary: summary.created_at, reverse=True)
    return summaries


@router.get("/{officer_id}/audio")
async def get_officer_audio(officer_id: str) -> StreamingResponse:
    profile = await load_profile(officer_id)
    if not profile.reference_audio_key:
        raise HTTPException(status_code=404, detail="Officer has no voice sample")
    store = get_object_store()
    return StreamingResponse(store.stream(profile.reference_audio_key), media_type="audio/webm")


@router.delete("/{officer_id}", status_code=204)
async def delete_officer(officer_id: str) -> Response:
    await load_profile(officer_id)  # 404 if missing
    store = get_object_store()
    await store.delete_prefix(f"{officer_prefix(officer_id)}/")
    return Response(status_code=204)
