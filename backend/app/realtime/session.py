"""WebSocket patrol session endpoint.

Wire protocol (client -> server), all binary messages:
  byte 0       : message_kind (0x00 = video JPEG, 0x01 = audio chunk, 0x02 = audio clip)
  bytes 1..8   : float64 big-endian capture timestamp (seconds)
  bytes 9..12  : uint16 width, uint16 height (video only; zero for audio)
  bytes 13..   : payload

Audio arrives on two paths: continuous *chunks* (0x01) are fragments of one WebM stream,
concatenated into the recorded track; self-contained *clips* (0x02) are complete WebM files
sent to speech-to-text. Each frame/clip is recorded and fed to the AI pipeline; on disconnect
the session manifest is finalized so the stored media can be analyzed later.

Server -> client: JSON-encoded PatrolEvent objects (see app.models.events).
"""

from __future__ import annotations

import asyncio
import logging
import struct
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ai import get_ai_service
from app.ai.base import Frame
from app.ai.speaker_id import warm_up
from app.api.officers import load_profile
from app.models.events import GuidanceEvent, PatrolEvent, StatusEvent, TranscriptEvent
from app.models.officer import OfficerProfile
from app.pipeline.orchestrator import SessionPipeline
from app.recording import SessionRecorder
from app.storage import get_object_store

logger = logging.getLogger(__name__)

router = APIRouter()

# Strong references to in-flight background tasks. asyncio only holds a weak reference to
# tasks, so without this the post-session pass could be garbage-collected before it finishes.
_background_tasks: set[asyncio.Task[None]] = set()


async def _load_officer(officer_id: str | None) -> OfficerProfile | None:
    """Best-effort officer lookup — a bad/missing id must not break the patrol."""
    if not officer_id:
        return None
    try:
        return await load_profile(officer_id)
    except Exception as error:  # noqa: BLE001 - HTTPException(404) or storage hiccup
        logger.warning("Could not load officer %s: %s", officer_id, error)
        return None

MESSAGE_KIND_VIDEO = 0x00
MESSAGE_KIND_AUDIO = 0x01  # continuous WebM fragment, for the recorded track
MESSAGE_KIND_AUDIO_CLIP = 0x02  # complete, self-contained WebM file, for speech-to-text
_HEADER = struct.Struct(">d H H")  # timestamp, width, height (follows the 1-byte kind)


@router.websocket("/ws/patrol")
async def patrol_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    ai_service = get_ai_service()

    # Who is on patrol? (no-auth roster — passed as ?officer_id=... on the WS URL)
    officer = await _load_officer(websocket.query_params.get("officer_id"))
    officer_embedding = officer.embedding if officer and officer.embedding else None
    if officer_embedding:
        # Preload the voice encoder off the event loop so the first live match isn't slow.
        await asyncio.to_thread(warm_up)

    session_id = uuid.uuid4().hex
    recorder: SessionRecorder | None = None
    last_timestamp = 0.0
    logger.info(
        "WS patrol connected: session=%s officer=%s backend=%s",
        session_id,
        officer.name if officer else "(none)",
        type(ai_service).__name__,
    )

    async def emit(event: PatrolEvent) -> None:
        # Persist transcript/guidance so the recorded session can replay its logs.
        if recorder is not None and isinstance(event, TranscriptEvent | GuidanceEvent):
            recorder.record_event(event)
        await websocket.send_text(event.model_dump_json())

    pipeline = SessionPipeline(
        ai_service=ai_service, emit=emit, officer_embedding=officer_embedding
    )

    try:
        while True:
            message = await websocket.receive_bytes()
            if len(message) < 1 + _HEADER.size:
                continue
            message_kind = message[0]
            timestamp, width, height = _HEADER.unpack_from(message, 1)
            payload = message[1 + _HEADER.size :]
            last_timestamp = timestamp

            # Start the recording on the first message, once we have a capture clock.
            if recorder is None:
                recorder = SessionRecorder(
                    get_object_store(),
                    session_id,
                    timestamp,
                    officer_id=officer.officer_id if officer else None,
                    officer_name=officer.name if officer else None,
                )
                await emit(
                    StatusEvent(
                        ts=timestamp,
                        state="recording",
                        detail=f"ai_backend:{type(ai_service).__name__}",
                        session_id=session_id,
                    )
                )

            if message_kind == MESSAGE_KIND_VIDEO:
                await recorder.add_frame(payload, timestamp)
                await pipeline.handle_frame(
                    Frame(ts=timestamp, jpeg=payload, width=width, height=height)
                )
            elif message_kind == MESSAGE_KIND_AUDIO:
                # Continuous fragment — store it, but it is not independently decodable.
                recorder.add_audio_chunk(payload)
            elif message_kind == MESSAGE_KIND_AUDIO_CLIP:
                # Complete WebM file — transcribe it.
                await pipeline.handle_audio(payload, timestamp)
    except WebSocketDisconnect:
        pass
    finally:
        if recorder is not None:
            await recorder.finalize(last_timestamp)
            logger.info(
                "WS patrol ended: session=%s frames=%d audio=%s",
                session_id,
                recorder.frame_count,
                recorder.has_audio,
            )
            # Refine the transcript (if we can) and then build + dispatch the incident
            # report, in the background so the disconnect isn't blocked on either step.
            # Skip empty sessions entirely — no point reporting a patrol with no content,
            # and it avoids firing webhooks (incl. the cop registry) on stray connections.
            if recorder.has_content:
                run_identify = bool(officer_embedding and recorder.has_audio)
                task = asyncio.create_task(_finalize_session(session_id, run_identify))
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
            else:
                logger.info("Session %s had no content — skipping report.", session_id)


async def _finalize_session(session_id: str, run_identify: bool) -> None:
    """Post-session pipeline: optional speaker-ID, then the incident report + webhooks.

    Identification runs first when possible so the report captures the diarized
    officer/person1/person2 transcript rather than the live officer/subject labels.
    Each step is isolated — a failure is logged and never raised from the background task.
    """
    from app.api.sessions import generate_session_report, run_session_identification

    if run_identify:
        try:
            await run_session_identification(session_id)
        except Exception as error:  # noqa: BLE001 - background task; nothing to surface to
            logger.warning("Post-session identification failed for %s: %s", session_id, error)

    try:
        await generate_session_report(session_id)
    except Exception as error:  # noqa: BLE001 - background task; nothing to surface to
        logger.warning("Incident report generation failed for %s: %s", session_id, error)
