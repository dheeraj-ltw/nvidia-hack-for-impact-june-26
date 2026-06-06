"""WebSocket patrol session endpoint.

Wire protocol (client -> server), all binary messages:
  byte 0       : message_kind (0x00 = video JPEG, 0x01 = audio chunk)
  bytes 1..8   : float64 big-endian capture timestamp (seconds)
  bytes 9..12  : uint16 width, uint16 height (video only; zero for audio)
  bytes 13..   : payload

Each frame is recorded to object storage and fed to the AI pipeline. On disconnect the
session manifest is finalized so the stored media can be analyzed later.

Server -> client: JSON-encoded PatrolEvent objects (see app.models.events).
"""

from __future__ import annotations

import struct
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ai import get_ai_service
from app.ai.base import Frame
from app.models.events import GuidanceEvent, PatrolEvent, StatusEvent, TranscriptEvent
from app.pipeline.orchestrator import SessionPipeline
from app.recording import SessionRecorder
from app.storage import get_object_store

router = APIRouter()

MESSAGE_KIND_VIDEO = 0x00
MESSAGE_KIND_AUDIO = 0x01
_HEADER = struct.Struct(">d H H")  # timestamp, width, height (follows the 1-byte kind)


@router.websocket("/ws/patrol")
async def patrol_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    ai_service = get_ai_service()

    session_id = uuid.uuid4().hex
    recorder: SessionRecorder | None = None
    last_timestamp = 0.0

    async def emit(event: PatrolEvent) -> None:
        # Persist transcript/guidance so the recorded session can replay its logs.
        if recorder is not None and isinstance(event, TranscriptEvent | GuidanceEvent):
            recorder.record_event(event)
        await websocket.send_text(event.model_dump_json())

    pipeline = SessionPipeline(ai_service=ai_service, emit=emit)

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
                recorder = SessionRecorder(get_object_store(), session_id, timestamp)
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
                recorder.add_audio_chunk(payload)
                await pipeline.handle_audio(payload, timestamp)
    except WebSocketDisconnect:
        pass
    finally:
        if recorder is not None:
            await recorder.finalize(last_timestamp)
