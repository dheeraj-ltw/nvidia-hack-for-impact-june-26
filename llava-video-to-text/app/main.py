# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FastAPI app: stream video in (RTSP or upload), get text out via LLaVA.

Endpoints:
  GET    /health                       readiness (ok only once the local vLLM is ready)
  GET    /v1/models                    passthrough to vLLM (reports the served LLaVA model)
  PUT    /v1/video/{name}              upload mp4/mkv -> JSON captions+summary (or SSE if ?stream=true)
  POST   /v1/streams/add               register an RTSP stream for continuous captioning
  GET    /v1/streams/{id}/captions     SSE stream of captions for an active RTSP stream
  DELETE /v1/streams/delete/{id}       stop an RTSP stream
  POST   /v1/chat/completions          passthrough to vLLM (plain OpenAI-compatible LLaVA endpoint)
"""

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import Field
from sse_starlette.sse import EventSourceResponse

from .pipeline import caption_video_file
from .pipeline import iter_caption_video_file
from .streams import StreamManager
from .vlm_client import VlmClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"video/mp4", "video/x-matroska"}
_SUFFIX_BY_TYPE = {"video/mp4": ".mp4", "video/x-matroska": ".mkv"}

vlm = VlmClient()
streams = StreamManager()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    streams.attach_loop(asyncio.get_running_loop())
    yield
    streams.shutdown()
    await vlm.aclose()


app = FastAPI(title="LLaVA Video-to-Text", version="0.1.0", lifespan=lifespan)


def _safe_unlink(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


class AddStreamRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP URL of the live stream to caption")
    name: str = Field("", description="Friendly name for the stream")
    fps: float | None = Field(None, description="Frames per second to sample")
    chunk_seconds: float | None = Field(None, description="Seconds of video per caption window")
    prompt: str | None = Field(None, description="Override the captioning prompt")


@app.get("/health")
async def health() -> JSONResponse | dict[str, object]:
    ready = await vlm.health()
    if not ready:
        return JSONResponse({"status": "starting", "vlm_ready": False}, status_code=503)
    return {"status": "ok", "vlm_ready": True}


@app.get("/v1/models")
async def models() -> dict[str, object]:
    return await vlm.list_models()


@app.put("/v1/video/{name}")
async def upload_video(name: str, request: Request, stream: bool = False) -> object:
    content_type = request.headers.get("content-type", "").split(";")[0].strip()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported Content-Type '{content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    fd, path = tempfile.mkstemp(suffix=_SUFFIX_BY_TYPE[content_type])
    try:
        with os.fdopen(fd, "wb") as out:
            async for chunk in request.stream():
                out.write(chunk)
    except BaseException:
        _safe_unlink(path)
        raise

    if stream:

        async def event_gen() -> AsyncGenerator[dict[str, str], None]:
            try:
                async for caption in iter_caption_video_file(vlm, path):
                    yield {"event": "caption", "data": json.dumps(caption)}
                yield {"event": "end", "data": "{}"}
            finally:
                _safe_unlink(path)

        return EventSourceResponse(event_gen())

    try:
        return await caption_video_file(vlm, path)
    finally:
        _safe_unlink(path)


@app.post("/v1/streams/add")
async def add_stream(req: AddStreamRequest) -> dict[str, str]:
    stream_id = streams.add(req.rtsp_url, req.name, req.fps, req.chunk_seconds, req.prompt)
    return {"stream_id": stream_id, "name": req.name or stream_id}


@app.get("/v1/streams/{stream_id}/captions")
async def stream_captions(stream_id: str) -> EventSourceResponse:
    stream = streams.get(stream_id)
    if stream is None:
        raise HTTPException(status_code=404, detail="Unknown stream id")

    async def event_gen() -> AsyncGenerator[dict[str, str], None]:
        while True:
            item = await stream.queue.get()
            item_type = item.get("type")
            if item_type == "end":
                yield {"event": "end", "data": "{}"}
                break
            if item_type == "error":
                yield {"event": "error", "data": json.dumps(item)}
                continue
            yield {"event": "caption", "data": json.dumps(item)}

    return EventSourceResponse(event_gen())


@app.delete("/v1/streams/delete/{stream_id}")
async def delete_stream(stream_id: str) -> dict[str, str]:
    if not streams.delete(stream_id):
        raise HTTPException(status_code=404, detail="Unknown stream id")
    return {"deleted": stream_id}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, object]:
    payload = await request.json()
    return await vlm.chat(payload)
