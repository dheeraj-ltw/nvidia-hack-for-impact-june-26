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
"""RTSP live-stream captioning.

Each stream runs in a dedicated worker thread that pulls frames with OpenCV, samples them at the
requested FPS, and captions one ``chunk_seconds`` window at a time. Captions are pushed onto a
per-stream asyncio queue (thread-safe via ``loop.call_soon_threadsafe``) consumed by the SSE
endpoint in main.py.
"""

import asyncio
import logging
import threading
import time
import uuid
from typing import Any

import cv2

from .config import settings
from .frames import encode_frame
from .frames import subsample
from .vlm_client import build_caption_payload
from .vlm_client import caption_sync

logger = logging.getLogger(__name__)


class _Stream:
    def __init__(
        self,
        stream_id: str,
        rtsp_url: str,
        name: str,
        fps: float,
        chunk_seconds: float,
        prompt: str,
    ) -> None:
        self.id = stream_id
        self.rtsp_url = rtsp_url
        self.name = name
        self.fps = fps
        self.chunk_seconds = chunk_seconds
        self.prompt = prompt
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None


def _safe_put(queue: asyncio.Queue[dict[str, Any]], item: dict[str, Any]) -> None:
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        logger.warning("Stream caption queue full; dropping item")


class StreamManager:
    """Tracks active RTSP captioning streams and bridges worker threads to the event loop."""

    def __init__(self) -> None:
        self._streams: dict[str, _Stream] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def add(
        self,
        rtsp_url: str,
        name: str,
        fps: float | None,
        chunk_seconds: float | None,
        prompt: str | None,
    ) -> str:
        stream_id = uuid.uuid4().hex[:12]
        stream = _Stream(
            stream_id,
            rtsp_url,
            name or stream_id,
            fps or settings.sample_fps,
            chunk_seconds or settings.chunk_seconds,
            prompt or settings.prompt,
        )
        self._streams[stream_id] = stream
        stream.thread = threading.Thread(target=self._worker, args=(stream,), daemon=True)
        stream.thread.start()
        return stream_id

    def get(self, stream_id: str) -> _Stream | None:
        return self._streams.get(stream_id)

    def delete(self, stream_id: str) -> bool:
        stream = self._streams.pop(stream_id, None)
        if stream is None:
            return False
        stream.stop_event.set()
        return True

    def shutdown(self) -> None:
        for stream in self._streams.values():
            stream.stop_event.set()
        self._streams.clear()

    def _emit(self, stream: _Stream, item: dict[str, Any]) -> None:
        if self._loop is None:
            logger.error("Event loop not attached; dropping stream item")
            return
        self._loop.call_soon_threadsafe(_safe_put, stream.queue, item)

    def _worker(self, stream: _Stream) -> None:
        cap = cv2.VideoCapture(stream.rtsp_url)
        if not cap.isOpened():
            msg = f"Could not open RTSP stream: {stream.rtsp_url}"
            logger.error(msg)
            self._emit(stream, {"type": "error", "error": msg})
            self._emit(stream, {"type": "end"})
            return

        interval = 1.0 / max(stream.fps, 1e-6)
        buffer: list[tuple[float, str]] = []
        start_wall = time.monotonic()
        last_sample = 0.0
        chunk_start = start_wall
        try:
            while not stream.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.5)
                    continue
                now = time.monotonic()
                if now - last_sample >= interval:
                    last_sample = now
                    buffer.append((now - start_wall, encode_frame(frame)))
                if now - chunk_start >= stream.chunk_seconds and buffer:
                    chunk, buffer = buffer, []
                    chunk_start = now
                    self._caption_and_emit(stream, chunk)
        finally:
            cap.release()
            self._emit(stream, {"type": "end"})

    def _caption_and_emit(self, stream: _Stream, chunk: list[tuple[float, str]]) -> None:
        start, end = chunk[0][0], chunk[-1][0]
        frames = [b64 for _, b64 in subsample(chunk, settings.max_frames)]
        prompt = (
            f"{stream.prompt}\n\nThese {len(frames)} frames are sampled in order from the live "
            f"stream segment spanning {start:.1f}s to {end:.1f}s since stream start."
        )
        try:
            text = caption_sync(build_caption_payload(frames, prompt))
        except Exception as exc:  # surface any captioning failure to the SSE client
            logger.exception("Captioning failed for stream %s", stream.id)
            self._emit(stream, {"type": "error", "error": str(exc)})
            return
        self._emit(
            stream,
            {"type": "caption", "start": round(start, 2), "end": round(end, 2), "text": text.strip()},
        )
