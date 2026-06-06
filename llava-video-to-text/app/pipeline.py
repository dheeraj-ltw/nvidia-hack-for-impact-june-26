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
"""Video-to-text pipeline: sample frames -> chunk -> caption -> aggregate."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from .config import settings
from .frames import sample_file
from .frames import subsample
from .vlm_client import VlmClient

Frame = tuple[float, str]


def group_into_chunks(frames: list[Frame], chunk_seconds: float) -> list[list[Frame]]:
    """Group timestamped frames into windows spanning ``chunk_seconds`` each."""
    chunks: list[list[Frame]] = []
    current: list[Frame] = []
    window_start: float | None = None
    for ts, b64 in frames:
        if window_start is None:
            window_start = ts
        if ts - window_start >= chunk_seconds and current:
            chunks.append(current)
            current = []
            window_start = ts
        current.append((ts, b64))
    if current:
        chunks.append(current)
    return chunks


def _chunk_prompt(prompt: str, start: float, end: float, n_frames: int, *, live: bool) -> str:
    origin = "live stream segment" if live else "video segment"
    return (
        f"{prompt}\n\nThese {n_frames} frames are sampled in order from the {origin} "
        f"spanning {start:.1f}s to {end:.1f}s."
    )


async def caption_chunk(
    vlm: VlmClient,
    chunk: list[Frame],
    prompt: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Caption a single chunk, returning a timestamped caption record."""
    start, end = chunk[0][0], chunk[-1][0]
    frames = [b64 for _, b64 in subsample(chunk, settings.max_frames)]
    full_prompt = _chunk_prompt(prompt, start, end, len(frames), live=False)
    async with sem:
        text = await vlm.caption(frames, full_prompt)
    return {"start": round(start, 2), "end": round(end, 2), "text": text.strip()}


async def caption_video_file(
    vlm: VlmClient,
    path: str,
    *,
    prompt: str | None = None,
    sample_fps: float | None = None,
    chunk_seconds: float | None = None,
) -> dict[str, Any]:
    """Caption a whole video file and return captions plus an aggregated summary."""
    prompt = prompt or settings.prompt
    frames = await asyncio.to_thread(sample_file, path, sample_fps or settings.sample_fps)
    if not frames:
        return {"captions": [], "summary": ""}

    chunks = group_into_chunks(frames, chunk_seconds or settings.chunk_seconds)
    sem = asyncio.Semaphore(settings.caption_concurrency)
    captions = await asyncio.gather(*(caption_chunk(vlm, c, prompt, sem) for c in chunks))

    if len(captions) > 1:
        joined = "\n".join(f"[{c['start']}-{c['end']}s] {c['text']}" for c in captions)
        summary = await vlm.summarize(
            joined,
            "Summarize the following timestamped video captions into a concise overall description:",
        )
    else:
        summary = captions[0]["text"] if captions else ""

    return {"captions": captions, "summary": summary.strip()}


async def iter_caption_video_file(
    vlm: VlmClient,
    path: str,
    *,
    prompt: str | None = None,
    sample_fps: float | None = None,
    chunk_seconds: float | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield captions one chunk at a time (ordered) for SSE streaming of an uploaded file."""
    prompt = prompt or settings.prompt
    frames = await asyncio.to_thread(sample_file, path, sample_fps or settings.sample_fps)
    chunks = group_into_chunks(frames, chunk_seconds or settings.chunk_seconds)
    sem = asyncio.Semaphore(settings.caption_concurrency)
    for chunk in chunks:
        yield await caption_chunk(vlm, chunk, prompt, sem)
