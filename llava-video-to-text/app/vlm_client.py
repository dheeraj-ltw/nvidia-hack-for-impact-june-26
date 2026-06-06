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
"""Client for the co-located vLLM OpenAI-compatible server serving LLaVA.

Mirrors the message shape used by ``agent/src/vss_agents/tools/video_caption.py``: a text prompt
followed by N base64 ``image_url`` entries posted to ``/v1/chat/completions``.
"""

from typing import Any

import httpx

from .config import settings


def _build_messages(frames: list[str], prompt: str) -> list[dict[str, Any]]:
    """Build OpenAI chat messages: one text part followed by base64 image parts."""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
    return [{"role": "user", "content": content}]


def build_caption_payload(
    frames: list[str],
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Build a chat-completions request body for captioning a set of frames."""
    return {
        "model": model or settings.vlm_model,
        "messages": _build_messages(frames, prompt),
        "max_tokens": max_tokens or settings.max_tokens,
        "temperature": settings.temperature if temperature is None else temperature,
    }


def parse_caption(data: dict[str, Any]) -> str:
    """Extract the assistant message content from a chat-completions response."""
    return data["choices"][0]["message"]["content"]


def caption_sync(payload: dict[str, Any], *, base_url: str | None = None, timeout: float | None = None) -> str:
    """Blocking caption call — used by the RTSP worker thread (see streams.py)."""
    url = f"{base_url or settings.vlm_base_url}/v1/chat/completions"
    with httpx.Client(timeout=timeout or settings.request_timeout) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return parse_caption(resp.json())


class VlmClient:
    """Async client for the local vLLM server, shared across the FastAPI app."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self.base_url = base_url or settings.vlm_base_url
        self._client = httpx.AsyncClient(timeout=timeout or settings.request_timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> dict[str, Any]:
        resp = await self._client.get(f"{self.base_url}/v1/models")
        resp.raise_for_status()
        return resp.json()

    async def caption(self, frames: list[str], prompt: str, **kwargs: Any) -> str:
        payload = build_caption_payload(frames, prompt, **kwargs)
        resp = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return parse_caption(resp.json())

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Pass a raw OpenAI chat-completions request straight through to vLLM."""
        resp = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def summarize(self, text: str, instruction: str) -> str:
        """Text-only aggregation of per-chunk captions into one description."""
        payload = {
            "model": settings.vlm_model,
            "messages": [{"role": "user", "content": f"{instruction}\n\n{text}"}],
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
        }
        resp = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return parse_caption(resp.json())
