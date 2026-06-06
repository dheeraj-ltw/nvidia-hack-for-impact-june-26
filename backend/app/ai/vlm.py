"""Live vision-language captioning — the Video→VLM→scene-summary branch.

`describe_frame` posts a single JPEG frame to Nebius Token Factory's hosted Qwen2.5-VL
(the same provider the offline CLI uses, see src/video_to_text.py) and returns a one-line
scene description. That description becomes the "Scene" line of the SCENE CARD the reasoner
is prompted with, so guidance is grounded in what the camera sees rather than audio alone.

Kept deliberately small: one frame per call (the realtime loop already throttles to a sane
cadence), low max_tokens for latency, and any error degrades to an empty summary rather than
crashing the session — mirroring the rest of the live adapter.
"""

from __future__ import annotations

import base64
import logging

import httpx

logger = logging.getLogger(__name__)

# A concise prompt: the reasoner only needs a short situational summary, and short
# completions keep the per-frame latency inside the realtime budget.
_SCENE_PROMPT = (
    "You are the eyes of a police body-worn camera. In one or two sentences, describe the "
    "scene an officer is facing right now: the people (number, attire, visible actions, "
    "demeanour) and any notable objects (vehicles, bags, weapons). State only what is "
    "visible — do not speculate about intent, guilt, or anything off-camera."
)


class VlmCaptioner:
    """Async client for Nebius-hosted Qwen2.5-VL (OpenAI-compatible chat/completions)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: httpx.Timeout | float = 20.0,
        max_tokens: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_tokens = max_tokens

    async def describe_frame(self, jpeg: bytes) -> str:
        """Return a short scene description for a JPEG frame, or "" on any failure.

        With no Nebius API key configured, vision is off and this is a no-op.
        """
        if not self._api_key or not jpeg:
            return ""

        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        logger.info("→ Nebius VLM %s/chat/completions model=%s", self._base_url, self._model)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self._model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": _SCENE_PROMPT},
                                    {"type": "image_url", "image_url": {"url": data_url}},
                                ],
                            }
                        ],
                        "max_tokens": self._max_tokens,
                        "temperature": 0.0,
                    },
                )
            response.raise_for_status()
            # TypeError guards an unexpected response shape (choices not a list, message not a
            # dict) — degrade to no summary, never crash the realtime session.
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            logger.warning("Nebius VLM captioning failed: %s", error)
            return ""

        summary = (content or "").strip()
        logger.info("← Nebius VLM scene=%r", summary)
        return summary
