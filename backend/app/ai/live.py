"""Live AI backend — ElevenLabs speech, fine-tuned PoliceAI reasoning.

- transcribe: ElevenLabs Speech-to-Text (Scribe). Expects a *complete* audio file.
- speak: ElevenLabs Text-to-Speech, returns mp3 bytes.
- analyze_frame: no vision model is wired yet (lands in a separate PR), so this returns
  no detections.
- reason: the fine-tuned PoliceAI model via its OpenAI-compatible server (police-llm/),
  prompted with the SCENE CARD format it was trained on.

Network calls are defensive: any provider error degrades to an empty result rather than
crashing the realtime session.
"""

from __future__ import annotations

import logging

import httpx

from app.ai import policeai
from app.ai.base import Frame, ReasoningInput
from app.config import get_settings
from app.models.events import BoundingBox, GuidanceEvent

logger = logging.getLogger(__name__)

_ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class LiveAIService:
    """Implements the AIService protocol against ElevenLabs + the fine-tuned PoliceAI model."""

    def __init__(self) -> None:
        settings = get_settings()
        self._elevenlabs_key = settings.elevenlabs_api_key
        self._voice_id = settings.elevenlabs_voice_id
        self._tts_model = settings.elevenlabs_tts_model
        self._stt_model = settings.elevenlabs_stt_model
        self._policeai_base_url = settings.policeai_base_url.rstrip("/")
        self._policeai_model = settings.policeai_model

    async def transcribe(self, audio_chunk: bytes) -> tuple[str, bool]:
        if not self._elevenlabs_key or len(audio_chunk) < 1024:
            return "", False
        logger.info("→ ElevenLabs STT model=%s (%d bytes)", self._stt_model, len(audio_chunk))
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/speech-to-text",
                    headers={"xi-api-key": self._elevenlabs_key},
                    data={"model_id": self._stt_model},
                    files={"file": ("clip.webm", audio_chunk, "audio/webm")},
                )
            response.raise_for_status()
            text = response.json().get("text", "").strip()
            return text, True
        except (httpx.HTTPError, ValueError) as error:
            logger.warning("ElevenLabs STT failed: %s", error)
            return "", False

    async def analyze_frame(self, frame: Frame) -> tuple[list[BoundingBox], str]:
        # No vision model is connected yet; frames are still recorded for later analysis.
        return [], ""

    async def reason(self, context: ReasoningInput) -> GuidanceEvent | None:
        # Need *some* context to reason about — skip empty turns to save a model call.
        if not context.transcript.strip() and not context.scene_summary.strip():
            return None

        headers = {"Content-Type": "application/json"}

        logger.info(
            "→ PoliceAI %s/chat/completions model=%s",
            self._policeai_base_url,
            self._policeai_model,
        )
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{self._policeai_base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self._policeai_model,
                        "messages": policeai.build_messages(context),
                        "temperature": 0.2,
                        "max_tokens": 700,
                    },
                )
            response.raise_for_status()
            # TypeError guards against an unexpected response shape (e.g. choices is not a
            # list, or message is not a dict) — degrade to no guidance, never crash the session.
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            logger.warning("PoliceAI reasoning failed: %s", error)
            return None

        logger.info("← PoliceAI returned %d chars", len(content))
        return policeai.parse_guidance(content, context.ts)

    async def speak(self, text: str) -> bytes:
        if not self._elevenlabs_key or not self._voice_id or not text:
            return b""
        logger.info("→ ElevenLabs TTS model=%s (%d chars)", self._tts_model, len(text))
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/text-to-speech/{self._voice_id}",
                    headers={
                        "xi-api-key": self._elevenlabs_key,
                        "accept": "audio/mpeg",
                    },
                    json={"text": text, "model_id": self._tts_model},
                )
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as error:
            logger.warning("ElevenLabs TTS failed: %s", error)
            return b""
