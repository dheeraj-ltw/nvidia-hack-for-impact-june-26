"""Live AI backend — ElevenLabs speech, optional NVIDIA Nemotron reasoning.

- transcribe: ElevenLabs Speech-to-Text (Scribe). Expects a *complete* audio file.
- speak: ElevenLabs Text-to-Speech, returns mp3 bytes.
- analyze_frame: no vision model is wired yet, so this returns no detections.
- reason: NVIDIA Nemotron when an API key is configured; otherwise no guidance.

Network calls are defensive: any provider error degrades to an empty result rather than
crashing the realtime session.
"""

from __future__ import annotations

import logging

import httpx

from app.ai.base import Frame, ReasoningInput
from app.config import get_settings
from app.models.events import BoundingBox, GuidanceEvent, LegalCitation, Severity

logger = logging.getLogger(__name__)

_ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

_REASONING_SYSTEM_PROMPT = (
    "You are a law-aligned decision-support assistant for a police officer on patrol. "
    "Given the scene and recent dialogue, give one short, actionable, lawful suggestion "
    "and cite the legal basis. You advise; the officer decides. If nothing warrants "
    "guidance, reply with exactly: NONE."
)


class LiveAIService:
    """Implements the AIService protocol against ElevenLabs and (optionally) NVIDIA NIM."""

    def __init__(self) -> None:
        settings = get_settings()
        self._elevenlabs_key = settings.elevenlabs_api_key
        self._voice_id = settings.elevenlabs_voice_id
        self._tts_model = settings.elevenlabs_tts_model
        self._stt_model = settings.elevenlabs_stt_model
        self._nvidia_key = settings.nvidia_api_key
        self._nvidia_base_url = settings.nvidia_base_url
        self._nemotron_model = settings.nemotron_model

    async def transcribe(self, audio_chunk: bytes) -> tuple[str, bool]:
        if not self._elevenlabs_key or len(audio_chunk) < 1024:
            return "", False
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
        if not self._nvidia_key:
            return None
        prompt = (
            f"Scene: {context.scene_summary or 'n/a'}\n"
            f"Recent dialogue: {context.transcript or 'n/a'}\n"
            "Provide guidance."
        )
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{self._nvidia_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._nvidia_key}"},
                    json={
                        "model": self._nemotron_model,
                        "messages": [
                            {"role": "system", "content": _REASONING_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 200,
                    },
                )
            response.raise_for_status()
            suggestion = response.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            logger.warning("Nemotron reasoning failed: %s", error)
            return None

        if not suggestion or suggestion.upper() == "NONE":
            return None
        return GuidanceEvent(
            ts=context.ts,
            suggestion=suggestion,
            severity=Severity.INFO,
            citations=[LegalCitation(title="Model-cited basis", reference="see suggestion")],
        )

    async def speak(self, text: str) -> bytes:
        if not self._elevenlabs_key or not self._voice_id or not text:
            return b""
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
