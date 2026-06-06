"""Live AI backend — ElevenLabs speech, fine-tuned PoliceAI reasoning.

- transcribe: ElevenLabs Speech-to-Text (Scribe). Expects a *complete* audio file.
- speak: ElevenLabs Text-to-Speech, returns mp3 bytes.
- analyze_frame: a no-op on the live path. Vision (VLM scene captioning) runs post-session
  over the recorded frames so it can never block live transcription — see
  app.api.sessions.run_scene_analysis. Object detection is still pending.
- reason: NVIDIA NIM Nemotron composes the SCENE CARD from the transcript (+ any scene
  summary), then the fine-tuned PoliceAI model (OpenAI-compatible server, police-llm/) reasons
  over it. Nemotron is skipped when NVIDIA_API_KEY is unset, falling back to the deterministic
  card so reasoning never stalls on that step.

Network calls are defensive: any provider error degrades to an empty result rather than
crashing the realtime session.
"""

from __future__ import annotations

import logging

import httpx

from app.ai import policeai
from app.ai.base import Frame, ReasoningInput, Transcription
from app.ai.nemotron import SceneCardComposer
from app.ai.speaker_id import extract_words
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
        # Nemotron composes the SCENE CARD before PoliceAI reasons over it. This is the second
        # Nebius Token Factory call (the first is the VLM): same endpoint + key, but the
        # Nemotron model rather than Qwen-VL. No Nebius key => compose() is a no-op and we fall
        # back to the deterministic card.
        self._scene_card = SceneCardComposer(
            base_url=settings.nebius_base_url,
            model=settings.nemotron_model,
            api_key=settings.nebius_api_key,
        )

    async def transcribe(self, audio_chunk: bytes) -> Transcription:
        if not self._elevenlabs_key or len(audio_chunk) < 1024:
            return Transcription(text="", is_final=False)
        logger.info("→ ElevenLabs STT model=%s (%d bytes)", self._stt_model, len(audio_chunk))
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                response = await client.post(
                    f"{_ELEVENLABS_BASE}/speech-to-text",
                    headers={"xi-api-key": self._elevenlabs_key},
                    # Diarize + word timestamps so the speaker can be labeled from the dominant
                    # voice's audio alone (see speaker_id.label_dominant_speaker).
                    data={
                        "model_id": self._stt_model,
                        "diarize": "true",
                        "timestamps_granularity": "word",
                    },
                    files={"file": ("clip.webm", audio_chunk, "audio/webm")},
                )
            response.raise_for_status()
            payload = response.json()
            text = (payload.get("text") or "").strip()
            return Transcription(text=text, is_final=True, words=extract_words(payload))
        except (httpx.HTTPError, ValueError) as error:
            logger.warning("ElevenLabs STT failed: %s", error)
            return Transcription(text="", is_final=False)

    async def analyze_frame(self, frame: Frame) -> tuple[list[BoundingBox], str]:
        # No-op on the live path: vision runs post-session (run_scene_analysis) so it never
        # blocks transcription. Frames are recorded for that pass as they arrive.
        return [], ""

    async def reason(self, context: ReasoningInput) -> GuidanceEvent | None:
        # Need *some* context to reason about — skip empty turns to save a model call.
        if not context.transcript.strip() and not context.scene_summary.strip():
            return None

        # Nemotron refines the deterministic draft into a cleaner SCENE CARD; on any failure
        # (or no NVIDIA key) compose() returns "" and build_messages falls back to the draft.
        scene_card = await self._scene_card.compose(policeai.build_scene_card(context))

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
                        "messages": policeai.build_messages(context, scene_card=scene_card),
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
