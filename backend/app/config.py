from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["development", "production"] = "development"
    log_level: str = "info"
    cors_origins: str = "http://localhost:3000"

    # AI provider selection: stub (deterministic), live (NIM + ElevenLabs)
    ai_backend: Literal["stub", "live"] = "stub"

    # Nebius Token Factory — the single hosted-model endpoint, used for two calls:
    #   1. nebius_vlm_model (Qwen2.5-VL) captions a frame into a scene summary, and
    #   2. nemotron_model composes that summary + transcript into the SCENE CARD
    #      (see app.ai.nemotron) before the PoliceAI reasoner runs.
    # Both share nebius_api_key/nebius_base_url. Leave nebius_api_key blank to skip vision and
    # the compose step: reason() then falls back to the deterministic card from build_scene_card.
    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    nebius_vlm_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    nemotron_model: str = "nvidia/nemotron-3-super-120b-a12b"

    # Scene-card compiler — Nebius-hosted Nemotron 3 Super. Reuses nebius_api_key but on a
    # separate regional endpoint. Compiles each recorded session's transcript + VLM scene into
    # a factual SCENE CARD (no legal reasoning) — see app/ai/scene_compiler.py.
    nebius_scene_base_url: str = "https://api.tokenfactory.us-central1.nebius.com/v1/"
    nebius_scene_model: str = "nvidia/nemotron-3-super-120b-a12b"

    # Fine-tuned PoliceAI reasoner — an OpenAI-compatible server (see police-llm/).
    # The model was trained on SCENE CARD prompts; reason() rebuilds that exact format.
    # The server runs unauthenticated, so no API key is sent.
    policeai_base_url: str = "http://police-llm:8000/v1"
    policeai_model: str = "policeai"

    # Incident reporting — webhooks fired with the report when a session ends.
    # Each is optional; an unset URL is simply skipped. All dispatch is best-effort.
    logs_webhook_url: str = ""
    report_webhook_url: str = ""
    cop_registry_webhook_url: str = ""
    webhook_timeout_seconds: float = 10.0

    # ElevenLabs — speech-to-text + text-to-speech
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_tts_model: str = "eleven_multilingual_v2"
    elevenlabs_stt_model: str = "scribe_v1"
    # Pin STT to one language so Scribe doesn't auto-detect a different one per clip (which makes
    # non-speech audio get tagged in random languages). Blank = auto-detect. ISO-639-3, e.g. "eng".
    elevenlabs_stt_language: str = "eng"

    # Speaker identification: minimum cosine similarity for a voice to count as the officer.
    speaker_match_threshold: float = 0.70

    # Where to mirror per-session SCENE CARD JSONL files (request format) for dataset
    # collection. Blank → the repo's data/output/. Object storage always gets a copy too.
    scene_card_dir: str = ""

    # Object storage (MinIO / S3) — stores recorded session media + manifests
    s3_endpoint: str = "http://minio:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "evidence"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
