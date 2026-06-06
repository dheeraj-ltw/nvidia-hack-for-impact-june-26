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

    # NVIDIA NIM (vision model lands in a separate PR; kept for that integration)
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nemotron_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"

    # Nebius Token Factory — hosted Qwen2.5-VL. Powers live scene captioning (the
    # Video→VLM→scene-summary branch) when AI_BACKEND=live. Leave nebius_api_key blank to
    # skip vision: analyze_frame then degrades to no summary and the reasoner runs on audio.
    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    nebius_vlm_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"

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

    # Speaker identification: minimum cosine similarity for a voice to count as the officer.
    speaker_match_threshold: float = 0.70

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
