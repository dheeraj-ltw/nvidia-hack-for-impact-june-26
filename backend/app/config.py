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

    # NVIDIA NIM
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nemotron_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"

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
