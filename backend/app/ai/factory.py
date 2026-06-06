from functools import lru_cache

from app.ai.base import AIService
from app.ai.stub import StubAIService
from app.config import get_settings


@lru_cache
def get_ai_service() -> AIService:
    """Resolve the active AI backend from settings.

    - ``stub``: deterministic local output for exercising the event path without keys.
    - ``live``: NVIDIA NIM + ElevenLabs adapter.
    """
    if get_settings().ai_backend == "live":
        from app.ai.live import LiveAIService

        return LiveAIService()
    return StubAIService()
