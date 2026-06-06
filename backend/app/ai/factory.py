from functools import lru_cache

from app.ai.base import AIService
from app.ai.null import NullAIService
from app.ai.stub import StubAIService
from app.config import get_settings


@lru_cache
def get_ai_service() -> AIService:
    """Resolve the active AI backend from settings.

    - ``null``: no model connected (default) — empty results, session still recorded.
    - ``stub``: deterministic local output for exercising the event path without keys.
    - ``live``: NVIDIA NIM + ElevenLabs adapter.
    """
    backend = get_settings().ai_backend
    if backend == "stub":
        return StubAIService()
    if backend == "live":
        from app.ai.live import LiveAIService

        return LiveAIService()
    return NullAIService()
