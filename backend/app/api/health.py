from fastapi import APIRouter

from app.ai import get_ai_service
from app.config import get_settings

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "env": settings.app_env,
        "ai_backend": type(get_ai_service()).__name__,
    }
