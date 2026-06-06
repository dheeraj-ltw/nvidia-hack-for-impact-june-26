import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, officers, sessions
from app.config import get_settings
from app.realtime import session
from app.storage import get_object_store


def _configure_logging(level: str) -> None:
    """Apply the configured log level to the app's loggers.

    The realtime path logs what it receives and sends at INFO, so the default level
    surfaces the end-to-end data flow without extra setup.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Ensure the media bucket exists before accepting any recordings.
    await get_object_store().ensure_bucket()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings.log_level)
    app = FastAPI(title="JARVIS API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(session.router)
    app.include_router(sessions.router)
    app.include_router(officers.router)
    return app


app = create_app()
