"""Reverse geocoding for the patrol's GPS coordinates.

The frontend sends the officer's live position as ?lat=..&lon=.. on the WS URL. We turn that
into a human-readable place name (e.g. "High Street, Camden, London") for the SCENE CARD's
Location line, matching the format the PoliceAI reasoner was trained on.

Best-effort by design: any failure (no coords, network error, odd response shape) falls back to
the raw "lat, lon" string, so a patrol is never blocked on geocoding.
"""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


def _raw_coords(lat: float, lon: float) -> str:
    return f"{lat:.5f}, {lon:.5f}"


async def reverse_geocode(lat: float | None, lon: float | None) -> str | None:
    """Resolve coords to a place name, or the raw "lat, lon" string on failure.

    Returns None only when no coordinates were supplied — the caller then leaves the SCENE
    CARD's Location as "Unknown".
    """
    if lat is None or lon is None:
        return None

    settings = get_settings()
    logger.info("→ reverse geocode lat=%.5f lon=%.5f", lat, lon)
    try:
        async with httpx.AsyncClient(timeout=settings.geocoding_timeout_seconds) as client:
            response = await client.get(
                settings.geocoding_base_url,
                params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18},
                headers={"User-Agent": settings.geocoding_user_agent},
            )
        response.raise_for_status()
        name = (response.json().get("display_name") or "").strip()
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        logger.warning("Reverse geocoding failed: %s — using raw coords", error)
        return _raw_coords(lat, lon)

    if not name:
        return _raw_coords(lat, lon)
    logger.info("← reverse geocode %r", name)
    return name
