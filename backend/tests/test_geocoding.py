import httpx
import pytest

from app.ai import geocoding


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Route reverse_geocode's httpx.AsyncClient through a MockTransport (no real network)."""
    real_async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(geocoding.httpx, "AsyncClient", factory)


async def test_no_coords_returns_none() -> None:
    # Nothing to geocode => None, so the SCENE CARD keeps its "Unknown" fallback.
    assert await geocoding.reverse_geocode(None, None) is None
    assert await geocoding.reverse_geocode(51.54, None) is None


async def test_resolves_display_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["lat"] == "51.539"
        assert request.url.params["lon"] == "-0.143"
        return httpx.Response(200, json={"display_name": "High Street, Camden, London"})

    _patch_transport(monkeypatch, handler)
    assert await geocoding.reverse_geocode(51.539, -0.143) == "High Street, Camden, London"


async def test_http_error_falls_back_to_raw_coords(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    _patch_transport(monkeypatch, handler)
    # A failed lookup still grounds the card with coordinates rather than "Unknown".
    assert await geocoding.reverse_geocode(51.539, -0.143) == "51.53900, -0.14300"


async def test_empty_display_name_falls_back_to_raw_coords(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"display_name": ""})

    _patch_transport(monkeypatch, handler)
    assert await geocoding.reverse_geocode(51.539, -0.143) == "51.53900, -0.14300"
