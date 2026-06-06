from app.ai.nemotron import SceneCardComposer

_NVIDIA = "https://integrate.api.nvidia.com/v1"
_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"

_DRAFT = "SCENE CARD (live patrol)\nLocation: High Street\nQUERY: next step?"


async def test_compose_disabled_without_api_key() -> None:
    # No NVIDIA_API_KEY configured => compose is off, no network call, empty result so the
    # caller keeps the deterministic draft card.
    composer = SceneCardComposer(base_url=_NVIDIA, model=_MODEL, api_key="")
    assert await composer.compose(_DRAFT) == ""


async def test_compose_empty_on_blank_draft() -> None:
    # A blank draft degrades to no composition rather than posting an empty card.
    composer = SceneCardComposer(base_url=_NVIDIA, model=_MODEL, api_key="test-key")
    assert await composer.compose("   ") == ""
