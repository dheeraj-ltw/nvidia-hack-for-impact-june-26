from app.ai.vlm import VlmCaptioner

_NEBIUS = "https://api.tokenfactory.nebius.com/v1"
_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"


async def test_describe_frame_disabled_without_api_key() -> None:
    # No NEBIUS_API_KEY configured => vision is off, no network call, empty summary.
    captioner = VlmCaptioner(base_url=_NEBIUS, model=_MODEL, api_key="")
    assert await captioner.describe_frame(b"\xff\xd8\xff" * 100) == ""


async def test_describe_frame_empty_on_empty_frame() -> None:
    # A missing frame degrades to no summary rather than posting an empty image.
    captioner = VlmCaptioner(base_url=_NEBIUS, model=_MODEL, api_key="test-key")
    assert await captioner.describe_frame(b"") == ""
