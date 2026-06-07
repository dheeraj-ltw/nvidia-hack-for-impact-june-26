from app.ai.scene_compiler import SceneCardComposer

_NEBIUS = "https://api.tokenfactory.us-central1.nebius.com/v1"
_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def _composer() -> SceneCardComposer:
    return SceneCardComposer(base_url=_NEBIUS, model=_MODEL, api_key="test-key")


def _message(location: str | None) -> str:
    return _composer()._user_message(
        transcript="officer: stop there.",
        scene_text="one person near a parked car",
        officer_name="PC Smith",
        duration="0m 42s",
        updated_clock="12:00:00",
        location=location,
    )


def test_user_message_pins_known_location() -> None:
    # A reverse-geocoded location is pinned verbatim so the model reports it as-is.
    message = _message("High Street, Camden, London")
    assert "Location: High Street, Camden, London" in message
    # The "describe the setting" placeholder must not also appear for a known location.
    assert "type of setting observed" not in message


def test_user_message_falls_back_to_setting_placeholder() -> None:
    # No GPS fix => the model is asked to describe the observed setting (or write Unknown).
    message = _message(None)
    assert "type of setting observed" in message
