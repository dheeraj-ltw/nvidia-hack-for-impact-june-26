from app.ai.base import Frame, ReasoningInput
from app.ai.stub import StubAIService
from app.models.events import (
    DetectionEvent,
    GuidanceEvent,
    PatrolEvent,
    SpeechEvent,
)
from app.pipeline.orchestrator import SessionPipeline


async def test_stub_analyze_and_reason_emits_events() -> None:
    ai_service = StubAIService()
    emitted_events: list[PatrolEvent] = []

    async def emit(event: PatrolEvent) -> None:
        emitted_events.append(event)

    pipeline = SessionPipeline(ai_service=ai_service, emit=emit)

    # Probe several frame bodies until guidance fires, proving the full chain works:
    # analyze_frame -> detections -> reason -> guidance -> TTS speech event.
    for seed in range(50):
        emitted_events.clear()
        frame = Frame(ts=float(seed), jpeg=bytes([seed]) * 600, width=640, height=480)
        await pipeline.handle_frame(frame)
        emitted_types = {type(event) for event in emitted_events}
        if GuidanceEvent in emitted_types:
            assert DetectionEvent in emitted_types
            assert SpeechEvent in emitted_types  # guidance triggers TTS
            break
    else:
        raise AssertionError("expected at least one frame to produce guidance")


async def test_reason_returns_none_without_relevant_objects() -> None:
    ai_service = StubAIService()
    guidance = await ai_service.reason(
        ReasoningInput(ts=0.0, transcript="", scene_summary="empty", detections=[])
    )
    assert guidance is None
