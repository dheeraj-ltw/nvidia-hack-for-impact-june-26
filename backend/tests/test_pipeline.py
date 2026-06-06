from app.ai.base import Frame, ReasoningInput
from app.ai.null import NullAIService
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


async def test_null_backend_emits_no_ai_events() -> None:
    """With no model connected the pipeline stays silent — no fake detections or guidance."""
    pipeline_events: list[PatrolEvent] = []

    async def emit(event: PatrolEvent) -> None:
        pipeline_events.append(event)

    pipeline = SessionPipeline(ai_service=NullAIService(), emit=emit)
    await pipeline.handle_frame(Frame(ts=1.0, jpeg=b"x" * 600, width=640, height=480))
    await pipeline.handle_audio(b"y" * 600, timestamp=1.0)

    # A DetectionEvent with no boxes is still emitted; assert nothing meaningful fires.
    assert all(not isinstance(event, GuidanceEvent | SpeechEvent) for event in pipeline_events)
    detections = [event for event in pipeline_events if isinstance(event, DetectionEvent)]
    assert all(detection.boxes == [] for detection in detections)
