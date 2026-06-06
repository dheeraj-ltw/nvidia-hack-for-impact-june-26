from app.ai.base import Frame, ReasoningInput
from app.ai.stub import StubAIService
from app.models.events import (
    DetectionEvent,
    GuidanceEvent,
    PatrolEvent,
    SpeechEvent,
)
from app.pipeline.orchestrator import SessionPipeline


async def test_scheduled_frame_analysis_emits_events() -> None:
    ai_service = StubAIService()
    emitted_events: list[PatrolEvent] = []

    async def emit(event: PatrolEvent) -> None:
        emitted_events.append(event)

    pipeline = SessionPipeline(ai_service=ai_service, emit=emit)
    # Reasoning only runs when there's dialogue to reason about, so seed the transcript.
    pipeline._transcript = "Subject is near a vehicle."

    # schedule_frame dispatches analysis as a detached task; aclose() awaits it. Probe several
    # frame bodies until one produces guidance, proving the full chain works off the loop:
    # schedule_frame -> analyze -> reason -> guidance -> TTS speech event.
    for seed in range(50):
        emitted_events.clear()
        frame = Frame(ts=float(seed), jpeg=bytes([seed]) * 600, width=640, height=480)
        pipeline.schedule_frame(frame)
        await pipeline.aclose()
        emitted_types = {type(event) for event in emitted_events}
        if GuidanceEvent in emitted_types:
            assert SpeechEvent in emitted_types  # guidance triggers TTS
            break
        # Advance past the throttle window so the next schedule isn't dropped as "too soon".
        pipeline._last_analyze_timestamp = 0.0
    else:
        raise AssertionError("expected at least one frame to produce guidance")


async def test_frame_without_transcript_skips_reasoning() -> None:
    ai_service = StubAIService()
    emitted_events: list[PatrolEvent] = []

    async def emit(event: PatrolEvent) -> None:
        emitted_events.append(event)

    pipeline = SessionPipeline(ai_service=ai_service, emit=emit)
    # No transcript yet — reasoning is skipped entirely, so no guidance/speech is emitted.
    pipeline.schedule_frame(Frame(ts=1.0, jpeg=b"\x00" * 600, width=640, height=480))
    await pipeline.aclose()
    assert not any(isinstance(event, GuidanceEvent | SpeechEvent) for event in emitted_events)


async def test_reason_returns_none_without_relevant_objects() -> None:
    ai_service = StubAIService()
    guidance = await ai_service.reason(
        ReasoningInput(ts=0.0, transcript="", scene_summary="empty", detections=[])
    )
    assert guidance is None
