from app.ai.base import ReasoningInput
from app.ai.stub import StubAIService
from app.models.events import PatrolEvent, TranscriptEvent
from app.pipeline.orchestrator import SessionPipeline


async def test_handle_audio_emits_transcript() -> None:
    ai_service = StubAIService()
    emitted: list[PatrolEvent] = []

    async def emit(event: PatrolEvent) -> None:
        emitted.append(event)

    pipeline = SessionPipeline(ai_service=ai_service, emit=emit)
    await pipeline.handle_audio(b"x" * 600, timestamp=1.0)

    transcripts = [event for event in emitted if isinstance(event, TranscriptEvent)]
    assert len(transcripts) == 1
    assert transcripts[0].text  # stub returns a canned phrase
    assert transcripts[0].speaker == "officer"  # no officer enrolled


async def test_reason_returns_none_without_relevant_objects() -> None:
    ai_service = StubAIService()
    guidance = await ai_service.reason(
        ReasoningInput(ts=0.0, transcript="", scene_summary="empty", detections=[])
    )
    assert guidance is None
