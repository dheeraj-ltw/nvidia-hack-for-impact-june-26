from app.models.events import EventType
from app.models.session import RecordedEvent, SessionManifest
from app.reporting import build_report


def _manifest_with_events() -> SessionManifest:
    return SessionManifest(
        session_id="abc123",
        label="Camden stop",
        started_at=100.0,
        ended_at=160.0,
        frame_count=12,
        officer_id="off1",
        officer_name="PC Aservda",
        events=[
            RecordedEvent(
                offset_seconds=2.0,
                kind=EventType.TRANSCRIPT.value,
                payload={"text": "Stop there please.", "speaker": "officer"},
            ),
            RecordedEvent(
                offset_seconds=5.0,
                kind=EventType.GUIDANCE.value,
                payload={
                    "suggestion": "Explain the grounds before searching.",
                    "severity": "caution",
                    "citations": [{"title": "Legal basis", "reference": "PACE 1984, s.2"}],
                },
            ),
            RecordedEvent(
                offset_seconds=4.0,
                kind=EventType.TRANSCRIPT.value,
                payload={"text": "Why?", "speaker": "subject"},
            ),
        ],
    )


def test_build_report_orders_transcript_and_summarizes_guidance() -> None:
    report = build_report(_manifest_with_events(), generated_at=200.0)

    assert report.session_id == "abc123"
    assert report.officer_name == "PC Aservda"
    assert report.duration_seconds == 60.0
    assert report.generated_at == 200.0
    # Transcript is speaker-tagged and ordered by offset (2.0 before 4.0).
    assert report.transcript == "[officer] Stop there please.\n[subject] Why?"
    assert len(report.guidance) == 1
    assert report.guidance[0].severity == "caution"
    assert report.guidance[0].citations == ["Legal basis: PACE 1984, s.2"]


def test_build_report_handles_empty_session() -> None:
    manifest = SessionManifest(session_id="empty", started_at=0.0, ended_at=0.0)
    report = build_report(manifest, generated_at=1.0)
    assert report.transcript == ""
    assert report.guidance == []
    assert report.duration_seconds == 0.0


def test_citation_preserves_snippet_and_trailing_colon() -> None:
    manifest = SessionManifest(
        session_id="cite",
        started_at=0.0,
        ended_at=1.0,
        events=[
            RecordedEvent(
                offset_seconds=0.0,
                kind=EventType.GUIDANCE.value,
                payload={
                    "suggestion": "Advise rights.",
                    "severity": "info",
                    "citations": [
                        {
                            "title": "PACE Code C",
                            # Trailing colon must survive — strip(': ') used to eat it.
                            "reference": "para 10.1:",
                            "snippet": "You do not have to say anything.",
                        }
                    ],
                },
            )
        ],
    )
    report = build_report(manifest, generated_at=1.0)
    assert report.guidance[0].citations == [
        "PACE Code C: para 10.1: (You do not have to say anything.)"
    ]


def test_guidance_tolerates_malformed_citations() -> None:
    manifest = SessionManifest(
        session_id="bad",
        started_at=0.0,
        ended_at=1.0,
        events=[
            RecordedEvent(
                offset_seconds=0.0,
                kind=EventType.GUIDANCE.value,
                payload={
                    "suggestion": "Proceed.",
                    "severity": "info",
                    # Non-dict entries must not raise — they're skipped.
                    "citations": ["a bare string", None, {"title": "Good", "reference": "s.1"}],
                },
            )
        ],
    )
    report = build_report(manifest, generated_at=1.0)
    assert report.guidance[0].citations == ["Good: s.1"]
