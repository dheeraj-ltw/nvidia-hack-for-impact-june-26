"""Build incident reports from finalized sessions and dispatch them to webhooks.

When a patrol session ends, the manifest holds everything the report needs: the
transcript/guidance events, the officer, and the timing. We render that into a structured
IncidentReport, store it alongside the session media, and POST it to whichever of the three
sinks are configured — operations log, report store, and cop registry.

Dispatch is best-effort by design: a slow or failing webhook must never block finalizing a
session or surface as a 500 to the officer. Each delivery outcome is captured so callers
(and the manual endpoint) can see what actually went out.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from app.config import get_settings
from app.models.events import EventType
from app.models.session import (
    GuidanceSummary,
    IncidentReport,
    ReportResult,
    SessionManifest,
    WebhookDispatch,
)
from app.recording import session_prefix
from app.storage import ObjectStore

logger = logging.getLogger(__name__)


def _format_citation(citation: object) -> str:
    """Render one recorded LegalCitation payload as 'Title: reference (snippet)'."""
    if not isinstance(citation, dict):
        return ""
    title = str(citation.get("title", "")).strip()
    reference = str(citation.get("reference", "")).strip()
    snippet = str(citation.get("snippet") or "").strip()
    head = f"{title}: {reference}" if title and reference else title or reference
    if not head:
        return ""
    return f"{head} ({snippet})" if snippet else head


def _extract_transcript_and_guidance(
    manifest: SessionManifest,
) -> tuple[str, list[GuidanceSummary]]:
    """Single ordered pass over the events, splitting transcript lines from guidance.

    Events can be transcript or guidance; both need chronological order, so we sort once
    and fan out. Payloads are model_dump(mode="json") dicts (see SessionRecorder), but we
    defend against missing/oddly-typed fields so a malformed event can't break the report.
    """
    transcript_lines: list[str] = []
    guidance: list[GuidanceSummary] = []

    for event in sorted(manifest.events, key=lambda e: e.offset_seconds):
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.kind == EventType.TRANSCRIPT.value:
            text = str(payload.get("text", "")).strip()
            if not text:
                continue
            speaker = str(payload.get("speaker", "unknown"))
            transcript_lines.append(f"[{speaker}] {text}")
        elif event.kind == EventType.GUIDANCE.value:
            raw_citations = payload.get("citations", [])
            citations = raw_citations if isinstance(raw_citations, list) else []
            formatted = [_format_citation(citation) for citation in citations]
            guidance.append(
                GuidanceSummary(
                    offset_seconds=event.offset_seconds,
                    suggestion=str(payload.get("suggestion", "")),
                    severity=str(payload.get("severity", "info")),
                    citations=[citation for citation in formatted if citation],
                )
            )

    return "\n".join(transcript_lines), guidance


def build_report(manifest: SessionManifest, *, generated_at: float | None = None) -> IncidentReport:
    """Render a finalized session manifest into a structured incident report."""
    transcript, guidance = _extract_transcript_and_guidance(manifest)
    return IncidentReport(
        session_id=manifest.session_id,
        label=manifest.label,
        officer_id=manifest.officer_id,
        officer_name=manifest.officer_name,
        started_at=manifest.started_at,
        ended_at=manifest.ended_at,
        duration_seconds=manifest.duration_seconds,
        generated_at=generated_at if generated_at is not None else time.time(),
        frame_count=manifest.frame_count,
        transcript=transcript,
        guidance=guidance,
        speaker_labels=(manifest.diarization or {}).get("labels", {}),
        scene_summary=manifest.scene_summary or "",
    )


async def store_report(store: ObjectStore, report: IncidentReport) -> str:
    """Persist the report next to the session media; returns its object key."""
    key = f"{session_prefix(report.session_id)}/report.json"
    await store.put(key, report.model_dump_json(indent=2).encode("utf-8"), "application/json")
    return key


def _configured_targets() -> list[tuple[str, str]]:
    """(target name, url) pairs for every webhook that has a URL set."""
    settings = get_settings()
    candidates = [
        ("logs", settings.logs_webhook_url),
        ("report", settings.report_webhook_url),
        ("cop_registry", settings.cop_registry_webhook_url),
    ]
    return [(name, url.strip()) for name, url in candidates if url.strip()]


async def _post_one(
    client: httpx.AsyncClient, target: str, url: str, payload: dict[str, object]
) -> WebhookDispatch:
    """POST the report to one webhook, capturing the outcome. Never raises."""
    logger.info("→ webhook %s POST %s", target, url)
    try:
        response = await client.post(url, json={"target": target, "report": payload})
        response.raise_for_status()
        logger.info("← webhook %s delivered (%s)", target, response.status_code)
        return WebhookDispatch(
            target=target, url=url, delivered=True, status_code=response.status_code
        )
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        logger.warning("Webhook %s (%s) returned %s", target, url, status)
        return WebhookDispatch(
            target=target, url=url, delivered=False, status_code=status, error=str(error)
        )
    except httpx.HTTPError as error:
        logger.warning("Webhook %s (%s) failed: %s", target, url, error)
        return WebhookDispatch(target=target, url=url, delivered=False, error=str(error))


async def dispatch_report(report: IncidentReport) -> list[WebhookDispatch]:
    """POST the report to every configured webhook concurrently. Never raises."""
    targets = _configured_targets()
    if not targets:
        logger.info("No webhooks configured — skipping report dispatch for %s", report.session_id)
        return []

    settings = get_settings()
    timeout = httpx.Timeout(settings.webhook_timeout_seconds)
    payload = report.model_dump(mode="json")

    logger.info(
        "Dispatching report %s to %d webhook(s): %s",
        report.session_id,
        len(targets),
        ", ".join(name for name, _ in targets),
    )
    # Fire all webhooks at once so total latency is the slowest single call, not the sum.
    async with httpx.AsyncClient(timeout=timeout) as client:
        return list(
            await asyncio.gather(
                *(_post_one(client, target, url, payload) for target, url in targets)
            )
        )


async def generate_and_dispatch(
    store: ObjectStore, manifest: SessionManifest, *, generated_at: float | None = None
) -> ReportResult:
    """Build the report, store it, and fire the webhooks. The one-call entry point."""
    report = build_report(manifest, generated_at=generated_at)
    logger.info(
        "Built report for %s: %d guidance item(s), %d transcript chars",
        report.session_id,
        len(report.guidance),
        len(report.transcript),
    )
    await store_report(store, report)
    dispatches = await dispatch_report(report)
    return ReportResult(report=report, dispatches=dispatches)
