"""Deterministic AI stub — no external calls, no keys.

Produces plausible, varied output driven by frame/transcript content so the full realtime
loop and UI are demoable end-to-end. Swappable for the live adapter via AI_BACKEND.
"""

from __future__ import annotations

import base64

from app.ai.base import Frame, ReasoningInput, Transcription
from app.models.events import BoundingBox, GuidanceEvent, LegalCitation, Severity

# A single silent MPEG-1 Layer III frame, so the SpeechEvent path is exercised without TTS.
_SILENT_MP3 = base64.b64decode(
    "//uQZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAA=="
)

DETECTABLE_LABELS = ["person", "vehicle", "license_plate", "bag", "weapon?"]

# FNV-1a 32-bit constants — a fast, well-known content hash.
_FNV_OFFSET_BASIS = 2166136261
_FNV_PRIME = 16777619
_HASH_SAMPLE_BYTES = 512


def content_hash(data: bytes) -> int:
    """Deterministic FNV-1a hash over the first bytes — same input, same scene every time."""
    digest = _FNV_OFFSET_BASIS
    for byte_value in data[:_HASH_SAMPLE_BYTES]:
        digest = ((digest ^ byte_value) * _FNV_PRIME) & 0xFFFFFFFF
    return digest


class StubAIService:
    """Implements the AIService protocol with deterministic, content-derived output."""

    async def transcribe(self, audio_chunk: bytes) -> Transcription:
        if len(audio_chunk) < 256:
            return Transcription(text="", is_final=False)
        phrases = [
            "Dispatch, proceeding on foot.",
            "Subject is cooperative.",
            "Requesting backup at this location.",
            "Vehicle plate confirmed.",
        ]
        text = phrases[content_hash(audio_chunk) % len(phrases)]
        return Transcription(text=text, is_final=True)

    async def analyze_frame(self, frame: Frame) -> tuple[list[BoundingBox], str]:
        seed = content_hash(frame.jpeg)
        detection_count = (seed % 3) + 1
        boxes: list[BoundingBox] = []
        for index in range(detection_count):
            label = DETECTABLE_LABELS[(seed >> (index * 3)) % len(DETECTABLE_LABELS)]
            boxes.append(
                BoundingBox(
                    label=label,
                    confidence=0.6 + ((seed >> index) % 35) / 100,
                    x=((seed >> (index + 1)) % 60) / 100,
                    y=((seed >> (index + 2)) % 60) / 100,
                    w=0.15 + ((seed >> index) % 20) / 100,
                    h=0.2 + ((seed >> index) % 25) / 100,
                )
            )
        summary = f"{detection_count} object(s) in view: " + ", ".join(
            box.label for box in boxes
        )
        return boxes, summary

    async def reason(self, context: ReasoningInput) -> GuidanceEvent | None:
        detected_labels = {box.label for box in context.detections}
        if "weapon?" in detected_labels:
            return GuidanceEvent(
                ts=context.ts,
                suggestion=(
                    "Possible weapon detected. Maintain distance and verify before acting."
                ),
                rationale="Visual classifier flagged a potential weapon with low certainty.",
                severity=Severity.CRITICAL,
                citations=[
                    LegalCitation(
                        title="Use of Force — Reasonableness Standard",
                        reference="Graham v. Connor, 490 U.S. 386 (1989)",
                        snippet=(
                            "Force must be objectively reasonable given the "
                            "totality of circumstances."
                        ),
                    )
                ],
            )
        if "vehicle" in detected_labels or "license_plate" in detected_labels:
            return GuidanceEvent(
                ts=context.ts,
                suggestion=(
                    "Vehicle stop: state the reason for the stop before requesting documents."
                ),
                rationale="A lawful stop requires articulable reasonable suspicion.",
                severity=Severity.CAUTION,
                citations=[
                    LegalCitation(
                        title="Investigatory Stops",
                        reference="Terry v. Ohio, 392 U.S. 1 (1968)",
                    )
                ],
            )
        return None

    async def speak(self, text: str) -> bytes:
        return _SILENT_MP3
