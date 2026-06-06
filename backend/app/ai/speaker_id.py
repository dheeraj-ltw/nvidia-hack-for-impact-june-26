"""Speaker identification for patrol sessions.

ElevenLabs Scribe can *diarize* audio (split it into anonymous speakers speaker_0 /
speaker_1 / ...) but cannot recognise a *specific* person. So this module pairs ElevenLabs
diarization with a small local speaker-embedding model (resemblyzer) to answer "which
anonymous speaker is the officer?":

    conversation audio -> ElevenLabs STT (diarize=True, word timestamps)
                       -> slice each speaker's audio by those timestamps
                       -> voice embedding per speaker, compared against the officer's
                          enrollment embedding -> cosine match -> that speaker = "officer"
                       -> relabel: officer / person1 / person2 / ...

This is a backend port of src/speaker_id.py adapted for the FastAPI app:
  - audio is decoded with the ffmpeg already in the image (no librosa/soundfile), since the
    recorded track is concatenated WebM fragments that soundfile can't read reliably;
  - diarization uses httpx (mirroring app.ai.live) rather than the ElevenLabs SDK;
  - the officer's reference embedding is computed once at enrollment and reused, so live
    matching never re-embeds the reference.

All functions here are synchronous and CPU/IO-bound; async callers should wrap them in
`asyncio.to_thread` so they never block the event loop.

Public API:
    embed_reference(audio_bytes) -> list[float]          # enrollment
    warm_up()                                            # preload the encoder
    match_clip(clip_bytes, ref_embedding) -> (label, similarity)   # live officer-vs-other
    transcribe_with_speakers(audio_bytes) -> dict        # diarized words + speakers
    identify_officer(conversation_bytes, ref_embedding) -> dict    # post-session pass
"""

from __future__ import annotations

import logging
import subprocess
import warnings
from typing import Any

import httpx
import numpy as np

from app.config import get_settings

# webrtcvad (pulled in by resemblyzer) imports the deprecated pkg_resources at import
# time, which emits a noisy UserWarning. It's harmless; silence it.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000  # resemblyzer operates at 16 kHz mono.
_ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
_REQUEST_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
# Bound the ffmpeg decode so a malformed/crafted upload can't hang a worker thread forever.
_FFMPEG_TIMEOUT_SECONDS = 60

# One shared VoiceEncoder, loaded lazily on first use (weights are bundled with resemblyzer,
# so no download is needed).
_ENCODER = None


class SpeakerIdError(RuntimeError):
    """Raised when speaker identification cannot proceed (e.g. unembeddable audio)."""


def _encoder() -> Any:
    global _ENCODER
    if _ENCODER is None:
        from resemblyzer import VoiceEncoder

        _ENCODER = VoiceEncoder(verbose=False)
    return _ENCODER


def warm_up() -> None:
    """Preload the encoder so the first live match isn't penalised by model load time."""
    _encoder()


def decode_to_wav16k(audio: bytes) -> np.ndarray:
    """Decode an arbitrary audio buffer (WebM/MP3/WAV) to a float32 mono 16 kHz waveform.

    Uses the ffmpeg binary already present in the image (see recording/encoder.py). The
    recorded track is concatenated MediaRecorder WebM fragments, which soundfile/librosa
    can't decode reliably from a buffer — piping through ffmpeg is the robust path.
    """
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE),
                "pipe:1",
            ],
            input=audio,
            capture_output=True,
            check=True,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:  # ffmpeg not installed
        raise SpeakerIdError("ffmpeg is not available to decode audio") from error
    except subprocess.TimeoutExpired as error:
        raise SpeakerIdError("ffmpeg timed out decoding audio") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", "ignore")[:200]
        raise SpeakerIdError(f"ffmpeg failed to decode audio: {detail}") from error

    # Copy out of the read-only buffer (downstream resemblyzer ops need a writable array).
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wav.size == 0:
        raise SpeakerIdError("Decoded audio is empty — the clip may have no audio track.")
    return wav


def _embed_wav(wav: np.ndarray) -> np.ndarray | None:
    """Embed a 16 kHz mono waveform into a unit-norm speaker d-vector, or None if empty."""
    from resemblyzer import preprocess_wav

    try:
        processed = preprocess_wav(wav, source_sr=SAMPLE_RATE)
    except Exception:  # noqa: BLE001 - VAD can choke on very short/empty slices
        processed = wav
    if processed is None or len(processed) == 0:
        return None
    return np.asarray(_encoder().embed_utterance(processed), dtype=np.float32)


def embed_reference(audio: bytes) -> list[float]:
    """Embed an officer's enrollment clip into a reusable d-vector (JSON-serialisable list).

    Raises SpeakerIdError if the clip is too short/silent to embed.
    """
    emb = _embed_wav(decode_to_wav16k(audio))
    if emb is None:
        raise SpeakerIdError(
            "Could not embed the voice sample — it may be too short or silent."
        )
    return [float(x) for x in emb]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # resemblyzer embeddings are unit-norm, so the dot product is the cosine similarity.
    return float(np.dot(a, b))


def match_clip(clip: bytes, ref_embedding: list[float]) -> tuple[str, float]:
    """Live officer-vs-other label for a single short clip.

    Returns ("officer", sim) if the clip's voice matches the officer reference above the
    configured threshold, else ("subject", sim). On any decode/embed failure returns
    ("unknown", -1.0) so the realtime loop degrades gracefully.
    """
    try:
        emb = _embed_wav(decode_to_wav16k(clip))
    except SpeakerIdError:
        return "unknown", -1.0
    if emb is None:
        return "unknown", -1.0
    sim = _cosine(np.asarray(ref_embedding, dtype=np.float32), emb)
    threshold = get_settings().speaker_match_threshold
    return ("officer" if sim >= threshold else "subject"), sim


def transcribe_with_speakers(audio: bytes) -> dict[str, Any]:
    """Transcribe `audio` with anonymous speaker diarization + word timestamps.

    Returns {"text", "words": [{"text","start","end","speaker_id"}], "speakers": [...]}.
    Speaker ids are anonymous ('speaker_0', 'speaker_1', ...). Requires an ElevenLabs key;
    raises SpeakerIdError if it's missing or the request fails.
    """
    settings = get_settings()
    if not settings.elevenlabs_api_key:
        raise SpeakerIdError("No ELEVENLABS_API_KEY configured for diarized transcription.")

    try:
        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            response = client.post(
                f"{_ELEVENLABS_BASE}/speech-to-text",
                headers={"xi-api-key": settings.elevenlabs_api_key},
                data={
                    "model_id": settings.elevenlabs_stt_model,
                    "diarize": "true",
                    "timestamps_granularity": "word",
                },
                files={"file": ("conversation.webm", audio, "audio/webm")},
            )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise SpeakerIdError(f"ElevenLabs diarized STT failed: {error}") from error

    words: list[dict[str, Any]] = []
    for w in payload.get("words") or []:
        if w.get("type", "word") != "word":
            continue
        speaker = w.get("speaker_id")
        start = w.get("start")
        if speaker is None or start is None:
            continue
        end = w.get("end")
        words.append(
            {
                "text": w.get("text", ""),
                "start": float(start),
                "end": float(end if end is not None else start),
                "speaker_id": speaker,
            }
        )

    speakers: list[str] = []
    for w in words:
        if w["speaker_id"] not in speakers:
            speakers.append(w["speaker_id"])

    return {"text": payload.get("text", "") or "", "words": words, "speakers": speakers}


def _merge_segments(
    words: list[dict[str, Any]], label_for: dict[str, str]
) -> list[dict[str, Any]]:
    """Merge consecutive same-speaker words into turns with the relabelled speaker."""
    segments: list[dict[str, Any]] = []
    for w in words:
        label = label_for.get(w["speaker_id"], w["speaker_id"])
        if segments and segments[-1]["speaker_label"] == label:
            segments[-1]["text"] += (" " + w["text"]).rstrip()
            segments[-1]["end"] = w["end"]
        else:
            segments.append(
                {"speaker_label": label, "start": w["start"], "end": w["end"], "text": w["text"]}
            )
    for seg in segments:
        seg["text"] = seg["text"].strip()
    return segments


def identify_officer(conversation: bytes, ref_embedding: list[float]) -> dict[str, Any]:
    """Transcribe a conversation and label the officer's turns vs everyone else.

    The diarized speaker whose voice best matches `ref_embedding` (the officer's enrollment
    embedding) is relabelled `officer`; the rest become `person1`, `person2`, ... by order of
    first appearance.

    Returns:
      {
        "officer_speaker_id": str | None,
        "similarities": {speaker_id: float},
        "labels": {speaker_id: "officer" | "personN"},
        "segments": [{speaker_label, start, end, text}],
        "transcript": "[officer] ...\n[person1] ...",
        "matched": bool,           # best similarity >= threshold
      }
    Raises SpeakerIdError if diarization is unavailable.
    """
    diar = transcribe_with_speakers(conversation)
    words, speakers = diar["words"], diar["speakers"]
    if not words:
        return {
            "officer_speaker_id": None, "similarities": {}, "labels": {},
            "segments": [], "transcript": diar["text"], "matched": False,
        }

    conv_wav = decode_to_wav16k(conversation)
    ref_emb = np.asarray(ref_embedding, dtype=np.float32)

    # Per-speaker embedding from concatenated word-time slices of the conversation.
    similarities: dict[str, float] = {}
    for spk in speakers:
        chunks = [
            conv_wav[int(w["start"] * SAMPLE_RATE):int(w["end"] * SAMPLE_RATE)]
            for w in words if w["speaker_id"] == spk
        ]
        chunks = [c for c in chunks if len(c)]
        if not chunks:
            similarities[spk] = -1.0
            continue
        emb = _embed_wav(np.concatenate(chunks))
        similarities[spk] = _cosine(ref_emb, emb) if emb is not None else -1.0

    officer_spk = max(similarities, key=lambda s: similarities[s]) if similarities else None
    threshold = get_settings().speaker_match_threshold
    matched = officer_spk is not None and similarities[officer_spk] >= threshold

    # Relabel: best match -> officer, the rest -> person1, person2, ... (first-appearance order).
    labels: dict[str, str] = {}
    n = 0
    for spk in speakers:
        if spk == officer_spk:
            labels[spk] = "officer"
        else:
            n += 1
            labels[spk] = f"person{n}"

    segments = _merge_segments(words, labels)
    transcript = "\n".join(f"[{s['speaker_label']}] {s['text']}" for s in segments)

    return {
        "officer_speaker_id": officer_spk,
        "similarities": similarities,
        "labels": labels,
        "segments": segments,
        "transcript": transcript,
        "matched": matched,
    }
