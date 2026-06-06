"""Identify the police officer's voice in a multi-speaker conversation.

ElevenLabs Scribe can diarize audio (split it into anonymous speakers
speaker_0 / speaker_1 / ...) but cannot recognise a *specific* person. So this
module pairs ElevenLabs diarization with a small local speaker-embedding model
(resemblyzer) to answer "which anonymous speaker is the officer?":

    conversation audio -> ElevenLabs STT (diarize=True, word timestamps)
                       -> slice each speaker's audio by those timestamps
                       -> voice embedding per speaker + embedding of an officer
                          reference clip -> cosine match -> that speaker = "officer"
                       -> relabel: officer / person1 / person2 / ...

Public API (frontend-ready):
    transcribe_with_speakers(audio, ...)  -> diarized words + speakers
    identify_officer(conversation, officer_reference, ...) -> labelled transcript

CLI:
    python src/speaker_id.py identify --conversation mix.wav --officer ref.wav
"""
from __future__ import annotations

import os
import pathlib
import tempfile
import warnings
from typing import Union

# webrtcvad (pulled in by resemblyzer) imports the deprecated pkg_resources at
# import time, which emits a noisy UserWarning. It's harmless; silence it.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Reuse the ElevenLabs client + paths from the existing audio module (flat src/ imports).
from audio import AUDIO_DIR, ROOT, make_elevenlabs_client

AudioInput = Union[str, "os.PathLike[str]", bytes, bytearray]

STT_MODEL = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")
DEFAULT_MIN_SIMILARITY = 0.70
SAMPLE_RATE = 16_000  # resemblyzer / the embedding model operate at 16 kHz mono.

# One shared VoiceEncoder, loaded lazily on first use (the model weights are bundled
# with resemblyzer, so no download is needed).
_ENCODER = None


def _min_similarity() -> float:
    raw = os.environ.get("SPEAKER_MATCH_THRESHOLD")
    return float(raw) if raw else DEFAULT_MIN_SIMILARITY


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        from resemblyzer import VoiceEncoder

        _ENCODER = VoiceEncoder(verbose=False)
    return _ENCODER


def _load_wav_16k(audio: AudioInput):
    """Load audio (path or raw bytes) as a float32 mono waveform at 16 kHz."""
    import librosa

    if isinstance(audio, (bytes, bytearray)):
        # librosa/soundfile can't read mp3 from a buffer; round-trip via a temp file.
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(bytes(audio))
            tmp_path = tmp.name
        try:
            wav, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        finally:
            os.unlink(tmp_path)
        return wav

    wav, _ = librosa.load(str(audio), sr=SAMPLE_RATE, mono=True)
    return wav


def _embed(wav):
    """Embed a 16 kHz mono waveform into a unit-norm speaker d-vector, or None."""
    from resemblyzer import preprocess_wav

    try:
        processed = preprocess_wav(wav, source_sr=SAMPLE_RATE)
    except Exception:  # noqa: BLE001 - VAD can choke on very short/empty slices
        processed = wav
    if processed is None or len(processed) == 0:
        return None
    return _encoder().embed_utterance(processed)


def transcribe_with_speakers(
    audio: AudioInput,
    *,
    num_speakers: int | None = None,
    language_code: str | None = None,
    client=None,
) -> dict:
    """Transcribe `audio` with anonymous speaker diarization + word timestamps.

    Returns {"text", "words": [{"text","start","end","speaker_id"}], "speakers": [...]}.
    Speaker ids are anonymous ('speaker_0', 'speaker_1', ...).
    """
    client = client or make_elevenlabs_client()
    model_id = os.environ.get("ELEVENLABS_STT_MODEL", STT_MODEL)

    import io

    if isinstance(audio, (bytes, bytearray)):
        file_obj: object = io.BytesIO(bytes(audio))
    else:
        file_obj = io.BytesIO(pathlib.Path(audio).read_bytes())

    result = client.speech_to_text.convert(
        file=file_obj,
        model_id=model_id,
        diarize=True,
        timestamps_granularity="word",
        num_speakers=num_speakers,
        language_code=language_code,
    )

    words = []
    for w in getattr(result, "words", None) or []:
        speaker = getattr(w, "speaker_id", None)
        start = getattr(w, "start", None)
        end = getattr(w, "end", None)
        if getattr(w, "type", "word") != "word" or speaker is None or start is None:
            continue
        words.append(
            {"text": getattr(w, "text", ""), "start": float(start),
             "end": float(end if end is not None else start), "speaker_id": speaker}
        )

    # Speakers in order of first appearance.
    speakers: list[str] = []
    for w in words:
        if w["speaker_id"] not in speakers:
            speakers.append(w["speaker_id"])

    return {"text": getattr(result, "text", "") or "", "words": words, "speakers": speakers}


def _merge_segments(words: list[dict], label_for: dict) -> list[dict]:
    """Merge consecutive same-speaker words into turns with the relabelled speaker."""
    segments: list[dict] = []
    for w in words:
        label = label_for.get(w["speaker_id"], w["speaker_id"])
        if segments and segments[-1]["speaker_label"] == label:
            segments[-1]["text"] += (" " + w["text"]).rstrip()
            segments[-1]["end"] = w["end"]
        else:
            segments.append(
                {"speaker_label": label, "start": w["start"], "end": w["end"],
                 "text": w["text"]}
            )
    for seg in segments:
        seg["text"] = seg["text"].strip()
    return segments


def identify_officer(
    conversation_audio: AudioInput,
    officer_reference_audio: AudioInput,
    *,
    num_speakers: int | None = None,
    min_similarity: float | None = None,
    language_code: str | None = None,
    client=None,
) -> dict:
    """Transcribe a conversation and label the officer's turns vs everyone else.

    `officer_reference_audio` is a short clip of the officer's voice (enrollment).
    The diarized speaker whose voice best matches it is relabelled `officer`; the
    rest become `person1`, `person2`, ... by order of first appearance.

    Returns:
      {
        "officer_speaker_id": str | None,   # anonymous id matched to the officer
        "similarities": {speaker_id: float},
        "labels": {speaker_id: "officer" | "personN"},
        "segments": [{speaker_label, start, end, text}],
        "transcript": "[officer] ...\n[person1] ...",
        "matched": bool,                    # best similarity >= threshold
      }
    """
    import numpy as np

    threshold = _min_similarity() if min_similarity is None else min_similarity

    diar = transcribe_with_speakers(
        conversation_audio, num_speakers=num_speakers,
        language_code=language_code, client=client,
    )
    words, speakers = diar["words"], diar["speakers"]
    if not words:
        return {"officer_speaker_id": None, "similarities": {}, "labels": {},
                "segments": [], "transcript": diar["text"], "matched": False}

    conv_wav = _load_wav_16k(conversation_audio)
    ref_emb = _embed(_load_wav_16k(officer_reference_audio))
    if ref_emb is None:
        raise ValueError(
            "Could not embed the officer reference clip — it may be too short or silent."
        )

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
        emb = _embed(np.concatenate(chunks))
        # resemblyzer embeddings are unit-norm, so dot product == cosine similarity.
        similarities[spk] = float(np.dot(ref_emb, emb)) if emb is not None else -1.0

    officer_spk = max(similarities, key=similarities.get) if similarities else None
    matched = officer_spk is not None and similarities[officer_spk] >= threshold

    # Relabel: best match -> officer, remaining -> person1, person2, ... (first-appearance order).
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


def describe() -> str:
    has_key = bool(os.environ.get("ELEVENLABS_API_KEY"))
    return (
        f"speaker-id stt={os.environ.get('ELEVENLABS_STT_MODEL', STT_MODEL)} "
        f"embed=resemblyzer threshold={_min_similarity():.2f} "
        f"api_key={'set' if has_key else 'MISSING'}"
    )


def main() -> None:
    import argparse
    import json

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Identify the officer's voice among multiple speakers"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("identify", help="label officer vs person1/person2/... in a conversation")
    p.add_argument("--conversation", required=True, help="multi-speaker audio file")
    p.add_argument("--officer", required=True, help="reference clip of the officer's voice")
    p.add_argument("--num-speakers", type=int, default=None, help="hint: number of speakers")
    p.add_argument("--language-code", default=None, help="ISO language hint, e.g. eng")
    p.add_argument("--out", default=None, help="optional path to dump the JSON result")

    args = parser.parse_args()
    print(describe())

    result = identify_officer(
        args.conversation, args.officer,
        num_speakers=args.num_speakers, language_code=args.language_code,
    )

    sims = ", ".join(f"{k}={v:.3f}" for k, v in result["similarities"].items())
    print(f"officer = {result['officer_speaker_id']} "
          f"(matched={result['matched']}); similarities: {sims}")
    print("-" * 60)
    print(result["transcript"])

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote structured result -> {args.out}")


if __name__ == "__main__":
    main()
