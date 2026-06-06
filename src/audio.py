"""ElevenLabs voice pipeline: text-to-speech and speech-to-text.

Standalone, frontend-ready helpers built on the official `elevenlabs` SDK:

  - `text_to_speech(text, ...)` -> audio bytes (optionally written to a file)
  - `speech_to_text(audio, ...)` -> transcript string

Config comes from the environment (loaded from `.env` by the caller, same as
`llm.py`):

  ELEVENLABS_API_KEY   (required)
  ELEVENLABS_VOICE_ID  (optional; defaults to a public ElevenLabs voice)
  ELEVENLABS_TTS_MODEL (optional; defaults to eleven_multilingual_v2)
  ELEVENLABS_STT_MODEL (optional; defaults to scribe_v1)

There's also a small CLI for manual round-trip testing:

  python src/audio.py tts "some text" --out data/audio/sample.mp3
  python src/audio.py stt data/audio/sample.mp3
"""
from __future__ import annotations

import io
import os
import pathlib
from typing import Union

# Mirror the path-constant style in common.py without importing it (keeps this
# module standalone for frontend reuse).
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AUDIO_DIR = DATA / "audio"

# A public ElevenLabs voice ("Rachel"), used when ELEVENLABS_VOICE_ID is unset.
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_TTS_MODEL = "eleven_multilingual_v2"
DEFAULT_STT_MODEL = "scribe_v1"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

AudioInput = Union[str, "os.PathLike[str]", bytes, bytearray]


def make_elevenlabs_client():
    """Construct an ElevenLabs client from ELEVENLABS_API_KEY.

    Lazily imports the SDK (like llm.py does for openai/anthropic) so importing
    this module is cheap and doesn't require the dependency until first use.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit(
            "No ElevenLabs API key configured. Set ELEVENLABS_API_KEY in .env "
            "(see .env.example)."
        )
    from elevenlabs.client import ElevenLabs

    return ElevenLabs(api_key=api_key)


def text_to_speech(
    text: str,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    out_path: str | os.PathLike[str] | None = None,
    client=None,
) -> bytes:
    """Synthesize `text` to speech and return the raw audio bytes.

    If `out_path` is given the bytes are also written there (parent dirs are
    created). Defaults for voice/model come from the environment, falling back
    to documented public defaults.
    """
    if not text or not text.strip():
        raise ValueError("text_to_speech: `text` must be a non-empty string")

    client = client or make_elevenlabs_client()
    voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE_ID
    model_id = model_id or os.environ.get("ELEVENLABS_TTS_MODEL", DEFAULT_TTS_MODEL)

    # convert() returns an iterator of byte chunks; join into a single blob.
    stream = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format=output_format,
    )
    audio = stream if isinstance(stream, (bytes, bytearray)) else b"".join(stream)
    audio = bytes(audio)

    if out_path is not None:
        path = pathlib.Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)

    return audio


def speech_to_text(
    audio: AudioInput,
    *,
    model_id: str | None = None,
    language_code: str | None = None,
    client=None,
) -> str:
    """Transcribe audio to text and return the transcript string.

    `audio` may be a file path (str/PathLike) or raw bytes — the latter lets a
    frontend pass an upload buffer straight through.
    """
    client = client or make_elevenlabs_client()
    model_id = model_id or os.environ.get("ELEVENLABS_STT_MODEL", DEFAULT_STT_MODEL)

    if isinstance(audio, (bytes, bytearray)):
        file_obj: object = io.BytesIO(bytes(audio))
    else:
        file_obj = io.BytesIO(pathlib.Path(audio).read_bytes())

    result = client.speech_to_text.convert(
        file=file_obj,
        model_id=model_id,
        language_code=language_code,
    )
    return getattr(result, "text", "") or ""


def clone_voice(
    name: str,
    files: list[str | os.PathLike[str]],
    *,
    description: str | None = None,
    remove_background_noise: bool = True,
    client=None,
) -> dict:
    """Create an Instant Voice Clone (IVC) from sample recordings of a person.

    `files` are paths to clean, single-speaker clips (mp3/wav/m4a/...); ~1-3 min
    total is recommended. Returns {"voice_id", "name"} — set the returned
    voice_id as ELEVENLABS_VOICE_ID (or pass it to text_to_speech(voice_id=...)).

    NB: only clone a voice you have permission to use (ElevenLabs ToS).
    """
    if not files:
        raise ValueError("clone_voice: provide at least one sample audio file")

    client = client or make_elevenlabs_client()
    handles = [open(pathlib.Path(f), "rb") for f in files]
    try:
        voice = client.voices.ivc.create(
            name=name,
            files=handles,
            description=description,
            remove_background_noise=remove_background_noise,
        )
    finally:
        for h in handles:
            h.close()

    voice_id = getattr(voice, "voice_id", None) or getattr(voice, "voiceId", None)
    return {"voice_id": voice_id, "name": getattr(voice, "name", name)}


def describe() -> str:
    """One-line summary of the active config, for logging."""
    has_key = bool(os.environ.get("ELEVENLABS_API_KEY"))
    voice = os.environ.get("ELEVENLABS_VOICE_ID") or f"{DEFAULT_VOICE_ID} (default)"
    tts = os.environ.get("ELEVENLABS_TTS_MODEL", DEFAULT_TTS_MODEL)
    stt = os.environ.get("ELEVENLABS_STT_MODEL", DEFAULT_STT_MODEL)
    key_state = "set" if has_key else "MISSING"
    return f"elevenlabs api_key={key_state} voice={voice} tts={tts} stt={stt}"


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="ElevenLabs TTS/STT test harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_tts = sub.add_parser("tts", help="text -> speech")
    p_tts.add_argument("text", help="text to synthesize")
    p_tts.add_argument(
        "--out",
        default=str(AUDIO_DIR / "sample.mp3"),
        help="output audio path (default: data/audio/sample.mp3)",
    )
    p_tts.add_argument("--voice-id", default=None, help="override ELEVENLABS_VOICE_ID")
    p_tts.add_argument("--model-id", default=None, help="override ELEVENLABS_TTS_MODEL")

    p_stt = sub.add_parser("stt", help="speech -> text")
    p_stt.add_argument("audio", help="path to an audio file to transcribe")
    p_stt.add_argument("--model-id", default=None, help="override ELEVENLABS_STT_MODEL")
    p_stt.add_argument("--language-code", default=None, help="ISO language hint, e.g. eng")

    p_clone = sub.add_parser("clone", help="instant-voice-clone from sample recordings")
    p_clone.add_argument("--name", required=True, help="name for the cloned voice")
    p_clone.add_argument("files", nargs="+", help="sample audio files (1-3 min total)")
    p_clone.add_argument("--description", default=None, help="optional voice description")
    p_clone.add_argument("--keep-noise", action="store_true",
                         help="do NOT remove background noise from samples")

    args = parser.parse_args()
    print(describe())

    if args.command == "tts":
        text_to_speech(
            args.text,
            voice_id=args.voice_id,
            model_id=args.model_id,
            out_path=args.out,
        )
        print(f"wrote audio -> {args.out}")
    elif args.command == "stt":
        transcript = speech_to_text(
            args.audio,
            model_id=args.model_id,
            language_code=args.language_code,
        )
        print(transcript)
    elif args.command == "clone":
        result = clone_voice(
            args.name,
            args.files,
            description=args.description,
            remove_background_noise=not args.keep_noise,
        )
        print(f"cloned voice '{result['name']}' -> voice_id={result['voice_id']}")
        print(f"set ELEVENLABS_VOICE_ID={result['voice_id']} to use it as the default voice")


if __name__ == "__main__":
    main()
