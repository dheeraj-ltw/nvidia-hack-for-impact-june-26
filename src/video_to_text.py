"""Video-to-text via Nebius-hosted Qwen2.5-VL (OpenAI-compatible API).

Samples frames from a video with OpenCV, sends them as base64 images to the
Nebius Token Factory endpoint, and returns a text description — no local GPU /
vLLM needed (replaces the self-hosted llava-video-to-text Docker path).

Config (loaded from .env by the CLI):
    NEBIUS_API_KEY    (required)
    NEBIUS_BASE_URL   (default https://api.tokenfactory.nebius.com/v1/)
    NEBIUS_VLM_MODEL  (default Qwen/Qwen2.5-VL-72B-Instruct)

CLI:
    python src/video_to_text.py path/to/clip.mp4 --max-frames 8 --fps 1
"""
from __future__ import annotations

import base64
import math
import os
import pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
DEFAULT_PROMPT = (
    "You are an expert at video understanding. The images that follow are frames sampled from a "
    "single video in chronological order; each is labelled with its timestamp in seconds. "
    "Treat them as one continuous clip, not separate pictures. Describe what happens over the "
    "course of the video from beginning to end: how the scene, the people (attire, actions, "
    "expressions), and objects (make, model, colour) change or stay the same across the frames, "
    "and any notable events or movement. Write a single flowing paragraph that follows the clip "
    "chronologically rather than cataloguing a static snapshot."
)
_MAX_FRAMES_CAP = 100_000  # guard against codecs that report no frame count


def make_nebius_client(*, base_url: str | None = None, api_key: str | None = None):
    """Construct an OpenAI client pointed at the Nebius Token Factory endpoint."""
    api_key = api_key or os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise SystemExit(
            "No Nebius API key configured. Set NEBIUS_API_KEY in .env (see .env.example)."
        )
    from openai import OpenAI

    return OpenAI(
        base_url=base_url or os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
    )


def _encode_frame(frame: Any) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def sample_frames(
    video_path: str | os.PathLike[str], *, sample_fps: float = 1.0, max_frames: int = 8
) -> list[tuple[float, str]]:
    """Sample frames at `sample_fps`, evenly downsampled to at most `max_frames`.

    Returns [(timestamp_seconds, base64_jpeg), ...] in chronological order.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, math.floor(fps / max(sample_fps, 1e-6)))

        frames: list[tuple[float, str]] = []
        idx = 0
        while total <= 0 or idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((idx / fps, _encode_frame(frame)))
            idx += step
            if len(frames) >= _MAX_FRAMES_CAP:
                break
    finally:
        cap.release()

    # Evenly downsample to at most max_frames.
    if max_frames > 0 and len(frames) > max_frames:
        stride = len(frames) / max_frames
        frames = [frames[min(len(frames) - 1, int(i * stride))] for i in range(max_frames)]
    return frames


def video_to_text(
    video: str | os.PathLike[str],
    *,
    prompt: str | None = None,
    model: str | None = None,
    sample_fps: float = 1.0,
    max_frames: int = 8,
    max_tokens: int = 512,
    temperature: float = 0.0,
    client=None,
) -> str:
    """Describe a video by sampling frames and sending them to Nebius Qwen2.5-VL.

    Returns the model's text description. `video` is a path to a video file.
    """
    client = client or make_nebius_client()
    model = model or os.environ.get("NEBIUS_VLM_MODEL", DEFAULT_MODEL)
    prompt = prompt or DEFAULT_PROMPT

    frames = sample_frames(video, sample_fps=sample_fps, max_frames=max_frames)
    if not frames:
        raise ValueError(f"No frames could be sampled from {video}")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for ts, b64 in frames:
        content.append({"type": "text", "text": f"[t={ts:.1f}s]"})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def describe() -> str:
    has_key = bool(os.environ.get("NEBIUS_API_KEY"))
    base = os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("NEBIUS_VLM_MODEL", DEFAULT_MODEL)
    return f"nebius video-to-text base={base} model={model} api_key={'set' if has_key else 'MISSING'}"


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Describe a video with Nebius Qwen2.5-VL")
    parser.add_argument("video", help="path to a video file (mp4/mkv/mov/...)")
    parser.add_argument("--fps", type=float, default=1.0, help="frames sampled per second (default 1)")
    parser.add_argument("--max-frames", type=int, default=8, help="max frames sent to the model (default 8)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model", default=None, help="override NEBIUS_VLM_MODEL")
    parser.add_argument("--prompt", default=None, help="override the default video-understanding prompt")
    args = parser.parse_args()

    print(describe())
    text = video_to_text(
        args.video,
        prompt=args.prompt,
        model=args.model,
        sample_fps=args.fps,
        max_frames=args.max_frames,
        max_tokens=args.max_tokens,
    )
    print("-" * 60)
    print(text)


if __name__ == "__main__":
    main()
