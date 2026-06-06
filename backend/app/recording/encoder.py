"""Encode a recorded session's frames + audio into an MP4 with ffmpeg.

Frames are captured at a variable rate, so we drive ffmpeg with a concat demuxer that
carries each frame's real duration. The audio track (WebM) is muxed in when present.
The encoded MP4 is uploaded to object storage and cached via the manifest's video_key.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from app.models.session import SessionManifest
from app.recording.recorder import session_prefix
from app.storage import ObjectStore

_FALLBACK_FRAME_SECONDS = 0.5  # used for the last frame / when offsets are missing


class EncodingError(RuntimeError):
    pass


def _frame_durations(manifest: SessionManifest) -> list[float]:
    """Per-frame display durations derived from capture offsets."""
    offsets = manifest.frame_offsets
    durations: list[float] = []
    for index in range(len(offsets)):
        if index + 1 < len(offsets):
            durations.append(max(0.05, offsets[index + 1] - offsets[index]))
        else:
            durations.append(_FALLBACK_FRAME_SECONDS)
    return durations


async def encode_session_video(store: ObjectStore, manifest: SessionManifest) -> str:
    """Encode the session to MP4, upload it, and return the object key."""
    if manifest.frame_count == 0:
        raise EncodingError("session has no frames to encode")

    durations = _frame_durations(manifest)

    with TemporaryDirectory() as workdir:
        work = Path(workdir)
        frames_dir = work / "frames"
        frames_dir.mkdir()

        # Download frames and build the ffmpeg concat script.
        concat_lines: list[str] = []
        for index, frame_key in enumerate(manifest.frame_keys):
            frame_bytes, _ = await store.get(frame_key)
            frame_path = frames_dir / f"{index:06d}.jpg"
            frame_path.write_bytes(frame_bytes)
            concat_lines.append(f"file '{frame_path.as_posix()}'")
            concat_lines.append(f"duration {durations[index]:.3f}")
        # The concat demuxer needs the final file repeated so its duration is honored.
        last_frame = frames_dir / f"{manifest.frame_count - 1:06d}.jpg"
        concat_lines.append(f"file '{last_frame.as_posix()}'")
        concat_path = work / "frames.txt"
        concat_path.write_text("\n".join(concat_lines))

        audio_path: Path | None = None
        if manifest.audio_key:
            audio_bytes, _ = await store.get(manifest.audio_key)
            audio_path = work / "audio.webm"
            audio_path.write_bytes(audio_bytes)

        output_path = work / "session.mp4"
        try:
            await _run_ffmpeg(concat_path, audio_path, output_path)
        except EncodingError:
            # A corrupt or unreadable audio track shouldn't block the video — retry silent.
            if audio_path is None:
                raise
            await _run_ffmpeg(concat_path, None, output_path)

        video_key = f"{session_prefix(manifest.session_id)}/session.mp4"
        await store.put(video_key, output_path.read_bytes(), "video/mp4")
        return video_key


async def _run_ffmpeg(concat_path: Path, audio_path: Path | None, output_path: Path) -> None:
    command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path)]
    if audio_path is not None:
        command += ["-i", str(audio_path)]
    command += [
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # H.264 needs even dimensions
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "veryfast",
    ]
    if audio_path is not None:
        command += ["-c:a", "aac", "-shortest"]
    command.append(str(output_path))

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise EncodingError(stderr.decode(errors="replace")[-2000:])
