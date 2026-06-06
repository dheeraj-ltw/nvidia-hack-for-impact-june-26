# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenCV frame sampling helpers.

``sample_file`` ports the seek-and-encode approach from
``agent/src/vss_agents/utils/frame_select.py`` and additionally returns each frame's timestamp.
"""

import base64
import logging
import math
from typing import Any

import cv2

logger = logging.getLogger(__name__)

# Guard against unbounded reads when a container reports no frame count (e.g. some codecs).
_MAX_FRAMES_FILE = 100_000


def encode_frame(frame: Any) -> str:
    """Encode a BGR OpenCV frame as a base64 JPEG string."""
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def sample_file(video_path: str, sample_fps: float) -> list[tuple[float, str]]:
    """Sample a video file at ``sample_fps`` frames per second.

    Returns a list of ``(timestamp_seconds, base64_jpeg)`` tuples in chronological order.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            logger.warning("Video %s reports invalid FPS; defaulting to 30.0", video_path)
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, math.floor(fps / max(sample_fps, 1e-6)))

        frames: list[tuple[float, str]] = []
        idx = 0
        while total_frames <= 0 or idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((idx / fps, encode_frame(frame)))
            idx += step
            if len(frames) >= _MAX_FRAMES_FILE:
                logger.warning("Reached frame cap (%d) for %s", _MAX_FRAMES_FILE, video_path)
                break
        return frames
    finally:
        cap.release()


def subsample(items: list[tuple[float, str]], max_n: int) -> list[tuple[float, str]]:
    """Evenly downsample a list of timestamped frames to at most ``max_n`` items."""
    if max_n <= 0 or len(items) <= max_n:
        return items
    stride = len(items) / max_n
    return [items[min(len(items) - 1, int(i * stride))] for i in range(max_n)]
