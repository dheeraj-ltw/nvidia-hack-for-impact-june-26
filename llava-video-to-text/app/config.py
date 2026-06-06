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
"""Environment-driven configuration for the orchestrator.

Field names map case-insensitively to environment variables (e.g. ``vlm_model`` <- ``VLM_MODEL``),
which are supplied by ``compose.yml`` / ``.env`` and ``entrypoint.sh``. Unknown env vars
(``GPU_MEM_UTIL``, ``MAX_MODEL_LEN``, ... — consumed only by the vLLM launcher) are ignored.
"""

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

DEFAULT_PROMPT = (
    "You are an expert at video understanding. The images provided are frames sampled from a "
    "video segment in chronological order. Describe, in as much detail as possible, the scene, "
    "the people (attire, actions), objects (make, model, color), and any notable events. "
    "Respond with a single concise paragraph."
)


class Settings(BaseSettings):
    """Runtime settings for the video-to-text orchestrator."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Connection to the co-located vLLM OpenAI-compatible server (set by entrypoint.sh).
    vlm_model: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    vlm_base_url: str = "http://localhost:8000"

    # Orchestrator HTTP port.
    llava_port: int = 8900

    # Frame sampling / chunking defaults (overridable per request).
    sample_fps: float = 1.0
    chunk_seconds: float = 10.0
    max_frames: int = 8

    # VLM generation params.
    max_tokens: int = 512
    temperature: float = 0.0
    request_timeout: float = 120.0
    caption_concurrency: int = 2

    prompt: str = DEFAULT_PROMPT


settings = Settings()
