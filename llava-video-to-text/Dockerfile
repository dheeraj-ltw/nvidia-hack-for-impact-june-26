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

# Multi-arch base; the same vLLM image the repo runs on the DGX-SPARK hardware profile
# (see deployments/nim/qwen3-vl-8b-instruct/compose.yml). Ships vLLM + CUDA for linux/arm64.
FROM nvcr.io/nvidia/vllm:25.12.post1-py3

ENV DEBIAN_FRONTEND=noninteractive

# ffmpeg + libGL: OpenCV runtime and RTSP/codec decode. curl: healthcheck + readiness probe.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Orchestrator-only deps. fastapi/uvicorn/httpx/pydantic ship with the vLLM base image, so we
# pin nothing here to avoid clobbering vLLM's resolved constraints.
RUN pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        httpx \
        opencv-python-headless \
        pydantic-settings \
        sse-starlette

WORKDIR /app
COPY app/ /app/app/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8900
ENTRYPOINT ["/app/entrypoint.sh"]
