#!/usr/bin/env bash
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
#
# Single-container launcher: start vLLM (LLaVA, OpenAI-compatible) in the background, wait for it
# to become healthy, then start the FastAPI orchestrator. DGX-Spark-friendly vLLM defaults mirror
# deployments/nim/cosmos-reason2-8b/hw-DGX-SPARK.env (enforce-eager, small max-num-seqs, modest
# max-model-len).
set -euo pipefail

VLM_MODEL="${VLM_MODEL:-llava-hf/llava-v1.6-mistral-7b-hf}"
VLM_PORT_INTERNAL="${VLM_PORT_INTERNAL:-8000}"
LLAVA_PORT="${LLAVA_PORT:-8900}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_FRAMES="${MAX_FRAMES:-8}"

echo "[entrypoint] starting vLLM for '${VLM_MODEL}' on :${VLM_PORT_INTERNAL}"
python3 -m vllm.entrypoints.openai.api_server \
    --model "${VLM_MODEL}" \
    --trust-remote-code \
    --port "${VLM_PORT_INTERNAL}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enforce-eager \
    --limit-mm-per-prompt "{\"image\": ${MAX_FRAMES}}" &
VLLM_PID=$!

UVICORN_PID=""
cleanup() {
    echo "[entrypoint] shutting down"
    [ -n "${UVICORN_PID}" ] && kill "${UVICORN_PID}" 2>/dev/null || true
    kill "${VLLM_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup SIGTERM SIGINT EXIT

echo "[entrypoint] waiting for vLLM health on :${VLM_PORT_INTERNAL} (model load can take several minutes)..."
until curl -fs "http://localhost:${VLM_PORT_INTERNAL}/health" >/dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[entrypoint] vLLM exited before becoming ready" >&2
        exit 1
    fi
    sleep 5
done

echo "[entrypoint] vLLM ready; starting orchestrator on :${LLAVA_PORT}"
export VLM_BASE_URL="http://localhost:${VLM_PORT_INTERNAL}"
uvicorn app.main:app --host 0.0.0.0 --port "${LLAVA_PORT}" &
UVICORN_PID=$!

# Exit (and trigger cleanup) as soon as either process stops.
wait -n "${VLLM_PID}" "${UVICORN_PID}"
