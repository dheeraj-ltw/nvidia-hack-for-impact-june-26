<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# LLaVA Video-to-Text (DGX Spark)

A self-contained, single-container service that turns **streamed video into text** using
**LLaVA** (Large Language and Vision Assistant), built to run on an **NVIDIA DGX Spark**
(ARM64 / GB10 Grace-Blackwell). It is fully standalone — independent of the rest of this repo.

## How it works

One container runs two cooperating processes (launched by [`docker/entrypoint.sh`](docker/entrypoint.sh)):

1. **vLLM** serves LLaVA on an internal OpenAI-compatible API (`localhost:8000`), using the same
   `nvcr.io/nvidia/vllm` image this repo already runs on the DGX-SPARK profile.
2. A **FastAPI orchestrator** (port `8900`) accepts video, samples frames with OpenCV, sends them
   to LLaVA as base64 images, and returns text.

```
                     ┌──────────────── container ────────────────┐
 RTSP / upload  ───► │  FastAPI orchestrator (:8900)              │
                     │     │ sample frames (OpenCV)               │
                     │     ▼                                      │
                     │  vLLM + LLaVA  (:8000, OpenAI /v1)  ──► text│
                     └────────────────────────────────────────────┘
```

LLaVA-1.6 is an image model, so video-to-text works by sampling frames, captioning fixed-length
chunks, and (for uploads) aggregating chunk captions into a summary — the same approach used by
the agent's `video_caption` tool.

## Quick start (DGX Spark)

Requires Docker + the NVIDIA Container Toolkit on the host.

```bash
cd llava-video-to-text
cp .env.example .env            # adjust model / GPU / sampling if needed
docker compose up --build       # first run downloads the model (several minutes)
```

Wait for the container to become healthy, then:

```bash
curl localhost:8900/health      # {"status":"ok","vlm_ready":true}
curl localhost:8900/v1/models   # lists the served LLaVA model
```

## API

| Method & path | Purpose |
|---|---|
| `GET /health` | Ready only once vLLM has loaded the model |
| `GET /v1/models` | Passthrough — reports the served LLaVA model |
| `PUT /v1/video/{name}` | Upload `video/mp4` or `video/x-matroska`; returns captions + summary (add `?stream=true` for SSE) |
| `POST /v1/streams/add` | Register an RTSP stream for continuous captioning |
| `GET /v1/streams/{id}/captions` | SSE stream of captions for an active RTSP stream |
| `DELETE /v1/streams/delete/{id}` | Stop an RTSP stream |
| `POST /v1/chat/completions` | Passthrough — use as a plain OpenAI-compatible LLaVA endpoint |

### Upload a clip

```bash
curl -X PUT "localhost:8900/v1/video/clip1" \
  -H "Content-Type: video/mp4" \
  --data-binary @sample.mp4
# {"captions":[{"start":0.0,"end":9.0,"text":"..."}], "summary":"..."}
```

### Stream an RTSP camera

```bash
# 1) register the stream
curl -X POST localhost:8900/v1/streams/add \
  -H "Content-Type: application/json" \
  -d '{"rtsp_url":"rtsp://CAMERA/stream","name":"cam1","fps":1,"chunk_seconds":10}'
# {"stream_id":"<id>","name":"cam1"}

# 2) read live captions (Server-Sent Events)
curl -N localhost:8900/v1/streams/<id>/captions

# 3) stop
curl -X DELETE localhost:8900/v1/streams/delete/<id>
```

No physical camera? Loop a file into a local RTSP server (e.g. [MediaMTX](https://github.com/bluenviron/mediamtx)):

```bash
ffmpeg -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp rtsp://localhost:8554/test
```

## Configuration

All settings come from environment variables (see [`.env.example`](.env.example)). Notable ones:

| Variable | Default | Used by | Description |
|---|---|---|---|
| `VLM_MODEL` | `llava-hf/llava-v1.6-mistral-7b-hf` | entrypoint | HF model id (swap to any vLLM-supported LLaVA) |
| `VLM_DEVICE_ID` | `0` | compose | GPU to reserve |
| `GPU_MEM_UTIL` | `0.85` | entrypoint | vLLM `--gpu-memory-utilization` |
| `MAX_MODEL_LEN` | `8192` | entrypoint | vLLM `--max-model-len` |
| `MAX_NUM_SEQS` | `4` | entrypoint | vLLM `--max-num-seqs` (small for DGX Spark) |
| `SAMPLE_FPS` | `1` | orchestrator | Frames sampled per second of video |
| `CHUNK_SECONDS` | `10` | orchestrator | Seconds of video per caption |
| `MAX_FRAMES` | `8` | both | Max frames per VLM request (also vLLM `--limit-mm-per-prompt image=N`) |
| `HF_TOKEN` | _(empty)_ | compose | Only needed for gated models |

DGX-Spark-friendly vLLM flags (`--enforce-eager`, small `--max-num-seqs`, modest `--max-model-len`)
mirror `deployments/nim/cosmos-reason2-8b/hw-DGX-SPARK.env`.

> **Multi-image note:** `MAX_FRAMES` frames are sent per VLM request, and vLLM is launched with
> `--limit-mm-per-prompt image=$MAX_FRAMES`. If the chosen LLaVA variant only accepts a single
> image per prompt, set `MAX_FRAMES=1` — the orchestrator then captions one representative frame
> per `CHUNK_SECONDS` window. A larger value gives the model more temporal context per caption.

## Local development (no GPU)

You can run the orchestrator alone on a laptop and point it at any external OpenAI-compatible VLM:

```bash
uv sync                                            # or: pip install -e .
VLM_BASE_URL=https://your-vlm-host/v1-root \
VLM_MODEL=<served-model-name> \
  uvicorn app.main:app --reload --port 8900
```

This exercises frame sampling and all endpoints without loading LLaVA locally. `VLM_BASE_URL`
should point at the root that exposes `/v1/chat/completions` and `/health`.
