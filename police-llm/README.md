# police-llm

An **OpenAI-compatible inference server** that loads a GGUF model with
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) and exposes it
over a FastAPI HTTP API. Any service that can talk to the OpenAI API can use
this by simply pointing its `base_url` at this server — no code changes beyond
the URL (and an optional key).

Designed to serve the fine-tuned PoliceAI model (`Llama-3.3-Nemotron-Super-49B`)
on an **NVIDIA DGX Spark (GB10, sm_121)**, but works with any GGUF on any GPU
or CPU.

## Endpoints

| Method | Path                   | Notes                                  |
| ------ | ---------------------- | -------------------------------------- |
| GET    | `/health`              | Liveness + whether the model is loaded |
| GET    | `/v1/models`           | Lists the served model                 |
| POST   | `/v1/chat/completions` | Chat API, streaming + non-streaming    |
| POST   | `/v1/completions`      | Text completion, streaming + non-streaming |

## Prerequisites

1. A GGUF model file (produced by the fine-tuning pipeline in `../models/fine-tune`):

   ```bash
   # from models/fine-tune, after training:
   ./gguf.sh   # merges the LoRA -> bf16 -> GGUF (Q4_K_M)
   ```

2. Put the `.gguf` file in `police-llm/models/`:

   ```bash
   mkdir -p police-llm/models
   cp /path/to/policeai-super-49b-Q4_K_M.gguf police-llm/models/
   ```

3. On DGX Spark, the NVIDIA Container Toolkit must be installed so Docker can
   access the GPU.

## Configuration

Copy the sample env file and edit paths/ports as needed:

```bash
cd police-llm
cp .env.example .env   # skip if .env already exists
```

Key variables (see `.env.example` for the full list):

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `LLM_PORT` | `8001` | Host port (maps to container `PORT`) |
| `MODEL_PATH` | `/models/...` | GGUF path inside the container (`./models/` → `/models`) |
| `MODEL_NAME` | `policeai-super-49b` | Name on `/v1/models` |
| `N_GPU_LAYERS` | `-1` | GPU offload (`-1` = all layers) |
| `N_CTX` | `8192` | Context window |
| `API_KEY` | _(empty)_ | Optional Bearer auth |
| `DEFAULT_SYSTEM_PROMPT` | PoliceAI + reasoning | Injected when no system message |

Multi-line prompts in `.env` use `\n` escapes (decoded at runtime).

## Run with docker compose (recommended)

```bash
cd police-llm
docker compose up --build
```

The server is then available at `http://localhost:8001` (host port `8001` →
container `8000`, chosen to avoid clashing with the main API on `8000`).

> The first build compiles `llama-cpp-python` against CUDA and can take several
> minutes. Model load on first request can also take a while for a 49B model.

## Run with plain docker

```bash
cd police-llm
docker build -t police-llm .

docker run --gpus all -p 8001:8000 \
  -v "$(pwd)/models:/models:ro" \
  -e MODEL_PATH=/models/policeai-super-49b-Q4_K_M.gguf \
  -e MODEL_NAME=policeai-super-49b \
  police-llm
```

## Usage

### curl

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "policeai-super-49b",
    "messages": [
      {"role": "system", "content": "detailed thinking on"},
      {"role": "user", "content": "What are my grounds to search for a weapon?"}
    ],
    "temperature": 0.6,
    "max_tokens": 512
  }'
```

### Python (official `openai` client)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed",  # or your API_KEY if you set one
)

resp = client.chat.completions.create(
    model="policeai-super-49b",
    messages=[
        {"role": "system", "content": "detailed thinking on"},
        {"role": "user", "content": "Do I have grounds to search this person?"},
    ],
    temperature=0.6,
)
print(resp.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="policeai-super-49b",
    messages=[{"role": "user", "content": "Summarise PACE s.1 grounds."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

### From another container in the same compose network

Use the service name as the host: `http://police-llm:8000/v1`.

## Troubleshooting

### Docker build: `libcuda.so.1 not found` / `undefined reference to cuMemCreate`

The NVIDIA **driver** is not present during `docker build` (only at runtime with
`--gpus all`). The Dockerfile works around this by:

1. Setting `GGML_CUDA_NO_VMM=on` so ggml-cuda does not link `libcuda.so` (VMM
   driver APIs like `cuMemCreate`).
2. Disabling llama.cpp CLI/tool targets (`LLAMA_BUILD_TOOLS=OFF`) — only the
   Python bindings are needed.
3. Symlinking CUDA driver **stubs** (`libcuda.so.1` → `libcuda.so`) as a fallback.

If you still hit linker errors, confirm you are using the `nvidia/cuda:*-devel`
base image (not `runtime`) and rebuild:

```bash
docker compose build --no-cache
```

At **runtime**, the container must have GPU access (`deploy.resources.reservations
.devices` in compose, or `docker run --gpus all`).

## Notes & limitations

- **Concurrency:** `llama.cpp` uses a single context that is not re-entrant, so
  requests are serialised behind a lock (one in-flight generation at a time).
  For higher throughput, run multiple replicas behind a load balancer, or
  switch to a batching server (e.g. vLLM) — out of scope here.
- **GPU arch:** the image builds for `sm_121` (GB10) by default. For other GPUs,
  rebuild with `--build-arg CUDA_ARCH=<cc>` (e.g. `89` for RTX 4090). For a
  CPU-only image, build with `--build-arg LLAMA_CMAKE_ARGS=""`.
- **Chat template:** the GGUF carries the model's chat template, so you don't
  normally need `CHAT_FORMAT`. Set it only if loading a GGUF that lacks one.
