"""OpenAI-compatible inference server for a GGUF model, backed by llama.cpp.

Exposes the endpoints other modules expect from the OpenAI API:
  GET  /v1/models
  POST /v1/chat/completions   (streaming + non-streaming)
  POST /v1/completions        (streaming + non-streaming)
  GET  /health
  GET  /ready

Because any `openai`-compatible client just needs a base_url + key, downstream
services can point `OpenAI(base_url="http://police-llm:8000/v1")` at this server.

llama.cpp's `Llama` object is NOT safe for concurrent calls, so all generation
is serialised behind a single lock and executed on one dedicated worker thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import subprocess
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .config import settings
from .schemas import ChatCompletionRequest, CompletionRequest, to_llama_kwargs

# Populated on startup.
_llm: Any = None
_gpu_offload: bool | None = None
# Serialise access: llama.cpp is single-context and not re-entrant.
_lock = asyncio.Lock()
# A single worker thread keeps all llama.cpp calls on the same OS thread.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _log_runtime_diagnostics():
    """Print host/GPU info to help debug connection resets (usually OOM or segfault)."""
    print("[police-llm] ---- runtime diagnostics ----", flush=True)
    print(f"[police-llm] MODEL_PATH={settings.model_path}", flush=True)
    model = Path(settings.model_path)
    if model.is_file():
        print(f"[police-llm] model file OK ({model.stat().st_size / 1e9:.1f} GB)", flush=True)
    else:
        print(f"[police-llm] ERROR: model file missing: {model}", flush=True)

    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                print(f"[police-llm] {line}", flush=True)
        else:
            print("[police-llm] WARNING: nvidia-smi failed — GPU may not be visible in container", flush=True)
            if out.stderr:
                print(f"[police-llm] nvidia-smi stderr: {out.stderr.strip()}", flush=True)
    except FileNotFoundError:
        print("[police-llm] WARNING: nvidia-smi not found", flush=True)

    print(
        f"[police-llm] n_gpu_layers={settings.n_gpu_layers} "
        f"n_ctx={settings.n_ctx} n_batch={settings.n_batch}",
        flush=True,
    )
    print("[police-llm] -----------------------------", flush=True)


def _load_model():
    from llama_cpp import Llama

    _log_runtime_diagnostics()
    if not Path(settings.model_path).is_file():
        raise FileNotFoundError(f"GGUF not found: {settings.model_path}")

    print("[police-llm] loading model (this can take several minutes) …", flush=True)
    t0 = time.time()
    llm = Llama(**settings.llama_kwargs())
    print(f"[police-llm] model loaded in {time.time() - t0:.0f}s", flush=True)

    global _gpu_offload
    try:
        from llama_cpp.llama_cpp import llama_supports_gpu_offload

        _gpu_offload = bool(llama_supports_gpu_offload())
        print(f"[police-llm] gpu_offload_supported={_gpu_offload}", flush=True)
        if settings.n_gpu_layers != 0 and not _gpu_offload:
            print(
                "[police-llm] WARNING: CUDA offload not available but N_GPU_LAYERS "
                f"is {settings.n_gpu_layers}. Set N_GPU_LAYERS=0 or fix GPU passthrough.",
                flush=True,
            )
    except Exception as exc:
        print(f"[police-llm] could not query gpu offload support: {exc}", flush=True)

    return llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm
    try:
        _llm = await run_in_threadpool(_load_model)
        print("[police-llm] ready to serve.", flush=True)
    except Exception:
        traceback.print_exc()
        print("[police-llm] FATAL: model failed to load; exiting.", flush=True)
        sys.exit(1)

    yield

    _llm = None
    _executor.shutdown(wait=False)


app = FastAPI(title="police-llm", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Auth (optional; enabled only when API_KEY is set)
# ---------------------------------------------------------------------------

async def verify_api_key(authorization: str | None = Header(default=None)):
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stamp_model(obj: dict) -> dict:
    """Advertise our configured model name rather than the raw file path."""
    if isinstance(obj, dict):
        obj["model"] = settings.model_name
    return obj


async def _run_blocking(fn):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn)


async def _generate(make_fn, *, label: str):
    """Run a llama.cpp call; convert Python errors to HTTP 500 (not connection reset)."""
    try:
        return await _run_blocking(make_fn)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed during {label}: {type(exc).__name__}: {exc}",
        ) from exc


async def _sse_from_sync(make_iter) -> Iterator[str]:
    """Bridge a blocking llama.cpp generator into an async SSE stream."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def produce():
        try:
            for chunk in make_iter():
                loop.call_soon_threadsafe(queue.put_nowait, _stamp_model(chunk))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    async with _lock:
        future = loop.run_in_executor(_executor, produce)
        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, Exception):
                traceback.print_exception(type(item), item, item.__traceback__)
                yield f"data: {json.dumps({'error': str(item)})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"
        await future
    yield "data: [DONE]\n\n"


def _inject_default_system(messages: list[dict]) -> list[dict]:
    if not settings.default_system_prompt:
        return messages
    if any(m.get("role") == "system" for m in messages):
        return messages
    return [{"role": "system", "content": settings.default_system_prompt}, *messages]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _llm is not None,
        "model_path": settings.model_path,
        "gpu_offload_supported": _gpu_offload,
    }


@app.get("/ready")
async def ready():
    if _llm is None:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "model not loaded"})
    if not Path(settings.model_path).is_file():
        return JSONResponse(status_code=503, content={"ready": False, "reason": "model file missing"})
    return {"ready": True, "model": settings.model_name}


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "police-llm",
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: ChatCompletionRequest):
    if _llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    messages = _inject_default_system([m.model_dump(exclude_none=True) for m in req.messages])
    kwargs = to_llama_kwargs(req)

    if req.stream:
        make_iter = lambda: _llm.create_chat_completion(messages=messages, stream=True, **kwargs)
        return StreamingResponse(_sse_from_sync(make_iter), media_type="text/event-stream")

    async with _lock:
        result = await _generate(
            lambda: _llm.create_chat_completion(messages=messages, **kwargs),
            label="chat completion",
        )
    return _stamp_model(result)


@app.post("/v1/completions", dependencies=[Depends(verify_api_key)])
async def completions(req: CompletionRequest):
    if _llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    kwargs = to_llama_kwargs(req)

    if req.stream:
        make_iter = lambda: _llm.create_completion(prompt=req.prompt, stream=True, **kwargs)
        return StreamingResponse(_sse_from_sync(make_iter), media_type="text/event-stream")

    async with _lock:
        result = await _generate(
            lambda: _llm.create_completion(prompt=req.prompt, **kwargs),
            label="completion",
        )
    return _stamp_model(result)
