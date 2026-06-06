"""OpenAI-compatible inference server for a GGUF model, backed by llama.cpp.

Exposes the endpoints other modules expect from the OpenAI API:
  GET  /v1/models
  POST /v1/chat/completions   (streaming + non-streaming)
  POST /v1/completions        (streaming + non-streaming)
  GET  /health

Because any `openai`-compatible client just needs a base_url + key, downstream
services can point `OpenAI(base_url="http://police-llm:8000/v1")` at this server.

llama.cpp's `Llama` object is NOT safe for concurrent calls, so all generation
is serialised behind a single lock and executed on one dedicated worker thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .config import settings
from .schemas import ChatCompletionRequest, CompletionRequest, to_llama_kwargs

# Populated on startup.
_llm: Any = None
# Serialise access: llama.cpp is single-context and not re-entrant.
_lock = asyncio.Lock()
# A single worker thread keeps all llama.cpp calls on the same OS thread.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm
    from llama_cpp import Llama

    print(f"[police-llm] loading model: {settings.model_path}")
    print(f"[police-llm] n_gpu_layers={settings.n_gpu_layers} n_ctx={settings.n_ctx}")
    _llm = await run_in_threadpool(lambda: Llama(**settings.llama_kwargs()))
    print("[police-llm] model loaded; ready to serve.")
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


async def _sse_from_sync(make_iter) -> Iterator[str]:
    """Bridge a blocking llama.cpp generator into an async SSE stream.

    The lock is held for the whole stream so a second request cannot interleave
    on the shared context. Chunks are produced on the worker thread and handed
    to the event loop via a queue.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def produce():
        try:
            for chunk in make_iter():
                loop.call_soon_threadsafe(queue.put_nowait, _stamp_model(chunk))
        except Exception as exc:  # surface generation errors to the client
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
                raise item
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
    return {"status": "ok", "model_loaded": _llm is not None}


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
        result = await _run_blocking(
            lambda: _llm.create_chat_completion(messages=messages, **kwargs)
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
        result = await _run_blocking(
            lambda: _llm.create_completion(prompt=req.prompt, **kwargs)
        )
    return _stamp_model(result)
