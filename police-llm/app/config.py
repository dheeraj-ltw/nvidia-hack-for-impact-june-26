"""Runtime configuration for the police-llm OpenAI-compatible server.

Everything is driven by environment variables so the same image can serve any
GGUF model on any host (DGX Spark, a workstation GPU, or CPU-only for testing).
"""

import os
from dataclasses import dataclass


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _as_optional_str(name: str) -> str | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return None
    # .env files are single-line; allow explicit \n escapes for multi-line prompts.
    return raw.replace("\\n", "\n")


@dataclass
class Settings:
    # ---- model ----
    model_path: str = os.getenv("MODEL_PATH", "/models/model.gguf")
    # Public name advertised on /v1/models and echoed back in responses.
    model_name: str = os.getenv("MODEL_NAME", "policeai")

    # ---- llama.cpp runtime ----
    # -1 => offload all layers to GPU. Use 0 for CPU-only.
    n_gpu_layers: int = _as_int("N_GPU_LAYERS", -1)
    n_ctx: int = _as_int("N_CTX", 8192)
    n_batch: int = _as_int("N_BATCH", 512)
    # 0 => let llama.cpp pick a sensible default.
    n_threads: int = _as_int("N_THREADS", 0)
    # Override only if the GGUF lacks an embedded chat template. When None,
    # llama-cpp-python uses the template baked into the GGUF metadata.
    chat_format: str | None = _as_optional_str("CHAT_FORMAT")
    verbose: bool = os.getenv("LLAMA_VERBOSE", "false").lower() == "true"

    # ---- server / API ----
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _as_int("PORT", 8000)
    # If set, callers must send `Authorization: Bearer <api_key>`.
    api_key: str | None = _as_optional_str("API_KEY")
    # Optional system prompt injected when a request has no system message.
    # The Nemotron reasoning mode is enabled by "detailed thinking on".
    default_system_prompt: str | None = _as_optional_str("DEFAULT_SYSTEM_PROMPT")

    def llama_kwargs(self) -> dict:
        kwargs = dict(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            verbose=self.verbose,
        )
        if self.n_threads > 0:
            kwargs["n_threads"] = self.n_threads
        if self.chat_format:
            kwargs["chat_format"] = self.chat_format
        return kwargs


settings = Settings()
