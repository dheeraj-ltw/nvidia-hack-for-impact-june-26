"""Pydantic request models mirroring the subset of the OpenAI API we support.

Responses are passed through from llama-cpp-python, which already emits
OpenAI-shaped dicts, so we only need to validate inbound requests.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repeat_penalty: float | None = None
    seed: int | None = None
    # OpenAI JSON-mode / schema, forwarded to llama.cpp grammar constraint.
    response_format: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repeat_penalty: float | None = None
    seed: int | None = None

    model_config = {"extra": "allow"}


# Parameters that map 1:1 onto llama-cpp-python create_* calls.
_PASSTHROUGH = (
    "temperature", "top_p", "top_k", "max_tokens", "stop",
    "presence_penalty", "frequency_penalty", "repeat_penalty",
    "seed", "response_format",
)


def to_llama_kwargs(req: BaseModel) -> dict:
    """Build kwargs for llama-cpp-python, dropping unset (None) fields."""
    data = req.model_dump(exclude_none=True)
    return {k: data[k] for k in _PASSTHROUGH if k in data}
