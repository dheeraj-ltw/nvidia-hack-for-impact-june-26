"""Generation backend abstraction.

Picks a backend from the environment (loaded from .env by the caller):
  - LiteLLM / OpenAI-compatible gateway  (if LITELLM_API_KEY is set)  [preferred]
  - native Anthropic API                 (if ANTHROPIC_API_KEY is set) [fallback]

Exposes a single `make_chat()` factory returning a `chat(system, user) -> str`
callable, plus `describe()` for logging which backend/model is active.
"""
from __future__ import annotations

import os
from typing import Callable


def _litellm_chat(model: str, base_url: str, api_key: str) -> Callable[[str, str], str]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Some models (e.g. Bedrock Claude Opus 4.x thinking models) reject the
    # `temperature` parameter. Track that and stop sending it after the first refusal.
    state = {"send_temperature": True}

    def chat(system: str, user: str, *, temperature: float = 0.4,
             max_tokens: int = 2000) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if state["send_temperature"]:
            kwargs["temperature"] = temperature
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if "temperature" in str(exc).lower() and state["send_temperature"]:
                state["send_temperature"] = False
                kwargs.pop("temperature", None)
                resp = client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""

    return chat


def _anthropic_chat(model: str, api_key: str) -> Callable[[str, str], str]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def chat(system: str, user: str, *, temperature: float = 0.4,
             max_tokens: int = 2000) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    return chat


def _backend() -> tuple[str, str]:
    """Return (backend, model) without constructing a client (for describe())."""
    if os.environ.get("LITELLM_API_KEY"):
        return "litellm", os.environ.get("LITELLM_MODEL", "gemini-2.5-flash-uk")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", os.environ.get("GEN_MODEL", "claude-opus-4-8")
    return "none", ""


def describe() -> str:
    backend, model = _backend()
    if backend == "litellm":
        return f"litellm gateway ({os.environ.get('LITELLM_BASE_URL')}) model={model}"
    if backend == "anthropic":
        return f"anthropic api model={model}"
    return "no backend configured"


def make_chat() -> Callable[..., str]:
    backend, model = _backend()
    if backend == "litellm":
        return _litellm_chat(
            model=model,
            base_url=os.environ["LITELLM_BASE_URL"],
            api_key=os.environ["LITELLM_API_KEY"],
        )
    if backend == "anthropic":
        return _anthropic_chat(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
    raise SystemExit(
        "No generation backend configured. Set LITELLM_API_KEY (preferred) or "
        "ANTHROPIC_API_KEY in .env (see .env.example)."
    )
