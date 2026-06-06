"""Scene-card synthesis — the NVIDIA NIM Nemotron step.

Between perception (the ElevenLabs STT transcript + the Nebius VLM scene caption) and
reasoning (the fine-tuned PoliceAI model), this composes the SCENE CARD with
`nvidia/llama-3.1-nemotron-70b-instruct`. PoliceAI was fine-tuned on a fixed SCENE CARD
format, so Nemotron is anchored to that exact skeleton (the deterministic draft from
policeai.build_scene_card) and only sharpens the content — turning a noisy live transcript and
a raw frame caption into one coherent incident summary, without changing the card's structure.

Kept deliberately small and defensive, like the VLM client: any error — or no
NVIDIA_API_KEY — degrades to an empty result, and the caller falls back to the deterministic
card, so reasoning never stalls on this step.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# Nemotron refines the draft card; it must NOT invent facts or change the layout, or it drifts
# the prompt away from the format PoliceAI was fine-tuned on. The draft is the single source of
# truth for structure — Nemotron only tightens wording and merges the transcript + scene into a
# clearer narrative.
_COMPOSE_PROMPT = (
    "You are a police dispatch analyst preparing a SCENE CARD for a legal-advisor model. "
    "Below is a draft SCENE CARD assembled from a live body-camera feed. Rewrite it into a "
    "cleaner, more coherent SCENE CARD that keeps the EXACT same section structure, labels, "
    "and the final QUERY line verbatim. Tighten the Scene description and reconcile it with the "
    "dialogue, but state only what the draft supports — never add facts, intent, guilt, or "
    "anything not present in the draft. Output only the SCENE CARD, nothing else.\n\n"
    "DRAFT:\n"
)


class SceneCardComposer:
    """Async client for NVIDIA NIM Nemotron (OpenAI-compatible chat/completions)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: httpx.Timeout | float = 20.0,
        max_tokens: int = 400,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_tokens = max_tokens

    async def compose(self, draft_card: str) -> str:
        """Return a refined SCENE CARD for the draft, or "" on any failure.

        With no NVIDIA API key configured, this is a no-op and the caller keeps the draft.
        """
        if not self._api_key or not draft_card.strip():
            return ""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        logger.info("→ Nemotron %s/chat/completions model=%s", self._base_url, self._model)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "user", "content": _COMPOSE_PROMPT + draft_card}
                        ],
                        "max_tokens": self._max_tokens,
                        "temperature": 0.0,
                    },
                )
            response.raise_for_status()
            # TypeError guards an unexpected response shape (choices not a list, message not a
            # dict) — degrade to the draft card, never crash the realtime session.
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            logger.warning("Nemotron scene-card composition failed: %s", error)
            return ""

        card = (content or "").strip()
        logger.info("← Nemotron composed scene card (%d chars)", len(card))
        return card
