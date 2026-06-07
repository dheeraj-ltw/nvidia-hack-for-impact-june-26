"""Scene-card compiler — Nebius-hosted Nemotron 3 Super.

This model's ONLY job is to compile a factual SCENE CARD from a session's two recorded
context streams — the speech-to-text transcript and the VLM scene descriptions. It describes
what was seen and said; it does NOT give legal advice, cite or interpret law, assess
suspicion/guilt, or decide what the officer should do. Legal reasoning is a separate, later
step (not wired up here).

Mirrors `vlm.py`'s async-httpx + Bearer + defensive-degradation pattern. On any failure it
returns "" so the caller can fall back to the deterministic `policeai.build_session_scene_card`.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# The compiler is deliberately NOT the PoliceAI legal-advisor prompt — it must stay purely
# descriptive. The PoliceAI system prompt is attached later, around the finished card, for the
# downstream legal model.
_COMPILER_SYSTEM = (
    "You compile a factual SCENE CARD for a police body-worn-camera session. You are given a "
    "speech-to-text transcript of what was said and chronological visual scene descriptions "
    "from a vision model. Produce a SCENE CARD using EXACTLY the field layout you are given, "
    "keeping every field name and the order. Fill each field ONLY from what is observed in the "
    "inputs; write \"Unknown\" when a field cannot be determined. Summarise repeated visual "
    "descriptions into one coherent account. Report only observable facts: the people (number, "
    "appearance, actions, demeanour), objects, the type of setting, and what was said. You MUST "
    "NOT give legal advice, cite or interpret any law, assess suspicion or guilt, or state what "
    "the officer should do. Output ONLY the SCENE CARD text — nothing before or after it."
)


class SceneCardComposer:
    """Async client that composes a factual SCENE CARD via the Nebius-hosted model."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: httpx.Timeout | float = 90.0,
        max_tokens: int = 1500,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._max_tokens = max_tokens

    def _user_message(
        self, *, transcript: str, scene_text: str, officer_name: str | None,
        duration: str, updated_clock: str,
    ) -> str:
        return (
            "VISUAL SCENE ANALYSIS (chronological):\n"
            f"{scene_text.strip() or '(no visual analysis captured)'}\n\n"
            "SPEECH-TO-TEXT TRANSCRIPT:\n"
            f"{transcript.strip() or '(no speech transcribed)'}\n\n"
            "Fill this SCENE CARD exactly, keeping the field names and order:\n\n"
            f"SCENE CARD (updated {updated_clock})\n"
            "Location: <type of setting observed, e.g. indoors / roadside; or Unknown>\n"
            "Incident type: <short factual category, e.g. street encounter / traffic stop / "
            "welfare check; or Unknown>\n"
            "Subjects: <number, gender if visible, appearance, demeanour>\n"
            f"Officer: {officer_name or 'Unknown'}\n"
            f"Duration: {duration}\n"
            "Key facts: <one coherent factual summary of what is seen and said>\n"
            "Weather: <if observable, else Unknown>\n"
            "Escalation index: <0.00-1.00> (<LOW | ELEVATED | HIGH>)\n\n"
            "QUERY: <one neutral question summarising what situation the officer is handling>\n\n"
            "OFFICER CONTEXT:\n"
            "Jurisdiction: England & Wales\n"
            "Years experience: Unknown\n"
            "Certs: Unknown\n"
            "Prior incidents at location: Unknown"
        )

    async def compose(
        self, *, transcript: str, scene_text: str, officer_name: str | None,
        duration: str, updated_clock: str,
    ) -> str:
        """Return a compiled SCENE CARD string, or "" on any failure / missing key."""
        if not self._api_key:
            return ""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        user = self._user_message(
            transcript=transcript, scene_text=scene_text, officer_name=officer_name,
            duration=duration, updated_clock=updated_clock,
        )

        logger.info("→ Nebius scene compiler %s model=%s", self._base_url, self._model)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": _COMPILER_SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.2,
                        "max_tokens": self._max_tokens,
                    },
                )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            logger.warning("Nebius scene compiler failed: %s", error)
            return ""

        card = (content or "").strip()
        # Strip any stray markdown fences the model may wrap the card in.
        if card.startswith("```"):
            card = card.strip("`").lstrip("\n")
            if card.lower().startswith("text\n"):
                card = card[5:]
        logger.info("← Nebius scene compiler returned %d chars", len(card))
        return card.strip()
