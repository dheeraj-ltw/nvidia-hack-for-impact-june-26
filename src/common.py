"""Shared helpers for the PoliceAI dataset pipeline (England & Wales)."""
from __future__ import annotations

import json
import pathlib
import time

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
LEGISLATION_DIR = CORPUS / "legislation"
CASELAW_DIR = CORPUS / "caselaw"
GUIDANCE_DIR = CORPUS / "police_guidance"
CORPUS_INDEX = DATA / "corpus_index.jsonl"
OUTPUT_DIR = DATA / "output"
# Final, shippable training data (separate top-level folder), in format.jsonl shape.
TRAINING_DIR = ROOT / "training_data"

USER_AGENT = (
    "PoliceAI-dataset-builder/0.1 (NVIDIA AI-for-Impact hackathon; "
    "educational research; contact: dheerajksingh@gmail.com)"
)

# Be a polite scraper of public services.
REQUEST_DELAY_S = 1.0


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def fetch(url: str, *, accept: str | None = None) -> httpx.Response:
    """GET a URL with retries, a descriptive UA, and redirect following."""
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_S)
    return resp


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
