"""Generate the PoliceAI fine-tuning dataset (England & Wales) via the Claude API.

For each scenario seed: retrieve grounding snippets from the corpus, ask Claude to
produce the assistant turn (<think> + JSON) citing only the supplied law, then write
a clean training record matching format.jsonl.

Outputs:
  data/output/dataset.jsonl        - training records {"messages": [...]}
  data/output/generation_meta.jsonl - aligned metadata (allowed citations, scene card)

Usage:
  python generate_dataset.py --n 100        # full run
  python generate_dataset.py --n 5          # smoke test
  python generate_dataset.py --n 100 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import re

from dotenv import load_dotenv

from common import OUTPUT_DIR, ROOT, write_jsonl
from llm import describe, make_chat
from parsing import normalise_assistant
from retrieval import Retriever
from scenario_taxonomy import build_scenarios

load_dotenv(ROOT / ".env")

PROMPTS = ROOT / "prompts"
SYSTEM_POLICEAI = (PROMPTS / "generation_system.txt").read_text(encoding="utf-8").strip()
INSTRUCTIONS = (PROMPTS / "generation_instructions.txt").read_text(encoding="utf-8").strip()

DATASET_PATH = OUTPUT_DIR / "dataset.jsonl"
META_PATH = OUTPUT_DIR / "generation_meta.jsonl"

TOP_K = 8
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def render_scene_card(s: dict) -> str:
    """Render the SCENE CARD user message (this becomes the training user turn)."""
    name, years, certs = s["officer"]
    return (
        f"SCENE CARD (updated {s['time']})\n"
        f"Location: {s['location']}\n"
        f"Incident type: {s['incident_type']}\n"
        f"Subjects: {s['subjects']}\n"
        f"Officer: {name}, {years} years experience\n"
        f"Duration: {s['duration']}\n"
        f"Key facts: {s['key_facts']}\n"
        f"Weather: {s['weather']}\n"
        f"Escalation index: {s['escalation_index']} ({s['escalation_label']})\n\n"
        f"QUERY: {s['query']}\n\n"
        f"OFFICER CONTEXT:\n"
        f"Jurisdiction: {s['jurisdiction']}\n"
        f"Years experience: {years}\n"
        f"Certs: {certs}\n"
        f"Prior incidents at location: {s['prior_incidents']}"
    )


def render_grounding(snippets: list[dict]) -> str:
    blocks = []
    for sn in snippets:
        blocks.append(
            f"[{sn['source_type'].upper()}] CITATION: {sn['citation']}\n{sn['text']}"
        )
    return "\n\n".join(blocks)


def build_gen_user(scene_card: str, snippets: list[dict]) -> str:
    return (
        f"=== GROUNDING SNIPPETS (cite only from these) ===\n"
        f"{render_grounding(snippets)}\n\n"
        f"=== SCENE CARD INPUT (this is the officer's turn you must answer) ===\n"
        f"{scene_card}\n\n"
        f"Now produce ONLY the assistant turn: a <think>...</think> block followed by the JSON object."
    )


def extract_assistant(text: str) -> str | None:
    """Validate and normalise to '<think>...</think>\\n{compact json}', or None."""
    return normalise_assistant(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--resume", action="store_true",
                    help="skip scenarios already present in dataset.jsonl")
    ap.add_argument("--workers", type=int, default=6,
                    help="concurrent generation requests (1 = sequential)")
    args = ap.parse_args()

    chat = make_chat()
    print(f"[generate] backend: {describe()} | workers={args.workers}")
    retriever = Retriever()
    scenarios = build_scenarios(args.n)
    system = f"{SYSTEM_POLICEAI}\n\n{INSTRUCTIONS}"

    done_ids: set[str] = set()
    records: list[dict] = []
    metas: list[dict] = []
    from common import read_jsonl
    if args.resume and META_PATH.exists():
        metas = read_jsonl(META_PATH)
        records = read_jsonl(DATASET_PATH)
        done_ids = {m["id"] for m in metas}
        print(f"[generate] resuming; {len(done_ids)} already done")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pending = [s for s in scenarios if s["id"] not in done_ids]

    def gen_one(s: dict) -> dict | None:
        """Generate one record; returns {'record':..., 'meta':...} or None on failure."""
        scene_card = render_scene_card(s)
        snippets = retriever.top(
            f"{s['retrieval_terms']} {s['query']} {s['key_facts']}", k=args.k
        )
        allowed = sorted({a for sn in snippets for a in sn["aliases"]})
        user = build_gen_user(scene_card, snippets)
        assistant = None
        for _ in range(3):
            try:
                # Generous budget: thinking models spend hidden reasoning tokens
                # on top of our visible <think> block + JSON.
                raw = chat(system, user, temperature=0.4, max_tokens=8000)
            except Exception as exc:  # noqa: BLE001 - log and retry
                print(f"[generate] {s['id']}: call error: {exc}")
                continue
            assistant = extract_assistant(raw)
            if assistant:
                break
        if not assistant:
            return None
        return {
            "record": {
                "messages": [
                    {"role": "system", "content": SYSTEM_POLICEAI},
                    {"role": "user", "content": scene_card},
                    {"role": "assistant", "content": assistant},
                ]
            },
            "meta": {
                "id": s["id"],
                "incident_type": s["incident_type"],
                "allowed_citations": allowed,
                "retrieved_ids": [sn["id"] for sn in snippets],
            },
        }

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lock = threading.Lock()
    failures = 0
    completed = len(done_ids)
    total = len(scenarios)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(gen_one, s): s for s in pending}
        for fut in as_completed(futures):
            s = futures[fut]
            out = fut.result()
            with lock:
                if out is None:
                    failures += 1
                    print(f"[generate] {s['id']}: FAILED to parse output")
                else:
                    records.append(out["record"])
                    metas.append(out["meta"])
                    completed += 1
                    # Persist incrementally so the run stays resumable.
                    write_jsonl(DATASET_PATH, records)
                    write_jsonl(META_PATH, metas)
                    print(f"[generate] {completed}/{total} {s['id']}: ok")

    print(f"[generate] done: {len(records)} records, {failures} failures -> {DATASET_PATH}")


if __name__ == "__main__":
    main()
