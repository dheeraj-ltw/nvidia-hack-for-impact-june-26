"""Validate the generated PoliceAI dataset.

Checks each record for: exact format match (system prompt + <think> + JSON schema),
England & Wales jurisdiction (no US law leakage), no guilt/innocence opinions,
and citation grounding (every case/Code citation must appear in the snippets that
were retrieved for that record, per generation_meta.jsonl).

Writes data/output/dataset.valid.jsonl (records passing all HARD checks) and prints
a report. Soft issues (e.g. weak citation grounding) are reported as warnings.

Usage: python validate_dataset.py
"""
from __future__ import annotations

import json
import re

from jsonschema import Draft202012Validator

from common import OUTPUT_DIR, read_jsonl, write_jsonl
from parsing import get_json

DATASET_PATH = OUTPUT_DIR / "dataset.jsonl"
META_PATH = OUTPUT_DIR / "generation_meta.jsonl"
VALID_PATH = OUTPUT_DIR / "dataset.valid.jsonl"

SYSTEM_POLICEAI = (
    "You are PoliceAI, a real-time legal advisor for on-duty law enforcement officers. "
    "Your role is to provide legally-grounded, de-escalation-focused guidance during active "
    "incidents. You NEVER give an opinion on guilt or innocence. All advice must be phrased as "
    "guidance, not commands. Always cite specific case law or statutes. You must reason step by "
    "step through the legal situation before producing your final structured JSON response."
)

JSON_SCHEMA = {
    "type": "object",
    "required": ["legal_basis", "suggested_action", "tone_guidance",
                 "escalation_risk", "rights_reminder", "next_steps"],
    "properties": {
        "legal_basis": {"type": "string", "minLength": 20},
        "suggested_action": {"type": "string", "minLength": 20},
        "tone_guidance": {"type": "string", "minLength": 10},
        "escalation_risk": {"enum": ["low", "medium", "high"]},
        "rights_reminder": {"type": "string", "minLength": 10},
        "next_steps": {"type": "array", "minItems": 2, "items": {"type": "string"}},
    },
    "additionalProperties": True,
}
VALIDATOR = Draft202012Validator(JSON_SCHEMA)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
NEUTRAL_CITE_RE = re.compile(r"\[\d{4}\]\s+[A-Z][A-Za-z]*(?:\s+[A-Za-z]+){0,3}\s+\d+")
CODE_RE = re.compile(r"Code\s+[A-H]\b", re.IGNORECASE)

# Tokens that indicate US law leaked in (England & Wales only allowed).
US_LAW = re.compile(
    r"\bU\.S\.|United States|SCOTUS|Fourth Amendment|Fifth Amendment|Miranda\b|"
    r"\bILCS\b|\bCVC\b|Vehicle Code|Penal Code|\bCalifornia\b|\bIllinois\b|"
    r"Supreme Court of the United States|§",
    re.IGNORECASE,
)
# Context that makes "guilty"/"innocent" NOT an opinion on the subject: statutory
# offence definitions ("guilty of an offence"), and conditional/advisory phrasing
# (incl. the model correctly refusing to opine, or reasoning about the suspicion test).
_GUILT_EXCLUDE = (
    "whether", "if ", "not ", "never", "cannot", "can't", "do not", "don't",
    "view", "opinion", "express", "presum", "suspect", "grounds", "reasonable",
    "avoid", "refrain", "no comment", "without", "question of",
)


def find_guilt_opinion(text: str) -> str | None:
    """Flag only genuine assertions of the subject's guilt/innocence."""
    for m in re.finditer(r"\b(guilty|innocent)\b", text, re.IGNORECASE):
        pre = text[max(0, m.start() - 70):m.start()].lower()
        post = text[m.end():m.end() + 25].lower()
        if m.group(1).lower() == "guilty" and re.match(
            r"\s+(of\s+(an?|the)\s+offence|if\b|where\b|when\b|unless\b)", post
        ):
            continue  # statutory definition: "guilty of an offence" / "guilty if ..."
        if any(w in pre for w in _GUILT_EXCLUDE):
            continue  # conditional / advisory / reasoning about the test
        return text[max(0, m.start() - 40):m.end() + 25]
    return None


def get_assistant_json(assistant: str) -> dict | None:
    return get_json(assistant)


def check_record(rec: dict, meta: dict) -> dict:
    res = {"id": meta.get("id", "?"), "hard": [], "warn": []}
    msgs = rec.get("messages", [])
    roles = [m["role"] for m in msgs]
    if roles != ["system", "user", "assistant"]:
        res["hard"].append(f"bad role sequence: {roles}")
        return res
    if msgs[0]["content"].strip() != SYSTEM_POLICEAI:
        res["hard"].append("system prompt does not match canonical PoliceAI prompt")

    assistant = msgs[2]["content"]
    if "<think>" not in assistant or "</think>" not in assistant:
        res["hard"].append("missing <think> reasoning block")

    obj = get_assistant_json(assistant)
    if obj is None:
        res["hard"].append("assistant JSON not parseable")
        return res
    schema_errors = sorted(VALIDATOR.iter_errors(obj), key=lambda e: e.path)
    for e in schema_errors:
        res["hard"].append(f"schema: {e.message}")

    if US_LAW.search(assistant):
        res["hard"].append(f"US-law token present: {US_LAW.search(assistant).group(0)!r}")
    guilt = find_guilt_opinion(assistant)
    if guilt:
        res["hard"].append(f"guilt/innocence opinion: {guilt!r}")

    # --- citation grounding ---
    allowed = meta.get("allowed_citations", [])
    allowed_blob = " || ".join(allowed).lower()
    # Neutral citations must each appear in the allowed set.
    for cite in set(NEUTRAL_CITE_RE.findall(assistant)):
        if cite.lower() not in allowed_blob:
            res["warn"].append(f"ungrounded case citation: {cite!r}")
    # PACE Code references must be backed by a guidance alias.
    for code in set(c.upper() for c in CODE_RE.findall(assistant)):
        norm = code.lower().replace("code ", "code ")
        if norm not in allowed_blob:
            res["warn"].append(f"ungrounded Code reference: {code!r}")
    # legal_basis should cite at least one grounded source.
    lb = obj.get("legal_basis", "").lower()
    if allowed and not any(
        tok in lb for a in allowed for tok in [a.lower()] if len(a) > 4
    ):
        # fall back: any 'section N' / neutral cite / 'Code X' present
        if not (re.search(r"section\s+\d+|s\.\s*\d+", lb) or NEUTRAL_CITE_RE.search(lb)
                or CODE_RE.search(lb)):
            res["warn"].append("legal_basis cites no recognisable statute/case/Code")
    return res


def main() -> None:
    records = read_jsonl(DATASET_PATH)
    metas = read_jsonl(META_PATH)
    if len(records) != len(metas):
        print(f"!! warning: {len(records)} records vs {len(metas)} metas; "
              f"validating min length aligned by order")
    n = min(len(records), len(metas))

    valid: list[dict] = []
    seen_hashes: set[str] = set()
    hard_fail = 0
    warned = 0
    dupes = 0

    for rec, meta in zip(records[:n], metas[:n]):
        res = check_record(rec, meta)
        if res["hard"]:
            hard_fail += 1
            print(f"[FAIL] {res['id']}: " + "; ".join(res["hard"]))
            continue
        if res["warn"]:
            warned += 1
            print(f"[warn] {res['id']}: " + "; ".join(res["warn"]))
        # dedupe on the assistant JSON body
        obj = get_assistant_json(rec["messages"][2]["content"])
        key = json.dumps(
            {k: obj.get(k) for k in ("legal_basis", "suggested_action")},
            sort_keys=True,
        )
        h = str(hash(key))
        if h in seen_hashes:
            dupes += 1
            print(f"[dupe] {res['id']}: near-duplicate, dropped")
            continue
        seen_hashes.add(h)
        valid.append(rec)

    write_jsonl(VALID_PATH, valid)
    print("\n=== VALIDATION REPORT ===")
    print(f"records checked:   {n}")
    print(f"hard failures:     {hard_fail}")
    print(f"citation warnings: {warned}")
    print(f"duplicates dropped:{dupes}")
    print(f"VALID records:     {len(valid)} -> {VALID_PATH}")


if __name__ == "__main__":
    main()
