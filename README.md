# PoliceAI — England & Wales fine-tuning dataset builder

Builds a **grounded-synthetic** instruction dataset for fine-tuning an open-weight model to act as
**PoliceAI**: a real-time, de-escalation-focused legal advisor for on-duty officers in **England &
Wales**. Each record matches `format.jsonl`: a `system` / `user` / `assistant` chat turn where the
user message is a **SCENE CARD** and the assistant replies with a `<think>` reasoning block followed
by a structured JSON object.

The law cited in the data is **real**: scenarios are generated against a corpus scraped from public
sources, and the model is instructed to cite **only** from the snippets retrieved for each scenario.
A validation pass cross-checks every citation back against that corpus.

## Approach

```
scrape (real law) → corpus index (citation-tagged) → scenario seeds
                                                          │
                            BM25 retrieve grounding ──────┤
                                                          ▼
                              Claude API → <think> + JSON  → validate (schema,
                              jurisdiction, no-guilt, citation grounding, dedupe)
                                                          ▼
                                              train.jsonl / val.jsonl
```

## Data sources (all openly licensed)

| Source | What | Licence |
|---|---|---|
| [legislation.gov.uk](https://www.legislation.gov.uk/developer) | E&W statutes as CLML XML (`/data.xml`) | Open Government Licence v3.0 |
| [Find Case Law](https://nationalarchives.github.io/ds-find-caselaw-docs/public) (The National Archives) | Court judgments as LegalDocML XML | Open Justice Licence |
| GOV.UK + curated PACE Codes reference | PACE Codes A/C/G key provisions | Open Government Licence v3.0 |

Statutes scraped: PACE 1984, Misuse of Drugs Act 1971, CJPOA 1994 (incl. s.60), Road Traffic Act
1988, Criminal Law Act 1967, Public Order Act 1986, Human Rights Act 1998, Terrorism Act 2000,
Mental Health Act 1983. Case law: landmark police-powers judgments (Roberts, Hicks, Beghal, Hayes,
ZH, Wood) plus recent Chief-Constable / Commissioner judgments discovered via search and filtered to
police-powers cases only. Guidance: curated PACE Code A/C/G provisions (stop & search, caution,
detainee rights, arrest necessity).

> Note: College of Policing APP deep pages are behind bot protection (HTTP 403), so the substantive
> guidance grounding comes from the curated PACE Codes reference in
> `data/corpus/police_guidance/pace_codes_reference.json`.

## Incident coverage (~100 examples)

Stop & search (drugs, weapons, s.60 no-suspicion), road traffic (documents, drink-driving, vehicle
cannabis), arrest necessity, custody rights & caution, breach of the peace, public order, use of
force / proportionality, filming officers, mental health (s.136), entry & search of premises.

## Setup

```bash
make setup                    # create .venv + install requirements
cp .env.example .env          # then add your ANTHROPIC_API_KEY
```

## Run (Makefile)

```bash
make data                     # full pipeline: scrape -> corpus -> generate -> validate -> split
make data N=200               # generate a different number of examples
make smoke                    # generate just 5 examples first (sanity check)
```

Individual stages: `make scrape`, `make corpus`, `make generate N=100`, `make validate`,
`make split`. Run `make help` to list everything. (`./run_all.sh 100` does the same end-to-end run.)

The final training data is written to a **separate top-level `training_data/` folder**, in the same
`format.jsonl` shape.

## Outputs

Final training data — `training_data/` (separate folder, `format.jsonl` shape):
- `training_data/dataset.jsonl` — all validated records.
- `training_data/train.jsonl`, `training_data/val.jsonl` — 90/10 split, ready for TRL / axolotl.

Intermediate artifacts — `data/`:
- `data/corpus_index.jsonl` — citation-tagged grounding snippets.
- `data/output/dataset.jsonl` — every generated record (pre-validation).
- `data/output/generation_meta.jsonl` — per-record allowed citations + retrieved snippet ids.
- `data/output/dataset.valid.jsonl` — records passing all hard validation checks.

## Validation checks

Hard (record dropped on failure): exact PoliceAI system prompt, `<think>` block present, JSON schema
(six keys; `escalation_risk` ∈ {low, medium, high}; non-empty `next_steps`), **no US-law leakage**,
**no guilt/innocence opinion**. Soft (reported as warnings): citation grounding — every case neutral
citation and PACE Code reference must appear in the snippets retrieved for that record. Near-duplicate
records are dropped.

## Notes / disclaimers

This is a hackathon research dataset. Generated legal guidance is **synthetic** and must be reviewed
by a qualified person before any operational use. Citations are grounded in scraped law but should be
re-verified against the live, in-force text.

---

# Patrol Assist

AI decision-support for police patrol. A live video + audio feed streams to a backend that
**records the session** (frames + audio) and runs it through a pluggable AI layer for scene
understanding, speech-to-text, and law-aligned reasoning — surfacing **assistive, cited guidance**
to the officer, with text-to-speech for hands-free use.

> The system is **assistive and human-in-the-loop**. It suggests and cites law; it never decides.

The AI is optional and pluggable. With `AI_BACKEND=null` (the default) **no model is connected**:
the console shows the live feed and every session is recorded to object storage, ready for a model
to analyze later. Swapping `AI_BACKEND` adds intelligence without touching the UI.

## Architecture

![Patrol Assist architecture](docs/architecture.png)

Full diagram and component build-status: [docs/architecture.md](docs/architecture.md).

| `AI_BACKEND` | Behavior |
|--------------|----------|
| `null` (default) | No model. Live feed + session recording only — no fabricated output. |
| `stub` | Deterministic local detections/guidance for exercising the event path (no keys). |
| `live` | NVIDIA NIM (VLM + Nemotron) + ElevenLabs (STT/TTS). |

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs
- MinIO console: http://localhost:9101 (`minioadmin` / `minioadmin`)

Click **Start patrol**, allow the camera, and the session records to MinIO. Stop it, and the
recording appears under **Recorded sessions** with frame thumbnails and audio playback.

### Local dev (without Docker)

```bash
# backend (needs a reachable MinIO/S3 — see .env)
cd backend && uv sync && uv run uvicorn app.main:app --reload

# frontend
cd frontend && npm install && npm run dev
```

## Repo layout

| Path | What |
|------|------|
| `backend/app/realtime/` | WebSocket ingest of frames + audio |
| `backend/app/recording/` | Per-session recorder → object storage |
| `backend/app/storage/` | Async S3/MinIO client |
| `backend/app/api/sessions.py` | List recorded sessions + media playback |
| `backend/app/ai/` | `AIService` interface — null / stub / live backends |
| `backend/app/pipeline/` | Orchestrator: frame → AI → events |
| `frontend/app/` | Live patrol console + recorded-sessions list |
| `frontend/lib/` | WebSocket client, capture loop, API client, types |
