# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A single repo for **JARVIS** (pitch name; the code uses the older subsystem names **Patrol Assist** + **PoliceAI**) — an NVIDIA Impact Hackathon project: a real-time, on-device legal & de-escalation copilot for police, targeting the DGX Spark. It contains **five loosely-coupled subsystems** with separate toolchains; treat them independently:

1. **PoliceAI dataset pipeline** — root `src/` + `Makefile` + `prompts/` + `data/`. Builds a fine-tuning dataset of England & Wales policing scenarios. Python, root `.venv`.
2. **Fine-tuning** — `models/fine-tune/`. QLoRA on an NVIDIA Nemotron model for the DGX Spark. Its **own** `requirements.txt` and CUDA toolchain.
3. **Patrol Assist app** — `backend/` (FastAPI, `uv`) + `frontend/` (Next.js) + MinIO, wired by `docker-compose.yml`.
4. **Local inference microservices** — `police-llm/` (llama.cpp GGUF server) is the live `reason` backend (the backend calls it at `POLICEAI_BASE_URL`). `llava-video-to-text/` (vLLM video captioner) is built standalone but **not** wired into the live loop — the live vision path uses remote Nebius instead.
5. **Remote video-to-text** — `src/video_to_text.py` (offline, multi-frame CLI) and `backend/app/ai/vlm.py` (live, per-frame) both call Nebius-hosted Qwen2.5-VL via an OpenAI-compatible API.

Read `README.md` and `docs/architecture.md` (build-status table) before assuming a component is integrated — several are built standalone but not connected.

## Commands

### Dataset pipeline (root `.venv`, driven by the Makefile)
```bash
make setup                 # python3 -m venv .venv + pip install -r requirements.txt
cp .env.example .env       # then add ANTHROPIC_API_KEY (or LITELLM_* gateway)
make smoke                 # generate 5 examples (sanity check)
make data N=200            # full pipeline: scrape -> corpus -> generate -> validate -> split
make scrape | corpus | generate N=100 | validate | split   # individual stages
make video2text VIDEO=path/to.mov FRAMES=16 FPS=1           # Nebius Qwen2.5-VL (needs NEBIUS_API_KEY)
make tts TEXT="..." OUT=out.mp3 | stt AUDIO=in.mp3          # ElevenLabs (needs ELEVENLABS_API_KEY)
make identify CONV=conv.wav REF=officer_ref.mp3             # speaker labelling
make help                  # list all targets
```
`./run_all.sh [N]` is the same end-to-end run. Final training data is written to a **top-level `training_data/`** folder (not `data/`).

### Patrol Assist full stack (Docker)
```bash
docker compose up --build -d     # first run (or after dep changes); then just `up -d`
docker compose --profile reasoner up -d   # also start police-llm (needs a GGUF in ./models)
```
Frontend http://localhost:3000 · API docs http://localhost:8000/docs · MinIO console http://localhost:9101 (`minioadmin`/`minioadmin`). Code hot-reloads via bind mounts — only pass `--build` when `backend/pyproject.toml` or `frontend/package.json` change.

### Backend (without Docker)
```bash
cd backend && uv sync && uv run uvicorn app.main:app --reload
uv run pytest                                   # all tests
uv run pytest tests/test_pipeline.py::test_name # single test
uv run ruff check . && uv run mypy app          # lint + types (configured in pyproject.toml)
```

### Frontend (without Docker)
```bash
cd frontend && npm install && npm run dev
npm run lint && npm run typecheck && npm run build
```

### Fine-tuning (on a DGX Spark / CUDA host)
```bash
cd models/fine-tune && pip install -r requirements.txt
HF_TOKEN=... bash train.sh    # QLoRA train -> checkpoints/
bash eval.sh                  # eval an adapter
bash gguf.sh                  # merge adapter -> GGUF (Q4_K_M) for police-llm
```
`transformers` is **pinned to 4.48.3** — the Nemotron/DeciLM remote code breaks on >=4.50. Don't bump it.

### Pitch deck
`docs/JARVIS.md` (Marp) and `docs/JARVIS.pptx` are kept in sync. Regenerate the pptx with `.venv/bin/python docs/build_pptx.py`. The architecture image used by the deck is `docs/architecture_light.png` (from `architecture_light.mmd`); the dark `docs/architecture.png` is for the README.

## Architecture notes (the non-obvious parts)

**Dataset pipeline is grounded-then-validated, not free-generation.** `scrape_*.py` pull real law → `build_corpus.py` produces a citation-tagged corpus index → `retrieval.py` does BM25 retrieval (with a "guidance floor" that always reserves slots for PACE-Code provisions) → `generate_dataset.py` feeds the retrieved snippets + a deterministic SCENE CARD (from `scenario_taxonomy.py`) to Claude, which must cite **only** those snippets → `validate_dataset.py` enforces hard checks (exact system prompt, `<think>` block, JSON schema, no US-law leakage, no guilt opinions, **every citation must trace back to the retrieved snippets**, dedupe). The retrieval→generation→validation grounding loop is the core value; changes to any one stage must keep the citation contract intact.

**The backend AI layer is a swappable interface, not hardcoded logic.** `backend/app/ai/base.py` defines the `AIService` protocol (`transcribe`/`analyze_frame`/`reason`/`speak`); `factory.py` picks `stub.py` or `live.py` from the `AI_BACKEND` env var. The UI and recording path are identical regardless of backend, so **every session is recorded to MinIO even in `stub` mode** for later analysis. `stub` is the default and needs **no API keys** — use it to exercise the full event path. In `live` (`app/ai/live.py`): `transcribe`/`speak` call ElevenLabs Scribe/TTS; `analyze_frame` calls **Nebius-hosted Qwen2.5-VL** (`app/ai/vlm.py`) to caption the frame into a one-line scene summary (no bounding boxes yet — `boxes` is empty); `reason` calls the **fine-tuned PoliceAI** model via its OpenAI-compatible server (`POLICEAI_BASE_URL`, served by `police-llm/`), prompted with the SCENE CARD format from `app/ai/policeai.py`. Each provider call degrades to an empty result on error rather than crashing the session, and is skipped when its key is unset.

**Real-time path is throttled drop-if-busy.** `backend/app/pipeline/orchestrator.py` runs at most one analysis per ~0.7 s and drops frames while one is in flight — the live loop never queues up. Frames + audio arrive over a **binary WebSocket** (`backend/app/realtime/session.py`): a 1-byte kind (video JPEG / audio fragment / audio clip) + timestamp + dims + payload. Events (detection/transcript/guidance/speech) stream back as JSON.

**Speaker identification is two-stage.** Live: local `resemblyzer` voice embeddings match each clip against the enrolled officer (cosine ≥ `speaker_match_threshold`, default 0.70) → tags `officer`/`subject` with no API call. Post-session: a background pass re-transcribes with ElevenLabs Scribe diarization and relabels `officer`/`person1`/`person2`… Both live in `backend/app/ai/speaker_id.py` (and a standalone `src/speaker_id.py`).

**Config is one shared root `.env`.** `docker-compose.yml` (`env_file: .env`) and the root `src/` scripts read the same file. `.env.example` is the authoritative key list: `ANTHROPIC_API_KEY`/`LITELLM_*` (dataset generation), `ELEVENLABS_API_KEY` (voice), `NEBIUS_API_KEY` (video2text), `NVIDIA_API_KEY`+`NEMOTRON_MODEL` (live reasoner), `AI_BACKEND`, and S3/MinIO + port overrides.

**The fine-tuned model is served separately and plugged in by URL.** `gguf.sh` exports the trained adapter to a GGUF that `police-llm/` serves as an OpenAI-compatible endpoint. To use it as the live reasoner, start the `reasoner` compose profile and point the backend at `POLICEAI_BASE_URL=http://police-llm:8000/v1`. `police-llm` serializes all generation behind a single lock (llama.cpp context is not re-entrant).
