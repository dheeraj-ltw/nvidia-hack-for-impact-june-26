# JARVIS — live data flow

How a live patrol session flows through the system in `AI_BACKEND=live`. Solid arrows are
forward/requests; dotted arrows are responses. Green = AI providers, blue = browser I/O,
orange = object storage.

```mermaid
flowchart LR
    subgraph BROWSER["Browser — Patrol Console (usePatrolSession.ts)"]
        direction TB
        CAM["Camera"]
        MIC["Microphone"]
        CAP["Frame capture<br/>~2 fps JPEG q0.6"]
        CLIP["Clip recorder<br/>4 s complete WebM"]
        UI["Feed · Detections · Transcript · Guidance cards"]
        SPK["TTS playback<br/>(Audio element)"]
    end

    subgraph BE["FastAPI backend"]
        direction TB
        SESS["WebSocket session<br/>(session.py) — decode binary"]
        REC[("Recorder → MinIO<br/>frames · audio · manifest")]
        HF["Orchestrator.handle_frame<br/>throttled ≤1 / 0.7 s, drop-if-busy"]
        HA["Orchestrator.handle_audio"]
        CARD["build SCENE CARD<br/>Scene: + Visible objects: + RECENT DIALOGUE:"]
        GUID["parse → GuidanceEvent"]
        SPEECH["base64 → SpeechEvent"]
    end

    subgraph PROV["AI providers (AI_BACKEND=live)"]
        direction TB
        NEB["Nebius Qwen2.5-VL<br/>vlm.py · 1 image, max_tokens 120"]
        STT["ElevenLabs Scribe STT<br/>diarize + word timestamps"]
        POL["PoliceAI reasoner<br/>police-llm GGUF · think + JSON"]
        TTS["ElevenLabs TTS<br/>eleven_multilingual_v2 → mp3"]
    end

    %% capture
    CAM --> CAP -->|"0x00 video"| SESS
    MIC --> CLIP -->|"0x02 audio clip"| SESS
    SESS -->|"0x01 chunk (recorded only)"| REC
    SESS -->|"frames + transcript/guidance"| REC

    %% VISION / VLM path
    SESS -->|"frame"| HF
    HF -->|"jpeg"| NEB
    NEB -.->|"scene summary"| HF
    HF -->|"DetectionEvent (boxes=[], summary)"| UI

    %% STT path
    SESS -->|"clip"| HA
    HA -->|"clip.webm"| STT
    STT -.->|"text + words"| HA
    HA -->|"TranscriptEvent (officer/subject)"| UI

    %% REASONING (scene summary + rolling transcript)
    HF --> CARD
    HA -.->|"rolling transcript"| CARD
    CARD -->|"system + SCENE CARD"| POL
    POL -.->|"think + JSON"| GUID
    GUID -->|"GuidanceEvent (cited)"| UI

    %% TTS path
    GUID -->|"suggestion text"| TTS
    TTS -.->|"mp3 bytes"| SPEECH
    SPEECH -->|"data:audio/mp3"| SPK

    classDef prov fill:#eef7e0,stroke:#76b900,color:#1c212e;
    classDef store fill:#fff6e6,stroke:#f59e0b,color:#1c212e;
    classDef io fill:#eaf1ff,stroke:#2563eb,color:#1c212e;
    class NEB,STT,POL,TTS prov;
    class REC store;
    class CAM,MIC,UI,SPK io;
```

## The three paths

- **Vision / VLM** — `Camera → frame capture (~2 fps) → 0x00 → handle_frame` (throttled ≤1 / 0.7 s) `→ Nebius Qwen2.5-VL` captions one JPEG → scene summary returns → `DetectionEvent` to the UI.
- **Speech-to-Text** — `Mic → 4 s complete WebM clip → 0x02 → handle_audio → ElevenLabs Scribe` (diarized, word timestamps) → text + words return → appended to the rolling transcript and labelled `officer`/`subject` → `TranscriptEvent` to the UI.
- **Reasoning + Text-to-Speech** — the scene summary and rolling transcript build a **SCENE CARD** → the fine-tuned **PoliceAI** model (`police-llm`) returns `<think>` + JSON → parsed into a cited `GuidanceEvent` → its suggestion is sent to **ElevenLabs TTS** → mp3 → `SpeechEvent` → played in the browser.

> Continuous `0x01` audio fragments are stored for the recorded track only — **not** transcribed; only the self-contained `0x02` clips hit STT. Every frame, transcript, and guidance event is also persisted to MinIO regardless of backend.
