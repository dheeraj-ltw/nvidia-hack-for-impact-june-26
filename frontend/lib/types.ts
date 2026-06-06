// Mirrors backend/app/models/events.py — the WS wire contract.

export type Severity = "info" | "caution" | "critical";

export interface BoundingBox {
  label: string;
  confidence: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface LegalCitation {
  title: string;
  reference: string;
  snippet?: string | null;
}

export interface TranscriptEvent {
  type: "transcript";
  ts: number;
  text: string;
  speaker: "officer" | "subject" | "unknown";
  is_final: boolean;
}

export interface DetectionEvent {
  type: "detection";
  ts: number;
  boxes: BoundingBox[];
  summary?: string | null;
}

export interface GuidanceEvent {
  type: "guidance";
  ts: number;
  suggestion: string;
  rationale?: string | null;
  citations: LegalCitation[];
  severity: Severity;
}

export interface AlertEvent {
  type: "alert";
  ts: number;
  message: string;
  severity: Severity;
}

export interface SpeechEvent {
  type: "speech";
  ts: number;
  audio_b64: string;
  text: string;
}

export interface StatusEvent {
  type: "status";
  ts: number;
  state: string;
  detail?: string | null;
  session_id?: string | null;
}

export interface SessionSummary {
  session_id: string;
  label: string | null;
  started_at: number;
  ended_at: number | null;
  frame_count: number;
  has_audio: boolean;
  has_video: boolean;
  event_count: number;
}

export interface RecordedEvent {
  offset_seconds: number;
  kind: "transcript" | "guidance";
  payload: TranscriptEvent | GuidanceEvent;
}

export interface SessionManifest {
  session_id: string;
  label: string | null;
  started_at: number;
  ended_at: number | null;
  frame_count: number;
  has_audio: boolean;
  audio_key: string | null;
  video_key: string | null;
  events: RecordedEvent[];
}

export type PatrolEvent =
  | TranscriptEvent
  | DetectionEvent
  | GuidanceEvent
  | AlertEvent
  | SpeechEvent
  | StatusEvent;
