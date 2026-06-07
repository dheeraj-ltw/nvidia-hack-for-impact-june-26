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
  // "officer" / "subject" / "unknown" live; post-session ID also emits "person1", "person2", ...
  speaker: string;
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

export interface OfficerSummary {
  officer_id: string;
  name: string;
  created_at: number;
  has_audio: boolean;
}

export interface SessionSummary {
  session_id: string;
  label: string | null;
  status: string; // "processing" | "ready"
  started_at: number;
  ended_at: number | null;
  frame_count: number;
  has_audio: boolean;
  has_video: boolean;
  has_report: boolean;
  event_count: number;
  officer_name?: string | null;
}

export interface RecordedEvent {
  offset_seconds: number;
  kind: "transcript" | "guidance" | "detection";
  payload: TranscriptEvent | GuidanceEvent | DetectionEvent;
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
  report_key: string | null;
  officer_id?: string | null;
  officer_name?: string | null;
  diarization?: Record<string, unknown> | null;
  scene_summary?: string | null;
}

export interface GuidanceSummary {
  offset_seconds: number;
  suggestion: string;
  severity: string;
  citations: string[];
}

export interface IncidentReport {
  session_id: string;
  label: string | null;
  officer_id?: string | null;
  officer_name?: string | null;
  started_at: number;
  ended_at: number | null;
  duration_seconds: number;
  generated_at: number;
  frame_count: number;
  transcript: string;
  guidance: GuidanceSummary[];
  speaker_labels: Record<string, unknown>;
  scene_summary: string;
}

export interface WebhookDispatch {
  target: string;
  url: string;
  delivered: boolean;
  status_code?: number | null;
  error?: string | null;
}

export interface ReportResult {
  report: IncidentReport;
  dispatches: WebhookDispatch[];
}

export type PatrolEvent =
  | TranscriptEvent
  | DetectionEvent
  | GuidanceEvent
  | AlertEvent
  | SpeechEvent
  | StatusEvent;
