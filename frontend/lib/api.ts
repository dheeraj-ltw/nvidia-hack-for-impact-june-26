import type {
  IncidentReport,
  ReportResult,
  SessionManifest,
  SessionSummary,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function fetchSessions(): Promise<SessionSummary[]> {
  const response = await fetch(`${API_BASE}/sessions`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load sessions (${response.status})`);
  return response.json();
}

export async function fetchSession(sessionId: string): Promise<SessionManifest> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load session (${response.status})`);
  return response.json();
}

export async function renameSession(
  sessionId: string,
  label: string,
): Promise<SessionSummary> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  if (!response.ok) throw new Error(`Failed to rename session (${response.status})`);
  return response.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Failed to delete session (${response.status})`);
}

export async function fetchReport(sessionId: string): Promise<IncidentReport> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/report`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Failed to load report (${response.status})`);
  return response.json();
}

export async function generateReport(sessionId: string): Promise<ReportResult> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/report`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(`Failed to generate report (${response.status})`);
  return response.json();
}

// Scene card = the factual context the model compiles from STT + video analysis. Returns the
// SCENE CARD text (the `user` message of the request-format JSONL).
function sceneCardFromRecord(record: unknown): string {
  const messages = (record as { messages?: { role: string; content: string }[] })?.messages;
  return messages?.find((m) => m.role === "user")?.content ?? "";
}

export async function generateSceneCard(sessionId: string): Promise<string> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/scenecard`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(`Failed to generate scene card (${response.status})`);
  return sceneCardFromRecord(await response.json());
}

export async function fetchSceneCard(sessionId: string): Promise<string | null> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/scenecard`, {
    cache: "no-store",
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Failed to load scene card (${response.status})`);
  const text = (await response.text()).trim();
  try {
    return sceneCardFromRecord(JSON.parse(text));
  } catch {
    return null;
  }
}

export async function uploadVideo(file: File, officerId?: string): Promise<SessionSummary> {
  const form = new FormData();
  form.append("file", file);
  if (officerId) form.append("officer_id", officerId);
  const response = await fetch(`${API_BASE}/sessions/upload`, { method: "POST", body: form });
  if (!response.ok) throw new Error(`Failed to upload video (${response.status})`);
  return response.json();
}

export function frameUrl(sessionId: string, index: number): string {
  return `${API_BASE}/sessions/${sessionId}/frames/${index}`;
}

export function audioUrl(sessionId: string): string {
  return `${API_BASE}/sessions/${sessionId}/audio`;
}

export function videoUrl(sessionId: string): string {
  return `${API_BASE}/sessions/${sessionId}/video`;
}
