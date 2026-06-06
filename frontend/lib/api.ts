import type { SessionManifest, SessionSummary } from "./types";

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

export function frameUrl(sessionId: string, index: number): string {
  return `${API_BASE}/sessions/${sessionId}/frames/${index}`;
}

export function audioUrl(sessionId: string): string {
  return `${API_BASE}/sessions/${sessionId}/audio`;
}

export function videoUrl(sessionId: string): string {
  return `${API_BASE}/sessions/${sessionId}/video`;
}
