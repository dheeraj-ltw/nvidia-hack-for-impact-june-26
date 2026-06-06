import type { SessionSummary } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function fetchSessions(): Promise<SessionSummary[]> {
  const response = await fetch(`${API_BASE}/sessions`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load sessions (${response.status})`);
  return response.json();
}

export function frameUrl(sessionId: string, index: number): string {
  return `${API_BASE}/sessions/${sessionId}/frames/${index}`;
}

export function audioUrl(sessionId: string): string {
  return `${API_BASE}/sessions/${sessionId}/audio`;
}
