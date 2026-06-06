import type { OfficerSummary } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function fetchOfficers(): Promise<OfficerSummary[]> {
  const response = await fetch(`${API_BASE}/officers`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load officers (${response.status})`);
  return response.json();
}

export async function createOfficer(
  name: string,
  audio: Blob,
): Promise<OfficerSummary> {
  const form = new FormData();
  form.append("name", name);
  form.append("audio", audio, "reference.webm");
  const response = await fetch(`${API_BASE}/officers`, { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Failed to enroll officer (${response.status})`);
  }
  return response.json();
}

export async function deleteOfficer(officerId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/officers/${officerId}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Failed to delete officer (${response.status})`);
}

export function officerAudioUrl(officerId: string): string {
  return `${API_BASE}/officers/${officerId}/audio`;
}
