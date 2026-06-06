"use client";

import { useCallback, useEffect, useState } from "react";
import { audioUrl, fetchSessions, frameUrl } from "@/lib/api";
import type { SessionSummary } from "@/lib/types";

function formatDuration(session: SessionSummary): string {
  if (session.ended_at === null) return "—";
  const seconds = Math.max(0, Math.round(session.ended_at - session.started_at));
  const minutes = Math.floor(seconds / 60);
  return minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
}

function SessionRow({ session }: { session: SessionSummary }) {
  const lastFrameIndex = Math.max(0, session.frame_count - 1);
  return (
    <article className="flex gap-3 rounded-lg border border-border bg-panel p-3">
      <div className="h-16 w-28 shrink-0 overflow-hidden rounded bg-black">
        {session.frame_count > 0 && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={frameUrl(session.session_id, lastFrameIndex)}
            alt="Last frame"
            className="h-full w-full object-cover"
          />
        )}
      </div>
      <div className="flex min-w-0 flex-1 flex-col justify-between">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate font-mono text-xs text-muted">
            {session.session_id.slice(0, 12)}
          </span>
          <span className="text-xs text-muted">{formatDuration(session)}</span>
        </div>
        <div className="flex items-center gap-3 text-xs text-muted">
          <span>{session.frame_count} frames</span>
          {session.has_audio && (
            <audio controls src={audioUrl(session.session_id)} className="h-7 max-w-[180px]" />
          )}
        </div>
      </div>
    </article>
  );
}

export function SessionsList() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setSessions(await fetchSessions());
      setError(null);
    } catch {
      setError("Could not load recorded sessions.");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (error) {
    return <p className="text-sm text-critical">{error}</p>;
  }

  if (sessions.length === 0) {
    return null;
  }

  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">
          Recorded sessions
        </h2>
        <button
          onClick={() => void load()}
          className="text-xs text-muted transition-colors hover:text-fg"
        >
          Refresh
        </button>
      </div>
      {sessions.map((session) => (
        <SessionRow key={session.session_id} session={session} />
      ))}
    </section>
  );
}
