"use client";

import { useCallback, useEffect, useState } from "react";
import { Download, Pencil, Play, RefreshCw, Trash2 } from "lucide-react";
import { PlaybackModal } from "@/components/PlaybackModal";
import { audioUrl, deleteSession, fetchSessions, frameUrl, renameSession } from "@/lib/api";
import type { SessionSummary } from "@/lib/types";

function formatDuration(session: SessionSummary): string {
  if (session.ended_at === null) return "—";
  const totalSeconds = Math.max(0, Math.round(session.ended_at - session.started_at));
  const minutes = Math.floor(totalSeconds / 60);
  return minutes > 0 ? `${minutes}m ${totalSeconds % 60}s` : `${totalSeconds}s`;
}

interface SessionCardProps {
  session: SessionSummary;
  onPlay: () => void;
  onRenamed: (updated: SessionSummary) => void;
  onDeleted: () => void;
}

function SessionCard({ session, onPlay, onRenamed, onDeleted }: SessionCardProps) {
  const [busy, setBusy] = useState(false);
  const lastFrameIndex = Math.max(0, session.frame_count - 1);
  const title = session.label ?? session.session_id.slice(0, 12);

  const handleRename = useCallback(async () => {
    const next = window.prompt("Session label", session.label ?? "");
    if (next === null || next.trim() === "") return;
    setBusy(true);
    try {
      onRenamed(await renameSession(session.session_id, next.trim()));
    } finally {
      setBusy(false);
    }
  }, [session.session_id, session.label, onRenamed]);

  const handleDelete = useCallback(async () => {
    if (!window.confirm(`Delete "${title}"? This cannot be undone.`)) return;
    setBusy(true);
    try {
      await deleteSession(session.session_id);
      onDeleted();
    } finally {
      setBusy(false);
    }
  }, [session.session_id, title, onDeleted]);

  return (
    <article className="group flex gap-3 rounded-xl border border-border bg-panel p-3 transition-colors hover:border-border-strong">
      <button
        onClick={onPlay}
        className="relative h-16 w-28 shrink-0 overflow-hidden rounded-lg bg-black"
        aria-label="Play session"
      >
        {session.frame_count > 0 && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={frameUrl(session.session_id, lastFrameIndex)}
            alt=""
            className="h-full w-full object-cover transition-opacity group-hover:opacity-50"
          />
        )}
        <span className="absolute inset-0 grid place-items-center">
          <span className="grid h-8 w-8 place-items-center rounded-full bg-black/50 text-white opacity-0 backdrop-blur-sm transition-opacity group-hover:opacity-100">
            <Play className="h-4 w-4 translate-x-px fill-current" />
          </span>
        </span>
      </button>

      <div className="flex min-w-0 flex-1 flex-col justify-between gap-1.5">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate text-sm font-medium">{title}</span>
          <span className="shrink-0 text-xs text-muted">{formatDuration(session)}</span>
        </div>

        <div className="flex items-center gap-1.5 text-xs text-muted">
          <span>{session.frame_count} frames</span>
          {session.event_count > 0 && <span>· {session.event_count} log entries</span>}
        </div>

        <div className="flex items-center gap-1">
          <IconAction onClick={onPlay} label="Play" icon={<Play className="h-3.5 w-3.5" />} />
          <IconAction
            onClick={handleRename}
            disabled={busy}
            label="Rename"
            icon={<Pencil className="h-3.5 w-3.5" />}
          />
          {session.has_audio && (
            <a
              href={audioUrl(session.session_id)}
              download
              title="Download audio"
              className="grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-panel-hover hover:text-fg"
            >
              <Download className="h-3.5 w-3.5" />
            </a>
          )}
          <IconAction
            onClick={handleDelete}
            disabled={busy}
            label="Delete"
            danger
            icon={<Trash2 className="h-3.5 w-3.5" />}
          />
        </div>
      </div>
    </article>
  );
}

interface IconActionProps {
  onClick: () => void;
  label: string;
  icon: React.ReactNode;
  disabled?: boolean;
  danger?: boolean;
}

function IconAction({ onClick, label, icon, disabled, danger }: IconActionProps) {
  const hover = danger ? "hover:text-critical" : "hover:text-fg";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      className={`grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-panel-hover disabled:opacity-40 ${hover}`}
    >
      {icon}
    </button>
  );
}

export function SessionLibrary() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<SessionSummary | null>(null);

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

  if (error) return <p className="text-sm text-critical">{error}</p>;
  if (sessions.length === 0) return null;

  return (
    <section className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">
          Recorded sessions
        </h2>
        <button
          onClick={() => void load()}
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted transition-colors hover:bg-panel-hover hover:text-fg"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        {sessions.map((session) => (
          <SessionCard
            key={session.session_id}
            session={session}
            onPlay={() => setPlaying(session)}
            onRenamed={(updated) =>
              setSessions((prev) =>
                prev.map((item) =>
                  item.session_id === updated.session_id ? updated : item,
                ),
              )
            }
            onDeleted={() =>
              setSessions((prev) =>
                prev.filter((item) => item.session_id !== session.session_id),
              )
            }
          />
        ))}
      </div>

      {playing && (
        <PlaybackModal session={playing} onClose={() => setPlaying(null)} />
      )}
    </section>
  );
}
