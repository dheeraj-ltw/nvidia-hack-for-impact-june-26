"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Download, Loader2, Pencil, Play, RefreshCw, Trash2 } from "lucide-react";
import { ConfirmDialog, PromptDialog } from "@/components/ui/Modal";
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
  onRenamed: (updated: SessionSummary) => void;
  onDeleted: () => void;
}

function SessionCard({ session, onRenamed, onDeleted }: SessionCardProps) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const lastFrameIndex = Math.max(0, session.frame_count - 1);
  const title = session.label ?? session.session_id.slice(0, 12);
  const processing = session.status === "processing";
  const open = () => router.push(`/sessions/${session.session_id}`);

  const handleRename = useCallback(
    async (label: string) => {
      setRenaming(false);
      setBusy(true);
      try {
        onRenamed(await renameSession(session.session_id, label));
      } finally {
        setBusy(false);
      }
    },
    [session.session_id, onRenamed],
  );

  const handleDelete = useCallback(async () => {
    setConfirmingDelete(false);
    setBusy(true);
    try {
      await deleteSession(session.session_id);
      onDeleted();
    } finally {
      setBusy(false);
    }
  }, [session.session_id, onDeleted]);

  return (
    <>
      <article
        role="button"
        tabIndex={0}
        onClick={open}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            open();
          }
        }}
        className="group flex cursor-pointer gap-3 rounded-xl border border-border bg-panel p-3 transition-colors hover:border-border-strong hover:bg-panel-hover focus-visible:border-accent"
      >
        <div className="relative h-16 w-28 shrink-0 overflow-hidden rounded-lg bg-black">
          {session.frame_count > 0 && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={frameUrl(session.session_id, lastFrameIndex)}
              alt=""
              className="h-full w-full object-cover transition-opacity group-hover:opacity-60"
            />
          )}
          <span className="absolute inset-0 grid place-items-center">
            <span className="grid h-8 w-8 place-items-center rounded-full bg-black/55 text-white opacity-0 backdrop-blur-sm transition-opacity group-hover:opacity-100">
              <Play className="h-4 w-4 translate-x-px fill-current" />
            </span>
          </span>
        </div>

        <div className="flex min-w-0 flex-1 flex-col justify-between gap-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="truncate text-sm font-medium">{title}</span>
            {processing ? (
              <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-accent-soft px-2 py-0.5 text-[11px] font-medium text-accent">
                <Loader2 className="h-3 w-3 animate-spin" />
                Processing
              </span>
            ) : (
              <span className="shrink-0 font-mono text-xs text-muted">{formatDuration(session)}</span>
            )}
          </div>

          <div className="flex items-center gap-1.5 text-xs text-muted">
            {session.officer_name && <span className="truncate">{session.officer_name} ·</span>}
            <span className="font-mono">{session.frame_count}</span>
            <span>frames</span>
            {session.event_count > 0 && (
              <>
                <span>·</span>
                <span className="font-mono">{session.event_count}</span>
                <span>events</span>
              </>
            )}
          </div>

          <div className="flex items-center gap-1">
            <IconAction
              onClick={() => setRenaming(true)}
              disabled={busy}
              label="Rename"
              icon={<Pencil className="h-3.5 w-3.5" />}
            />
            {session.has_audio && (
              <a
                href={audioUrl(session.session_id)}
                download
                onClick={(event) => event.stopPropagation()}
                title="Download audio"
                className="grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-panel-hover hover:text-fg"
              >
                <Download className="h-3.5 w-3.5" />
              </a>
            )}
            <IconAction
              onClick={() => setConfirmingDelete(true)}
              disabled={busy}
              label="Delete"
              danger
              icon={<Trash2 className="h-3.5 w-3.5" />}
            />
          </div>
        </div>
      </article>

      {renaming && (
        <PromptDialog
          title="Rename session"
          label="Session label"
          initialValue={session.label ?? ""}
          placeholder={session.session_id.slice(0, 12)}
          maxLength={120}
          onSubmit={handleRename}
          onCancel={() => setRenaming(false)}
        />
      )}

      {confirmingDelete && (
        <ConfirmDialog
          title="Delete session"
          message={`Delete "${title}"? This cannot be undone.`}
          confirmLabel="Delete"
          danger
          onConfirm={handleDelete}
          onCancel={() => setConfirmingDelete(false)}
        />
      )}
    </>
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
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onClick();
      }}
      disabled={disabled}
      title={label}
      aria-label={label}
      className={`grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-panel-hover disabled:opacity-40 ${hover}`}
    >
      {icon}
    </button>
  );
}

interface SessionLibraryProps {
  /** A just-ended/uploaded session id to poll for until its tile appears. */
  awaitSessionId?: string | null;
}

export function SessionLibrary({ awaitSessionId }: SessionLibraryProps) {
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

  // Give the awaited session a generous window to show up, then stop chasing it.
  const awaitDeadline = useRef(0);
  useEffect(() => {
    awaitDeadline.current = Date.now() + 25000;
  }, [awaitSessionId]);

  const awaitingMissing =
    !!awaitSessionId &&
    Date.now() < awaitDeadline.current &&
    !sessions.some((session) => session.session_id === awaitSessionId);
  const anyProcessing = sessions.some((session) => session.status === "processing");

  // Self-rescheduling polls: each fires once, then re-runs after `load()` mutates `sessions`,
  // and stops automatically once nothing is awaited/processing.
  useEffect(() => {
    if (!awaitingMissing) return;
    const id = window.setTimeout(() => void load(), 1500);
    return () => window.clearTimeout(id);
  }, [awaitingMissing, sessions, load]);

  useEffect(() => {
    if (!anyProcessing) return;
    const id = window.setTimeout(() => void load(), 3000);
    return () => window.clearTimeout(id);
  }, [anyProcessing, sessions, load]);

  return (
    <div className="flex flex-col lg:h-full lg:min-h-0">
      <div className="flex shrink-0 items-center justify-between border-b border-border px-4 py-3">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">
          Recorded sessions
        </h2>
        <button
          onClick={() => void load()}
          aria-label="Refresh"
          className="grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-panel-hover hover:text-fg"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3">
        {error && <p className="px-1 text-sm text-critical">{error}</p>}

        {!error && sessions.length === 0 && !awaitingMissing && (
          <div className="px-2 py-12 text-center">
            <p className="text-sm font-medium text-fg">No recordings yet</p>
            <p className="mt-1 text-xs text-muted">Start a patrol or upload a video to begin.</p>
          </div>
        )}

        {sessions.map((session) => (
          <SessionCard
            key={session.session_id}
            session={session}
            onRenamed={(updated) =>
              setSessions((prev) =>
                prev.map((item) => (item.session_id === updated.session_id ? updated : item)),
              )
            }
            onDeleted={() =>
              setSessions((prev) => prev.filter((item) => item.session_id !== session.session_id))
            }
          />
        ))}

        {sessions.length === 0 && awaitingMissing && (
          <div className="flex items-center gap-2 rounded-xl border border-dashed border-border bg-panel/40 p-4 text-sm text-muted">
            <Loader2 className="h-4 w-4 animate-spin" />
            Finalizing recording…
          </div>
        )}
      </div>
    </div>
  );
}
