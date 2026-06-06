"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Eye, FileText, Loader2, RefreshCw, X } from "lucide-react";
import { GuidanceCard } from "@/components/GuidanceCard";
import { formatSpeaker } from "@/components/LogsPanel";
import { fetchReport, fetchSession, generateReport, videoUrl } from "@/lib/api";
import type {
  GuidanceEvent,
  IncidentReport,
  RecordedEvent,
  SessionManifest,
  SessionSummary,
} from "@/lib/types";

interface PlaybackModalProps {
  session: SessionSummary;
  onClose: () => void;
}

function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export function PlaybackModal({ session, onClose }: PlaybackModalProps) {
  const [manifest, setManifest] = useState<SessionManifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [report, setReport] = useState<IncidentReport | null>(null);
  const [showReport, setShowReport] = useState(false);
  const [reportBusy, setReportBusy] = useState(false);
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const activeLogRef = useRef<HTMLDivElement | null>(null);

  // Load the manifest, then poll while the post-session scene analysis is still pending
  // (frames recorded but no scene_summary yet) so the scene context appears when it lands.
  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const load = async () => {
      try {
        const next = await fetchSession(session.session_id);
        if (cancelled) return;
        setManifest(next);
        const scenePending = next.frame_count > 0 && next.scene_summary == null;
        if (scenePending) timer = window.setTimeout(load, 3000);
      } catch {
        if (!cancelled) setError("Could not load this session.");
      }
    };

    void load();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [session.session_id]);

  // Close on Escape.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Keep the in-progress log entry scrolled into view.
  useEffect(() => {
    activeLogRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [currentTime]);

  // Load the stored report lazily when the operator opens the report view; if none was
  // generated yet (or it's stale), the regenerate action builds + redispatches one.
  const openReport = useCallback(async () => {
    setShowReport(true);
    if (report) return;
    try {
      setReport(await fetchReport(session.session_id));
    } catch {
      setReport(null); // no report yet — operator can generate one
    }
  }, [report, session.session_id]);

  const regenerateReport = useCallback(async () => {
    setReportBusy(true);
    try {
      const result = await generateReport(session.session_id);
      setReport(result.report);
    } finally {
      setReportBusy(false);
    }
  }, [session.session_id]);

  const seekTo = useCallback((offsetSeconds: number) => {
    const video = videoElementRef.current;
    if (video) {
      video.currentTime = offsetSeconds;
      void video.play().catch(() => {});
    }
  }, []);

  const events = manifest?.events ?? [];
  // The latest event whose offset has passed is the "current" one.
  const activeIndex = events.reduce(
    (latest, event, index) => (event.offset_seconds <= currentTime ? index : latest),
    -1,
  );

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="animate-in flex max-h-[88vh] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-border-strong bg-bg shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-4 border-b border-border px-4 py-3">
          <h2 className="truncate text-sm font-medium">
            {session.label ?? session.session_id.slice(0, 12)}
          </h2>
          <div className="flex items-center gap-1">
            <button
              onClick={() => (showReport ? setShowReport(false) : void openReport())}
              className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs transition-colors hover:bg-panel-hover ${
                showReport ? "text-fg" : "text-muted hover:text-fg"
              }`}
            >
              <FileText className="h-3.5 w-3.5" />
              {showReport ? "Log" : "Report"}
            </button>
            <button
              onClick={onClose}
              aria-label="Close"
              className="grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-panel-hover hover:text-fg"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>

        {error ? (
          <p className="p-6 text-sm text-critical">{error}</p>
        ) : (
          <div className="grid min-h-0 flex-1 grid-cols-1 gap-0 md:grid-cols-[1.6fr_1fr]">
            <div className="flex items-center justify-center bg-black p-2">
              {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
              <video
                ref={videoElementRef}
                src={videoUrl(session.session_id)}
                controls
                autoPlay
                onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                className="max-h-[70vh] w-full rounded"
              />
            </div>

            <div className="flex min-h-0 flex-col gap-2 overflow-y-auto border-t border-border p-3 md:border-l md:border-t-0">
              {showReport ? (
                <ReportPanel report={report} busy={reportBusy} onRegenerate={regenerateReport} />
              ) : (
                <>
                  {manifest && manifest.frame_count > 0 && (
                    <SceneContext
                      summary={manifest.scene_summary}
                      onSeek={seekTo}
                    />
                  )}
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">
                    Session log
                  </h3>
                  {events.length === 0 ? (
                    <p className="text-sm text-muted">No transcript or guidance was recorded.</p>
                  ) : (
                    events.map((entry, index) => (
                      <LogEntry
                        key={index}
                        entry={entry}
                        active={index === activeIndex}
                        officerName={manifest?.officer_name}
                        ref={index === activeIndex ? activeLogRef : undefined}
                        onSeek={() => seekTo(entry.offset_seconds)}
                      />
                    ))
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/** Parse one "[12s] caption" scene line into its offset + text (offset null if unprefixed). */
function parseSceneLine(line: string): { offset: number | null; text: string } {
  const match = line.match(/^\[(\d+(?:\.\d+)?)s\]\s*(.*)$/);
  if (!match) return { offset: null, text: line };
  return { offset: Number(match[1]), text: match[2] };
}

interface SceneContextProps {
  // null/undefined => the post-session VLM pass hasn't finished yet; "" => ran, nothing to show.
  summary: string | null | undefined;
  onSeek: (offsetSeconds: number) => void;
}

function SceneContext({ summary, onSeek }: SceneContextProps) {
  const pending = summary == null;
  const lines = (summary ?? "").split("\n").map((l) => l.trim()).filter(Boolean);

  return (
    <section className="rounded-md border border-border bg-panel p-2.5">
      <h3 className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted">
        <Eye className="h-3.5 w-3.5" />
        Scene context
      </h3>
      {pending ? (
        <p className="flex items-center gap-2 text-sm text-muted">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Analyzing video…
        </p>
      ) : lines.length === 0 ? (
        <p className="text-sm text-muted">No scene description available.</p>
      ) : (
        <ul className="space-y-1">
          {lines.map((line, index) => {
            const { offset, text } = parseSceneLine(line);
            return (
              <li key={index} className="text-sm leading-relaxed">
                {offset != null && (
                  <button
                    onClick={() => onSeek(offset)}
                    className="mr-1.5 font-mono text-[11px] text-muted transition-colors hover:text-fg"
                  >
                    {formatClock(offset)}
                  </button>
                )}
                {text}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

interface ReportPanelProps {
  report: IncidentReport | null;
  busy: boolean;
  onRegenerate: () => void;
}

function ReportPanel({ report, busy, onRegenerate }: ReportPanelProps) {
  return (
    <>
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">
          Incident report
        </h3>
        <button
          onClick={onRegenerate}
          disabled={busy}
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted transition-colors hover:bg-panel-hover hover:text-fg disabled:opacity-40"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`} />
          {report ? "Regenerate" : "Generate"}
        </button>
      </div>

      {!report ? (
        <p className="text-sm text-muted">
          No report generated yet. Generate one to build it and dispatch the configured webhooks.
        </p>
      ) : (
        <div className="flex flex-col gap-3 text-sm">
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
            {report.officer_name && (
              <>
                <dt className="text-muted">Officer</dt>
                <dd>{report.officer_name}</dd>
              </>
            )}
            <dt className="text-muted">Duration</dt>
            <dd>{Math.round(report.duration_seconds)}s</dd>
            <dt className="text-muted">Frames</dt>
            <dd>{report.frame_count}</dd>
            <dt className="text-muted">Guidance</dt>
            <dd>{report.guidance.length}</dd>
          </dl>

          <section>
            <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">
              Guidance issued
            </h4>
            {report.guidance.length === 0 ? (
              <p className="text-xs text-muted">No guidance was surfaced during this session.</p>
            ) : (
              <ul className="space-y-1.5">
                {report.guidance.map((item, index) => (
                  <li key={index} className="rounded-md border border-border bg-panel p-2">
                    <p className="text-xs">{item.suggestion}</p>
                    {item.citations.length > 0 && (
                      <p className="mt-1 text-[11px] italic text-muted">
                        {item.citations.join(" · ")}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>

          {report.scene_summary && (
            <section>
              <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">
                Scene context
              </h4>
              <pre className="whitespace-pre-wrap wrap-break-word rounded-md border border-border bg-panel p-2 text-xs leading-relaxed">
                {report.scene_summary}
              </pre>
            </section>
          )}

          <section>
            <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted">
              Transcript
            </h4>
            <pre className="whitespace-pre-wrap wrap-break-word rounded-md border border-border bg-panel p-2 text-xs leading-relaxed">
              {report.transcript || "(no transcript recorded)"}
            </pre>
          </section>
        </div>
      )}
    </>
  );
}

interface LogEntryProps {
  entry: RecordedEvent;
  active: boolean;
  officerName?: string | null;
  onSeek: () => void;
  ref?: React.Ref<HTMLDivElement>;
}

function LogEntry({ entry, active, officerName, onSeek, ref }: LogEntryProps) {
  const stamp = formatClock(entry.offset_seconds);
  const baseClasses =
    "cursor-pointer rounded-md border px-2 py-1.5 text-left transition-colors";
  const stateClasses = active
    ? "border-accent bg-panel"
    : "border-transparent hover:border-border";

  return (
    <div ref={ref} onClick={onSeek} className={`${baseClasses} ${stateClasses}`}>
      <span className="mr-2 font-mono text-[11px] text-muted">{stamp}</span>
      {entry.kind === "transcript" ? (
        <span className="text-sm">
          {(() => {
            const line = entry.payload as { text: string; speaker?: string };
            return (
              <>
                {line.speaker && (
                  <span
                    className={
                      line.speaker === "officer"
                        ? "mr-1.5 font-medium text-accent"
                        : "mr-1.5 font-medium text-muted"
                    }
                  >
                    {formatSpeaker(line.speaker, officerName)}
                  </span>
                )}
                {line.text}
              </>
            );
          })()}
        </span>
      ) : (
        <div className="mt-1">
          <GuidanceCard guidance={entry.payload as GuidanceEvent} acknowledgeable={false} />
        </div>
      )}
    </div>
  );
}
