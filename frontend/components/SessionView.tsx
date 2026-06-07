"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Eye, FileText, RefreshCw, ScrollText } from "lucide-react";
import { GuidanceCard } from "@/components/GuidanceCard";
import { formatSpeaker } from "@/components/LogsPanel";
import {
  fetchReport,
  fetchSceneCard,
  fetchSession,
  generateReport,
  generateSceneCard,
  videoUrl,
} from "@/lib/api";
import type {
  GuidanceEvent,
  IncidentReport,
  RecordedEvent,
  SessionManifest,
} from "@/lib/types";

function formatClock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export function SessionView({ sessionId }: { sessionId: string }) {
  const [manifest, setManifest] = useState<SessionManifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [report, setReport] = useState<IncidentReport | null>(null);
  const [reportBusy, setReportBusy] = useState(false);
  const [panel, setPanel] = useState<"log" | "report" | "scene">("log");
  const [sceneCard, setSceneCard] = useState<string | null>(null); // null = not loaded, "" = none
  const [sceneBusy, setSceneBusy] = useState(false);
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const activeLogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetchSession(sessionId)
      .then(setManifest)
      .catch(() => setError("Could not load this session."));
  }, [sessionId]);

  // Keep the in-progress log entry scrolled into view.
  useEffect(() => {
    activeLogRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [currentTime]);

  const openReport = useCallback(async () => {
    setPanel("report");
    if (report) return;
    try {
      setReport(await fetchReport(sessionId));
    } catch {
      setReport(null); // no report yet — operator can generate one
    }
  }, [report, sessionId]);

  const regenerateReport = useCallback(async () => {
    setReportBusy(true);
    try {
      const result = await generateReport(sessionId);
      setReport(result.report);
    } finally {
      setReportBusy(false);
    }
  }, [sessionId]);

  const openSceneCard = useCallback(async () => {
    setPanel("scene");
    if (sceneCard !== null) return;
    try {
      setSceneCard((await fetchSceneCard(sessionId)) ?? "");
    } catch {
      setSceneCard("");
    }
  }, [sceneCard, sessionId]);

  const regenerateSceneCard = useCallback(async () => {
    setSceneBusy(true);
    try {
      setSceneCard(await generateSceneCard(sessionId));
    } catch {
      setError("Could not compile the scene card for this session.");
    } finally {
      setSceneBusy(false);
    }
  }, [sessionId]);

  const seekTo = useCallback((offsetSeconds: number) => {
    const video = videoElementRef.current;
    if (video) {
      video.currentTime = offsetSeconds;
      void video.play().catch(() => {});
    }
  }, []);

  const events = manifest?.events ?? [];
  const activeIndex = events.reduce(
    (latest, event, index) => (event.offset_seconds <= currentTime ? index : latest),
    -1,
  );
  const title = manifest?.label ?? sessionId.slice(0, 12);

  const toggle = (target: "report" | "scene", open: () => void) =>
    panel === target ? setPanel("log") : void open();

  return (
    <main className="flex h-screen flex-col">
      <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-border bg-bg/80 px-4 backdrop-blur-md md:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-xs text-muted transition-colors hover:border-border-strong hover:text-fg"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back
          </Link>
          <span className="mx-1 h-4 w-px bg-border" />
          <h1 className="truncate text-sm font-medium">{title}</h1>
          <span className="hidden font-mono text-[11px] text-muted sm:inline">
            {sessionId.slice(0, 12)}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <TabButton
            active={panel === "scene"}
            onClick={() => toggle("scene", openSceneCard)}
            icon={<ScrollText className="h-3.5 w-3.5" />}
            label="Scene card"
          />
          <TabButton
            active={panel === "report"}
            onClick={() => toggle("report", openReport)}
            icon={<FileText className="h-3.5 w-3.5" />}
            label="Report"
          />
        </div>
      </header>

      {error ? (
        <p className="p-6 text-sm text-critical">{error}</p>
      ) : (
        <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
          <div className="flex items-center justify-center bg-black p-2 md:p-4">
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video
              ref={videoElementRef}
              src={videoUrl(sessionId)}
              controls
              autoPlay
              onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
              className="max-h-full max-w-full rounded-lg"
            />
          </div>

          <div className="flex min-h-0 flex-col gap-2 overflow-y-auto border-t border-border p-4 lg:border-l lg:border-t-0">
            {panel === "report" ? (
              <ReportPanel report={report} busy={reportBusy} onRegenerate={regenerateReport} />
            ) : panel === "scene" ? (
              <SceneCardPanel card={sceneCard} busy={sceneBusy} onRegenerate={regenerateSceneCard} />
            ) : (
              <>
                <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">
                  Session log
                </h2>
                {events.length === 0 ? (
                  <p className="text-sm text-muted">No transcript or scene was recorded.</p>
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
    </main>
  );
}

function TabButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs transition-colors hover:bg-panel-hover ${
        active ? "bg-panel-hover text-fg" : "text-muted hover:text-fg"
      }`}
    >
      {icon}
      {active ? "Log" : label}
    </button>
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
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">
          Incident report
        </h2>
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
            <dd className="font-mono">{Math.round(report.duration_seconds)}s</dd>
            <dt className="text-muted">Frames</dt>
            <dd className="font-mono">{report.frame_count}</dd>
            <dt className="text-muted">Guidance</dt>
            <dd className="font-mono">{report.guidance.length}</dd>
          </dl>

          <section>
            <h3 className="mb-1 text-xs font-semibold uppercase tracking-[0.12em] text-muted">
              Guidance issued
            </h3>
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

          <section>
            <h3 className="mb-1 text-xs font-semibold uppercase tracking-[0.12em] text-muted">
              Transcript
            </h3>
            <pre className="whitespace-pre-wrap wrap-break-word rounded-md border border-border bg-panel p-2 text-xs leading-relaxed">
              {report.transcript || "(no transcript recorded)"}
            </pre>
          </section>
        </div>
      )}
    </>
  );
}

interface SceneCardPanelProps {
  card: string | null; // null = still loading, "" = none yet
  busy: boolean;
  onRegenerate: () => void;
}

function SceneCardPanel({ card, busy, onRegenerate }: SceneCardPanelProps) {
  return (
    <>
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-muted">Scene card</h2>
        <button
          onClick={onRegenerate}
          disabled={busy}
          title="Compile the scene card from the transcript + video analysis"
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted transition-colors hover:bg-panel-hover hover:text-fg disabled:opacity-40"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${busy ? "animate-spin" : ""}`} />
          {card ? "Regenerate" : "Generate"}
        </button>
      </div>
      <p className="text-[11px] text-muted">
        Factual context compiled from speech-to-text + video analysis. No legal opinion.
      </p>
      {card === null ? (
        <p className="text-sm text-muted">Loading…</p>
      ) : card === "" ? (
        <p className="text-sm text-muted">
          No scene card yet — generate one from this session&apos;s transcript and scene.
        </p>
      ) : (
        <pre className="whitespace-pre-wrap wrap-break-word rounded-md border border-border bg-panel p-2 text-xs leading-relaxed">
          {card}
        </pre>
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
  const stateClasses = active
    ? "border-accent bg-panel"
    : "border-transparent hover:border-border";

  return (
    <div
      ref={ref}
      onClick={onSeek}
      className={`cursor-pointer rounded-md border px-2 py-1.5 text-left transition-colors ${stateClasses}`}
    >
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
      ) : entry.kind === "detection" ? (
        <span className="inline-flex items-start gap-1.5 text-sm italic text-muted">
          <Eye className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          {(entry.payload as { summary?: string | null }).summary ?? "(scene)"}
        </span>
      ) : (
        <div className="mt-1">
          <GuidanceCard guidance={entry.payload as GuidanceEvent} acknowledgeable={false} />
        </div>
      )}
    </div>
  );
}
