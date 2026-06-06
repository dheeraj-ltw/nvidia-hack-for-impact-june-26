"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { GuidanceCard } from "@/components/GuidanceCard";
import { fetchSession, videoUrl } from "@/lib/api";
import type { GuidanceEvent, RecordedEvent, SessionManifest, SessionSummary } from "@/lib/types";

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
  const videoElementRef = useRef<HTMLVideoElement | null>(null);
  const activeLogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetchSession(session.session_id).then(setManifest).catch(() => {
      setError("Could not load this session.");
    });
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
          <button
            onClick={onClose}
            aria-label="Close"
            className="grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-panel-hover hover:text-fg"
          >
            <X className="h-4 w-4" />
          </button>
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
                    ref={index === activeIndex ? activeLogRef : undefined}
                    onSeek={() => seekTo(entry.offset_seconds)}
                  />
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface LogEntryProps {
  entry: RecordedEvent;
  active: boolean;
  onSeek: () => void;
  ref?: React.Ref<HTMLDivElement>;
}

function LogEntry({ entry, active, onSeek, ref }: LogEntryProps) {
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
        <span className="text-sm">{(entry.payload as { text: string }).text}</span>
      ) : (
        <div className="mt-1">
          <GuidanceCard guidance={entry.payload as GuidanceEvent} acknowledgeable={false} />
        </div>
      )}
    </div>
  );
}
