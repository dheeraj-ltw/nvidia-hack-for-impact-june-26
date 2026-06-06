"use client";

import { useEffect, useRef } from "react";
import { MessageSquare, ShieldAlert } from "lucide-react";
import { GuidanceCard } from "@/components/GuidanceCard";
import type { GuidanceEvent, TranscriptEvent } from "@/lib/types";

interface LogsPanelProps {
  transcript: TranscriptEvent[];
  guidance: GuidanceEvent[];
  active: boolean;
}

function SectionHeader({ icon, label, count }: { icon: React.ReactNode; label: string; count?: number }) {
  return (
    <div className="flex items-center gap-2 px-0.5">
      {icon}
      <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</h2>
      {count !== undefined && count > 0 && (
        <span className="rounded-full bg-panel px-1.5 py-0.5 text-[10px] font-medium text-muted">
          {count}
        </span>
      )}
    </div>
  );
}

/** Live logs beside the feed: cited guidance on top, the transcript stream pinned below. */
export function LogsPanel({ transcript, guidance, active }: LogsPanelProps) {
  const transcriptScrollRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll the transcript *container only* — never the page (which would yank the
  // Stop button away). Skip if the user has scrolled up to read older lines.
  useEffect(() => {
    const container = transcriptScrollRef.current;
    if (!container) return;
    const distanceFromBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    if (distanceFromBottom < 80) {
      container.scrollTop = container.scrollHeight;
    }
  }, [transcript]);

  const hasGuidance = guidance.length > 0;
  const hasTranscript = transcript.length > 0;

  // Before a patrol starts and with nothing to show, present one clean prompt — not empty boxes.
  if (!active && !hasGuidance && !hasTranscript) {
    return (
      <aside className="flex h-full min-h-[16rem] flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-panel/40 p-6 text-center">
        <ShieldAlert className="h-6 w-6 text-muted" />
        <p className="text-sm font-medium text-fg">Guidance & transcript</p>
        <p className="max-w-[15rem] text-xs text-muted">
          Cited, law-aligned guidance and the live transcript will appear here once a patrol
          begins.
        </p>
      </aside>
    );
  }

  return (
    <aside className="flex h-full min-h-0 flex-col gap-4">
      {hasGuidance && (
        <section className="flex max-h-[55%] min-h-0 flex-col gap-2">
          <SectionHeader
            icon={<ShieldAlert className="h-3.5 w-3.5 text-muted" />}
            label="Guidance"
            count={guidance.length}
          />
          <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
            {guidance.map((event, index) => (
              <div key={`${event.ts}-${index}`} className="animate-in">
                <GuidanceCard guidance={event} />
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="flex min-h-0 flex-1 flex-col gap-2">
        <SectionHeader
          icon={<MessageSquare className="h-3.5 w-3.5 text-muted" />}
          label="Transcript"
        />
        <div
          ref={transcriptScrollRef}
          className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain rounded-xl border border-border bg-panel p-3"
        >
          {hasTranscript ? (
            <div className="mt-auto space-y-1.5 text-sm">
              {transcript.map((line, index) => (
                <p key={`${line.ts}-${index}`} className="animate-in leading-snug">
                  <span className="font-medium text-muted">{line.speaker}</span>
                  <span className="text-muted"> · </span>
                  {line.text}
                </p>
              ))}
            </div>
          ) : (
            <p className="flex items-center gap-2 text-sm text-muted">
              <span className="inline-flex gap-1">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted" />
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted [animation-delay:150ms]" />
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-muted [animation-delay:300ms]" />
              </span>
              Listening for speech…
            </p>
          )}
        </div>
      </section>
    </aside>
  );
}
