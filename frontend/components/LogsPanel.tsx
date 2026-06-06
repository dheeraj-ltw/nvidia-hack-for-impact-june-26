"use client";

import { useEffect, useRef } from "react";
import { GuidanceCard } from "@/components/GuidanceCard";
import type { GuidanceEvent, TranscriptEvent } from "@/lib/types";

interface LogsPanelProps {
  transcript: TranscriptEvent[];
  guidance: GuidanceEvent[];
  active: boolean;
}

/** Live logs shown beside the feed: cited guidance on top, running transcript below. */
export function LogsPanel({ transcript, guidance, active }: LogsPanelProps) {
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [transcript]);

  return (
    <aside className="flex h-full min-h-0 flex-col gap-4">
      <section className="flex min-h-0 flex-1 flex-col gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">Guidance</h2>
        <div className="min-h-0 flex-1 space-y-2 overflow-y-auto">
          {guidance.length === 0 ? (
            <p className="text-sm text-muted">
              {active ? "Awaiting guidance…" : "Cited guidance appears here during a patrol."}
            </p>
          ) : (
            guidance.map((event, index) => (
              <GuidanceCard key={`${event.ts}-${index}`} guidance={event} />
            ))
          )}
        </div>
      </section>

      <section className="flex min-h-0 flex-1 flex-col gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">Transcript</h2>
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto rounded-lg border border-border bg-panel p-3 text-sm">
          {transcript.length === 0 ? (
            <p className="text-muted">{active ? "Listening…" : "No transcript yet."}</p>
          ) : (
            transcript.map((line, index) => (
              <p key={`${line.ts}-${index}`} className="leading-snug">
                <span className="text-muted">{line.speaker}: </span>
                {line.text}
              </p>
            ))
          )}
          <div ref={transcriptEndRef} />
        </div>
      </section>
    </aside>
  );
}
