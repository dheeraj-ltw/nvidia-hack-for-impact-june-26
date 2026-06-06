"use client";

import { useEffect, useRef } from "react";
import type { TranscriptEvent } from "@/lib/types";

interface TranscriptStreamProps {
  transcript: TranscriptEvent[];
}

/** Renders only when there is transcript to show; the parent gates on length. */
export function TranscriptStream({ transcript }: TranscriptStreamProps) {
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Keep the latest line in view as transcript grows.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [transcript]);

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">Transcript</h2>
      <div
        ref={scrollRef}
        className="max-h-40 space-y-1 overflow-y-auto rounded-lg border border-border bg-panel p-3 text-sm"
      >
        {transcript.map((line, index) => (
          <p key={`${line.ts}-${index}`} className="leading-snug">
            <span className="text-muted">{line.speaker}: </span>
            {line.text}
          </p>
        ))}
      </div>
    </section>
  );
}
