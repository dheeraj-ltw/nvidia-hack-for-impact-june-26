"use client";

import { useEffect, useState } from "react";
import { FeedView } from "@/components/FeedView";
import { GuidancePanel } from "@/components/GuidancePanel";
import { SessionsList } from "@/components/SessionsList";
import { StatusIndicator } from "@/components/StatusIndicator";
import { TranscriptStream } from "@/components/TranscriptStream";
import { usePatrolSession } from "@/lib/usePatrolSession";

export default function PatrolConsole() {
  const { state, videoRef, start, stop, toggleMute } = usePatrolSession();
  const isLive = state.conn === "live";
  const isConnecting = state.conn === "connecting";

  const hasTranscript = state.transcript.length > 0;
  const hasGuidance = state.guidance.length > 0;

  // Reload the recorded-sessions list whenever a recording finishes.
  const [sessionsKey, setSessionsKey] = useState(0);
  useEffect(() => {
    if (state.conn === "closed") setSessionsKey((key) => key + 1);
  }, [state.conn]);

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-5 p-4 md:p-6">
      <header className="flex items-center justify-between gap-4">
        <h1 className="text-base font-semibold tracking-tight">Patrol Assist</h1>
        <div className="flex items-center gap-3">
          <StatusIndicator conn={state.conn} framesSent={state.framesSent} />
          {isLive && (
            <button
              onClick={toggleMute}
              aria-pressed={state.muted}
              className="rounded-md border border-border px-2.5 py-1.5 text-sm text-muted transition-colors hover:text-fg"
            >
              {state.muted ? "Audio off" : "Audio on"}
            </button>
          )}
          <button
            onClick={isLive || isConnecting ? stop : start}
            disabled={isConnecting}
            className="rounded-md border border-border bg-panel px-3.5 py-1.5 text-sm font-medium transition-colors hover:border-accent disabled:opacity-50"
          >
            {isLive ? "Stop" : isConnecting ? "Connecting…" : "Start patrol"}
          </button>
        </div>
      </header>

      {state.error && (
        <div
          role="alert"
          className="rounded-md border border-critical bg-critical/10 px-3 py-2 text-sm text-critical"
        >
          {state.error}
        </div>
      )}

      {/* Feed is full-width until guidance arrives, then the layout splits in two. */}
      <div className={`grid grid-cols-1 gap-5 ${hasGuidance ? "lg:grid-cols-[2fr_1fr]" : ""}`}>
        <div className="flex flex-col gap-5">
          <FeedView videoRef={videoRef} boxes={state.boxes} active={isLive} />
          {hasTranscript && <TranscriptStream transcript={state.transcript} />}
        </div>
        {hasGuidance && <GuidancePanel guidance={state.guidance} />}
      </div>

      <SessionsList key={sessionsKey} />
    </main>
  );
}
