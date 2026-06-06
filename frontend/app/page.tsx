"use client";

import { useEffect, useState } from "react";
import { FeedView } from "@/components/FeedView";
import { LogsPanel } from "@/components/LogsPanel";
import { SessionLibrary } from "@/components/SessionLibrary";
import { StatusIndicator } from "@/components/StatusIndicator";
import { usePatrolSession } from "@/lib/usePatrolSession";

export default function PatrolConsole() {
  const { state, videoRef, start, stop, toggleMute } = usePatrolSession();
  const isLive = state.conn === "live";
  const isConnecting = state.conn === "connecting";

  // Reload the recorded-sessions library whenever a recording finishes.
  const [libraryKey, setLibraryKey] = useState(0);
  useEffect(() => {
    if (state.conn === "closed") setLibraryKey((key) => key + 1);
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

      {/* Live feed on the left, logs (guidance + transcript) on the right. */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.7fr_1fr]">
        <FeedView videoRef={videoRef} boxes={state.boxes} active={isLive} />
        <LogsPanel transcript={state.transcript} guidance={state.guidance} active={isLive} />
      </div>

      <SessionLibrary key={libraryKey} />
    </main>
  );
}
