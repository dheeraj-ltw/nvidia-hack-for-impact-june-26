"use client";

import { useEffect, useState } from "react";
import { Loader2, Play, Square, Volume2, VolumeX } from "lucide-react";
import { FeedView } from "@/components/FeedView";
import { LogsPanel } from "@/components/LogsPanel";
import { SessionLibrary } from "@/components/SessionLibrary";
import { StatusIndicator } from "@/components/StatusIndicator";
import { Alert } from "@/components/ui/Alert";
import { Button } from "@/components/ui/Button";
import { usePatrolSession } from "@/lib/usePatrolSession";

export default function PatrolConsole() {
  const { state, videoRef, start, stop, toggleMute, dismissError } = usePatrolSession();
  const isLive = state.conn === "live";
  const isConnecting = state.conn === "connecting";

  // Reload the recorded-sessions library whenever a recording finishes.
  const [libraryKey, setLibraryKey] = useState(0);
  useEffect(() => {
    if (state.conn === "closed") setLibraryKey((key) => key + 1);
  }, [state.conn]);

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 p-4 md:p-6">
      <header className="flex items-center justify-between gap-4 border-b border-border pb-4">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-semibold tracking-tight">Patrol Assist</h1>
          <StatusIndicator conn={state.conn} framesSent={state.framesSent} />
        </div>

        <div className="flex items-center gap-2">
          {isLive && (
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleMute}
              aria-pressed={state.muted}
              icon={
                state.muted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />
              }
            >
              {state.muted ? "Muted" : "Audio"}
            </Button>
          )}

          {isLive || isConnecting ? (
            <Button
              variant="danger"
              onClick={stop}
              disabled={isConnecting}
              icon={
                isConnecting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Square className="h-3.5 w-3.5 fill-current" />
                )
              }
            >
              {isConnecting ? "Connecting" : "Stop patrol"}
            </Button>
          ) : (
            <Button variant="primary" onClick={start} icon={<Play className="h-4 w-4 fill-current" />}>
              Start patrol
            </Button>
          )}
        </div>
      </header>

      {state.error && <Alert message={state.error} onDismiss={dismissError} />}

      {/* Live feed on the left, logs (guidance + transcript) on the right. */}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.7fr_1fr]">
        <FeedView videoRef={videoRef} boxes={state.boxes} active={isLive} />
        <LogsPanel transcript={state.transcript} guidance={state.guidance} active={isLive} />
      </div>

      <SessionLibrary key={libraryKey} />
    </main>
  );
}
