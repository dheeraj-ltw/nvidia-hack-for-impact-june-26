"use client";

import { useEffect, useState } from "react";
import { Loader2, Play, Square, UserRound, Volume2, VolumeX } from "lucide-react";
import { FeedView } from "@/components/FeedView";
import { LogsPanel } from "@/components/LogsPanel";
import { OfficerOnboarding } from "@/components/OfficerOnboarding";
import { SessionLibrary } from "@/components/SessionLibrary";
import { StatusIndicator } from "@/components/StatusIndicator";
import { Alert } from "@/components/ui/Alert";
import { Button } from "@/components/ui/Button";
import { usePatrolSession } from "@/lib/usePatrolSession";
import type { OfficerSummary } from "@/lib/types";

const OFFICER_STORAGE_KEY = "patrol.officer";

export default function PatrolConsole() {
  const { state, videoRef, start, stop, toggleMute, dismissError } = usePatrolSession();
  const isLive = state.conn === "live";
  const isConnecting = state.conn === "connecting";

  // The officer on patrol. Restored from localStorage so onboarding isn't repeated each visit.
  const [officer, setOfficer] = useState<OfficerSummary | null>(null);
  useEffect(() => {
    const stored = window.localStorage.getItem(OFFICER_STORAGE_KEY);
    if (stored) {
      try {
        setOfficer(JSON.parse(stored) as OfficerSummary);
      } catch {
        window.localStorage.removeItem(OFFICER_STORAGE_KEY);
      }
    }
  }, []);

  const selectOfficer = (next: OfficerSummary | null) => {
    setOfficer(next);
    if (next) window.localStorage.setItem(OFFICER_STORAGE_KEY, JSON.stringify(next));
    else window.localStorage.removeItem(OFFICER_STORAGE_KEY);
  };

  // Reload the recorded-sessions library whenever a recording finishes.
  const [libraryKey, setLibraryKey] = useState(0);
  useEffect(() => {
    if (state.conn === "closed") setLibraryKey((key) => key + 1);
  }, [state.conn]);

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-4 pb-6 md:px-6">
      <header className="sticky top-0 z-30 -mx-4 flex items-center justify-between gap-4 border-b border-border bg-bg/90 px-4 pb-4 pt-4 backdrop-blur-sm md:-mx-6 md:px-6">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-semibold tracking-tight">Patrol Assist</h1>
          <StatusIndicator conn={state.conn} framesSent={state.framesSent} />
        </div>

        <div className="flex items-center gap-2">
          {officer && !isLive && !isConnecting && (
            <button
              onClick={() => selectOfficer(null)}
              className="flex items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-xs text-muted transition-colors hover:border-border-strong hover:text-fg"
              title="Switch officer"
            >
              <UserRound className="h-3.5 w-3.5" />
              {officer.name}
            </button>
          )}

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
            officer && (
              <Button
                variant="primary"
                onClick={() => start(officer.officer_id)}
                icon={<Play className="h-4 w-4 fill-current" />}
              >
                Start patrol
              </Button>
            )
          )}
        </div>
      </header>

      {state.error && <Alert message={state.error} onDismiss={dismissError} />}

      {/* Before a patrol, gate the feed/logs on choosing who's on patrol so we can identify
          speakers. The recorded-sessions library stays visible regardless. */}
      {!officer && state.conn !== "live" && state.conn !== "connecting" ? (
        <div className="grid place-items-center py-10">
          <OfficerOnboarding onSelect={selectOfficer} />
        </div>
      ) : (
        /* Live feed on the left, logs (guidance + transcript) on the right. */
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.7fr_1fr]">
          <FeedView videoRef={videoRef} boxes={state.boxes} active={isLive} />
          <LogsPanel
            transcript={state.transcript}
            guidance={state.guidance}
            active={isLive}
            officerName={officer?.name}
          />
        </div>
      )}

      <SessionLibrary key={libraryKey} />
    </main>
  );
}
