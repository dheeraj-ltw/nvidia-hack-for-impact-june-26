"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Image from "next/image";
import {
  Loader2,
  Play,
  Square,
  Upload,
  UserRound,
  Volume2,
  VolumeX,
} from "lucide-react";
import { FeedView } from "@/components/FeedView";
import { LogsPanel } from "@/components/LogsPanel";
import { OfficerOnboarding } from "@/components/OfficerOnboarding";
import { SessionLibrary } from "@/components/SessionLibrary";
import { StatusIndicator } from "@/components/StatusIndicator";
import { AlertDialog } from "@/components/ui/Modal";
import { Button } from "@/components/ui/Button";
import { uploadVideo } from "@/lib/api";
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

  // Reload the library after a recording finishes / upload starts, and tell it which session to
  // wait for so its tile reliably appears (the manifest write can lag the post-stop fetch).
  const [libraryKey, setLibraryKey] = useState(0);
  const [awaitSessionId, setAwaitSessionId] = useState<string | null>(null);
  useEffect(() => {
    if (state.conn === "closed") {
      setAwaitSessionId(state.sessionId);
      setLibraryKey((key) => key + 1);
    }
  }, [state.conn, state.sessionId]);

  // Upload a recorded video for processing (same pipeline as a live patrol).
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const onUploadPicked = useCallback(
    async (file: File | undefined) => {
      if (!file) return;
      setUploadBusy(true);
      setUploadError(null);
      try {
        const summary = await uploadVideo(file, officer?.officer_id);
        setAwaitSessionId(summary.session_id);
        setLibraryKey((key) => key + 1);
      } catch {
        setUploadError("Could not process that video. Use an .mp4, .mov or .mkv file.");
      } finally {
        setUploadBusy(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    },
    [officer?.officer_id],
  );

  const showConsole = officer || isLive || isConnecting;

  return (
    <div className="flex min-h-screen flex-col lg:h-screen lg:overflow-hidden">
      {/* Top bar — brand + live status (left), patrol controls (right). */}
      <header className="flex h-16 shrink-0 items-center justify-between gap-4 border-b border-border bg-bg px-4 md:px-6">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 shrink-0 place-items-center overflow-hidden rounded-lg border border-border-strong bg-panel">
              <Image
                src="/jarvis-logo.svg"
                alt="JARVIS"
                width={36}
                height={44}
                priority
                className="h-9 w-auto"
              />
            </div>
            <div className="flex min-w-0 flex-col leading-tight">
              <span className="text-xl font-semibold tracking-tight">JARVIS</span>
              <span
                className="hidden truncate text-sm text-muted lg:block"
                title="Judicial Advisor & Real-time Voice Intelligence System"
              >
                Judicial Advisor &amp; Real-time Voice Intelligence System
              </span>
            </div>
          </div>
          <span className="mx-1 hidden h-6 w-px bg-border sm:block" />
          <StatusIndicator conn={state.conn} framesSent={state.framesSent} />
        </div>

        <div className="flex items-center gap-2">
          {officer && !isLive && !isConnecting && (
            <button
              onClick={() => selectOfficer(null)}
              className="hidden items-center gap-1.5 rounded-lg border border-border px-2.5 py-1.5 text-xs text-muted transition-colors hover:border-border-strong hover:text-fg sm:flex"
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
              icon={state.muted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
            >
              {state.muted ? "Muted" : "Audio"}
            </Button>
          )}

          {!isLive && !isConnecting && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/quicktime,video/x-matroska,.mp4,.mov,.mkv"
                className="hidden"
                onChange={(event) => void onUploadPicked(event.target.files?.[0])}
              />
              <Button
                variant="secondary"
                size="md"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploadBusy}
                icon={
                  uploadBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />
                }
              >
                {uploadBusy ? "Uploading" : "Upload"}
              </Button>
            </>
          )}

          {isLive || isConnecting ? (
            <Button
              variant="danger"
              onClick={stop}
              disabled={isConnecting}
              className="bg-critical/10 text-critical hover:bg-critical/20"
              icon={
                isConnecting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Square className="h-3.5 w-3.5 fill-current" />
                )
              }
            >
              {isConnecting ? "Connecting" : "End patrol"}
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

      {(state.error || uploadError) && (
        <AlertDialog
          message={state.error ?? uploadError ?? ""}
          onDismiss={() => {
            dismissError();
            setUploadError(null);
          }}
        />
      )}

      {/* Body — recordings rail (left on desktop, below on mobile) + live console. */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <main className="order-1 flex min-w-0 flex-1 flex-col p-4 md:p-5 lg:order-2 lg:min-h-0 lg:overflow-hidden">
          {!showConsole ? (
            <div className="grid h-full place-items-center">
              <OfficerOnboarding onSelect={selectOfficer} />
            </div>
          ) : (
            <section className="grid h-full min-h-[24rem] gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
              <FeedView videoRef={videoRef} boxes={state.boxes} active={isLive} />
              <LogsPanel
                transcript={state.transcript}
                guidance={state.guidance}
                active={isLive}
                officerName={officer?.name}
              />
            </section>
          )}
        </main>

        <aside className="order-2 flex flex-col border-t border-border bg-panel/20 lg:order-1 lg:w-[320px] lg:shrink-0 lg:border-r lg:border-t-0 lg:overflow-hidden xl:w-[360px]">
          <SessionLibrary key={libraryKey} awaitSessionId={awaitSessionId} />
        </aside>
      </div>
    </div>
  );
}
