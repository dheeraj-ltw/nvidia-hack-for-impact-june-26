"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Mic, Plus, Square, Trash2, UserRound } from "lucide-react";
import { AlertDialog } from "@/components/ui/Modal";
import { Button } from "@/components/ui/Button";
import { createOfficer, deleteOfficer, fetchOfficers } from "@/lib/officers";
import type { OfficerSummary } from "@/lib/types";

interface OfficerOnboardingProps {
  /** Called once the user picks an officer to patrol as. */
  onSelect: (officer: OfficerSummary) => void;
}

const SAMPLE_HINT = "Read a sentence or two aloud (about 6–8 seconds) so we can learn your voice.";

/**
 * Pre-patrol gate: pick an enrolled officer, or add a new one by recording a short voice
 * sample. The sample is what lets the backend tell the officer apart from other speakers.
 */
export function OfficerOnboarding({ onSelect }: OfficerOnboardingProps) {
  const [officers, setOfficers] = useState<OfficerSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);

  const reload = useCallback(() => {
    fetchOfficers()
      .then(setOfficers)
      .catch(() => setError("Could not load the officer roster."));
  }, []);

  useEffect(reload, [reload]);

  const handleDelete = useCallback(
    async (officerId: string) => {
      try {
        await deleteOfficer(officerId);
        reload();
      } catch {
        setError("Could not remove that officer.");
      }
    },
    [reload],
  );

  return (
    <section className="mx-auto flex w-full max-w-md flex-col gap-4 rounded-xl border border-border bg-panel/40 p-6">
      <div className="flex flex-col items-center gap-1 text-center">
        <UserRound className="h-7 w-7 text-muted" />
        <h2 className="text-base font-semibold">Who&apos;s on patrol?</h2>
        <p className="text-xs text-muted">
          Pick yourself, or add a new officer. Your voice sample lets us label who&apos;s
          speaking.
        </p>
      </div>

      {error && <AlertDialog message={error} onDismiss={() => setError(null)} />}

      {adding ? (
        <AddOfficerForm
          onCancel={() => setAdding(false)}
          onCreated={(officer) => {
            setAdding(false);
            onSelect(officer);
          }}
        />
      ) : (
        <>
          {officers === null ? (
            <p className="flex items-center justify-center gap-2 py-4 text-sm text-muted">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading roster…
            </p>
          ) : officers.length === 0 ? (
            <p className="py-2 text-center text-sm text-muted">
              No officers yet. Add one to get started.
            </p>
          ) : (
            <ul className="flex flex-col gap-2">
              {officers.map((officer) => (
                <li key={officer.officer_id} className="flex items-center gap-2">
                  <button
                    onClick={() => onSelect(officer)}
                    className="flex flex-1 items-center gap-3 rounded-lg border border-border bg-panel px-3 py-2.5 text-left transition-colors hover:border-border-strong hover:bg-panel-hover"
                  >
                    <span className="grid h-8 w-8 place-items-center rounded-full bg-accent/15 text-accent">
                      <UserRound className="h-4 w-4" />
                    </span>
                    <span className="flex-1 truncate text-sm font-medium">{officer.name}</span>
                    {!officer.has_audio && (
                      <span className="text-[10px] uppercase text-muted">no voice</span>
                    )}
                  </button>
                  <button
                    onClick={() => handleDelete(officer.officer_id)}
                    aria-label={`Remove ${officer.name}`}
                    className="grid h-9 w-9 place-items-center rounded-lg text-muted transition-colors hover:bg-critical/10 hover:text-critical"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </li>
              ))}
            </ul>
          )}

          <Button variant="secondary" onClick={() => setAdding(true)} icon={<Plus className="h-4 w-4" />}>
            Add officer
          </Button>
        </>
      )}

      <p className="text-center text-[11px] text-muted">{SAMPLE_HINT}</p>
    </section>
  );
}

interface AddOfficerFormProps {
  onCancel: () => void;
  onCreated: (officer: OfficerSummary) => void;
}

type RecState = "idle" | "recording" | "recorded";

function AddOfficerForm({ onCancel, onCreated }: AddOfficerFormProps) {
  const [name, setName] = useState("");
  const [recState, setRecState] = useState<RecState>("idle");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const sampleRef = useRef<Blob | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  useEffect(
    () => () => {
      cleanupStream();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    },
    [cleanupStream, previewUrl],
  );

  const startRecording = useCallback(async () => {
    setError(null);
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setError("Microphone access was denied. Enable it and try again.");
      return;
    }
    streamRef.current = stream;
    chunksRef.current = [];
    const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      sampleRef.current = blob;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(URL.createObjectURL(blob));
      setRecState("recorded");
      cleanupStream();
    };
    recorderRef.current = recorder;
    recorder.start();
    setRecState("recording");
  }, [cleanupStream, previewUrl]);

  const stopRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  }, []);

  const submit = useCallback(async () => {
    if (!name.trim()) {
      setError("Please enter a name.");
      return;
    }
    if (!sampleRef.current) {
      setError("Please record a voice sample.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const officer = await createOfficer(name.trim(), sampleRef.current);
      onCreated(officer);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not enroll officer.");
      setSubmitting(false);
    }
  }, [name, onCreated]);

  return (
    <div className="flex flex-col gap-3">
      {error && <AlertDialog message={error} onDismiss={() => setError(null)} />}

      <label className="flex flex-col gap-1 text-xs font-medium text-muted">
        Name
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="e.g. PC Taylor"
          maxLength={120}
          className="h-9 rounded-lg border border-border bg-panel px-3 text-sm text-fg outline-none focus:border-accent"
        />
      </label>

      <div className="flex flex-col gap-2 rounded-lg border border-border bg-panel p-3">
        <span className="text-xs font-medium text-muted">Voice sample</span>
        {recState === "recording" ? (
          <Button variant="danger" onClick={stopRecording} icon={<Square className="h-3.5 w-3.5 fill-current" />}>
            Stop recording
          </Button>
        ) : (
          <Button
            variant="secondary"
            onClick={startRecording}
            icon={<Mic className="h-4 w-4" />}
          >
            {recState === "recorded" ? "Re-record" : "Record sample"}
          </Button>
        )}
        {recState === "recording" && (
          <p className="flex items-center gap-2 text-xs text-critical">
            <span className="h-2 w-2 animate-pulse rounded-full bg-critical" /> Recording…
          </p>
        )}
        {recState === "recorded" && previewUrl && (
          /* eslint-disable-next-line jsx-a11y/media-has-caption */
          <audio src={previewUrl} controls className="w-full" />
        )}
      </div>

      <div className="flex items-center justify-end gap-2">
        <Button variant="ghost" onClick={onCancel} disabled={submitting}>
          Cancel
        </Button>
        <Button
          variant="primary"
          onClick={submit}
          disabled={submitting}
          icon={submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : undefined}
        >
          {submitting ? "Enrolling" : "Save officer"}
        </Button>
      </div>
    </div>
  );
}
