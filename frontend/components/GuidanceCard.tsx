"use client";

import { useState } from "react";
import type { GuidanceEvent, Severity } from "@/lib/types";

const SEVERITY_COLORS: Record<Severity, string> = {
  info: "var(--color-accent)",
  caution: "var(--color-caution)",
  critical: "var(--color-critical)",
};

interface GuidanceCardProps {
  guidance: GuidanceEvent;
  /** Acknowledge is an operator action — hidden during playback of a past session. */
  acknowledgeable?: boolean;
}

export function GuidanceCard({ guidance, acknowledgeable = true }: GuidanceCardProps) {
  const [acknowledged, setAcknowledged] = useState(false);
  const accentColor = SEVERITY_COLORS[guidance.severity];

  return (
    <article
      className="rounded-lg border border-border bg-panel p-3 transition-opacity"
      style={{ borderLeft: `3px solid ${accentColor}`, opacity: acknowledged ? 0.55 : 1 }}
    >
      <div className="flex items-start justify-between gap-2">
        <span
          className="text-[10px] font-semibold uppercase tracking-wide"
          style={{ color: accentColor }}
        >
          {guidance.severity}
        </span>
        {acknowledgeable && (
          <button
            onClick={() => setAcknowledged(true)}
            disabled={acknowledged}
            className="shrink-0 rounded border border-border px-2 py-0.5 text-[11px] text-muted transition-colors hover:text-fg disabled:opacity-40"
          >
            {acknowledged ? "Acknowledged" : "Acknowledge"}
          </button>
        )}
      </div>

      <p className="mt-1 text-sm leading-snug">{guidance.suggestion}</p>
      {guidance.rationale && <p className="mt-1 text-xs text-muted">{guidance.rationale}</p>}

      {guidance.citations.length > 0 && (
        <ul className="mt-2 space-y-1 border-t border-border pt-2">
          {guidance.citations.map((citation, index) => (
            <li key={index} className="text-xs">
              <span className="font-medium">{citation.title}</span>{" "}
              <span className="text-muted">· {citation.reference}</span>
              {citation.snippet && <p className="mt-0.5 italic text-muted">{citation.snippet}</p>}
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}
