"use client";

import type { ConnState } from "@/lib/usePatrolSession";

const CONNECTION_LABELS: Record<ConnState, string> = {
  idle: "Ready",
  connecting: "Connecting",
  live: "Recording",
  closed: "Stopped",
  error: "Error",
};

const DOT_COLORS: Record<ConnState, string> = {
  idle: "bg-muted",
  connecting: "bg-caution",
  live: "bg-critical",
  closed: "bg-muted",
  error: "bg-critical",
};

interface StatusIndicatorProps {
  conn: ConnState;
  framesSent: number;
}

export function StatusIndicator({ conn, framesSent }: StatusIndicatorProps) {
  const isLive = conn === "live";
  return (
    <span className="flex items-center gap-1.5 text-xs text-muted">
      <span className="relative flex h-2 w-2">
        {isLive && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-critical opacity-60" />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${DOT_COLORS[conn]}`} />
      </span>
      {CONNECTION_LABELS[conn]}
      {isLive && <span className="opacity-60">· {framesSent} frames</span>}
    </span>
  );
}
