"use client";

import { AlertTriangle, X } from "lucide-react";

interface AlertProps {
  message: string;
  onDismiss?: () => void;
}

export function Alert({ message, onDismiss }: AlertProps) {
  return (
    <div
      role="alert"
      className="animate-in flex items-start gap-3 rounded-lg border border-critical/40 bg-critical/10 px-4 py-3"
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-critical" />
      <p className="flex-1 text-sm text-fg">{message}</p>
      {onDismiss && (
        <button
          onClick={onDismiss}
          aria-label="Dismiss"
          className="shrink-0 rounded p-0.5 text-critical/70 transition-colors hover:text-critical"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}
