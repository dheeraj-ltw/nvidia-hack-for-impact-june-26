"use client";

import { useEffect, useRef, useState } from "react";
import { AlertTriangle, X } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface ModalProps {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
  /** Hide the header close (X) button — e.g. for a forced acknowledgement. */
  hideClose?: boolean;
}

/**
 * Centered modal shell matching PlaybackModal: dimmed backdrop, click-outside and Escape
 * to close, click-through guarded on the panel. Wrap dialog content in this.
 */
export function Modal({ title, onClose, children, hideClose }: ModalProps) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className="animate-in flex w-full max-w-md flex-col overflow-hidden rounded-xl border border-border-strong bg-bg shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-4 border-b border-border px-4 py-3">
          <h2 className="truncate text-sm font-medium">{title}</h2>
          {!hideClose && (
            <button
              onClick={onClose}
              aria-label="Close"
              className="grid h-7 w-7 place-items-center rounded-md text-muted transition-colors hover:bg-panel-hover hover:text-fg"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </header>
        {children}
      </div>
    </div>
  );
}

interface ConfirmDialogProps {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  /** Style the confirm button as destructive. */
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/** Replacement for window.confirm — returns the decision via onConfirm/onCancel. */
export function ConfirmDialog({
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  danger,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  return (
    <Modal title={title} onClose={onCancel}>
      <div className="flex flex-col gap-4 p-4">
        <p className="text-sm text-muted">{message}</p>
        <div className="flex items-center justify-end gap-2">
          <Button variant="ghost" onClick={onCancel}>
            {cancelLabel}
          </Button>
          <Button variant={danger ? "danger" : "primary"} onClick={onConfirm}>
            {confirmLabel}
          </Button>
        </div>
      </div>
    </Modal>
  );
}

interface PromptDialogProps {
  title: string;
  label?: string;
  initialValue?: string;
  placeholder?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  maxLength?: number;
  /** Called with the trimmed value; an empty value is treated as a cancel. */
  onSubmit: (value: string) => void;
  onCancel: () => void;
}

/** Replacement for window.prompt — single text input, submit on Enter. */
export function PromptDialog({
  title,
  label,
  initialValue = "",
  placeholder,
  confirmLabel = "Save",
  cancelLabel = "Cancel",
  maxLength,
  onSubmit,
  onCancel,
}: PromptDialogProps) {
  const [value, setValue] = useState(initialValue);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Focus and select the text on open, so renaming is one keystroke away.
  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  const submit = () => {
    const trimmed = value.trim();
    if (trimmed) onSubmit(trimmed);
    else onCancel();
  };

  return (
    <Modal title={title} onClose={onCancel}>
      <div className="flex flex-col gap-4 p-4">
        <label className="flex flex-col gap-1.5 text-xs font-medium text-muted">
          {label}
          <input
            ref={inputRef}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submit();
            }}
            placeholder={placeholder}
            maxLength={maxLength}
            className="h-9 rounded-lg border border-border bg-panel px-3 text-sm text-fg outline-none focus:border-accent"
          />
        </label>
        <div className="flex items-center justify-end gap-2">
          <Button variant="ghost" onClick={onCancel}>
            {cancelLabel}
          </Button>
          <Button variant="primary" onClick={submit}>
            {confirmLabel}
          </Button>
        </div>
      </div>
    </Modal>
  );
}

interface AlertDialogProps {
  message: string;
  title?: string;
  dismissLabel?: string;
  onDismiss: () => void;
}

/** Modal error dialog — the blocking counterpart to the old inline Alert banner. */
export function AlertDialog({
  message,
  title = "Something went wrong",
  dismissLabel = "Dismiss",
  onDismiss,
}: AlertDialogProps) {
  return (
    <Modal title={title} onClose={onDismiss}>
      <div className="flex flex-col gap-4 p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-critical" />
          <p className="flex-1 text-sm text-fg">{message}</p>
        </div>
        <div className="flex items-center justify-end">
          <Button variant="primary" onClick={onDismiss}>
            {dismissLabel}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
