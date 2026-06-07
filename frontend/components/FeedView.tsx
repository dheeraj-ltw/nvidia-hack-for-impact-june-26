"use client";

import type { RefObject } from "react";
import { Video } from "lucide-react";
import type { BoundingBox } from "@/lib/types";

interface FeedViewProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  boxes: BoundingBox[];
  active: boolean;
}

function isCriticalLabel(label: string): boolean {
  return label.includes("?") || label === "weapon";
}

/** Live video with a normalized bounding-box overlay scaled to the player size. */
export function FeedView({ videoRef, boxes, active }: FeedViewProps) {
  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-border bg-black lg:aspect-auto lg:h-full">
      <video ref={videoRef} muted playsInline className="h-full w-full object-cover" />

      {!active && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-linear-to-b from-panel/30 to-bg/60">
          <div className="grid h-12 w-12 place-items-center rounded-full border border-border bg-panel/80">
            <Video className="h-5 w-5 text-muted" />
          </div>
          <p className="text-sm font-medium text-fg">No active feed</p>
          <p className="text-xs text-muted">Start a patrol to begin the live feed.</p>
        </div>
      )}

      {active && (
        <div className="absolute left-3 top-3 flex items-center gap-1.5 rounded-full bg-black/60 px-2.5 py-1 backdrop-blur-sm">
          <span className="h-2 w-2 animate-pulse rounded-full bg-critical" />
          <span className="text-[11px] font-semibold uppercase tracking-wide text-white">Rec</span>
        </div>
      )}

      <div className="pointer-events-none absolute inset-0">
        {boxes.map((box, index) => {
          const color = isCriticalLabel(box.label)
            ? "var(--color-critical)"
            : "var(--color-accent)";
          return (
            <div
              key={`${box.label}-${index}`}
              className="absolute rounded-sm border-2 transition-all duration-200"
              style={{
                left: `${box.x * 100}%`,
                top: `${box.y * 100}%`,
                width: `${box.w * 100}%`,
                height: `${box.h * 100}%`,
                borderColor: color,
              }}
            >
              <span
                className="absolute -top-5 left-0 whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-semibold"
                style={{ background: color, color: "var(--color-bg)" }}
              >
                {box.label} {Math.round(box.confidence * 100)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
