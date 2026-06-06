"use client";

import type { RefObject } from "react";
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
    <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-border bg-black">
      <video ref={videoRef} muted playsInline className="h-full w-full object-cover" />

      {!active && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-muted">
          Start a patrol to begin the live feed.
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
              className="absolute border-2 transition-all duration-200"
              style={{
                left: `${box.x * 100}%`,
                top: `${box.y * 100}%`,
                width: `${box.w * 100}%`,
                height: `${box.h * 100}%`,
                borderColor: color,
              }}
            >
              <span
                className="absolute -top-5 left-0 whitespace-nowrap rounded px-1 text-[10px] font-medium"
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
