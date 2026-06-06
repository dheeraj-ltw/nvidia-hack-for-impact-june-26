"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { encodeAudio, encodeVideo } from "./protocol";
import type {
  BoundingBox,
  GuidanceEvent,
  PatrolEvent,
  TranscriptEvent,
} from "./types";

const WS_BASE = process.env.NEXT_PUBLIC_WS_BASE ?? "ws://localhost:8000";
const FRAME_INTERVAL_MS = 500; // ~2 fps — matches the backend analysis cadence
const AUDIO_CHUNK_MS = 1000; // emit one audio blob per second
const JPEG_QUALITY = 0.6;

export type ConnState = "idle" | "connecting" | "live" | "closed" | "error";

export interface PatrolState {
  conn: ConnState;
  backend: string | null;
  sessionId: string | null;
  framesSent: number;
  error: string | null;
  boxes: BoundingBox[];
  transcript: TranscriptEvent[];
  guidance: GuidanceEvent[];
  muted: boolean;
}

const INITIAL: PatrolState = {
  conn: "idle",
  backend: null,
  sessionId: null,
  framesSent: 0,
  error: null,
  boxes: [],
  transcript: [],
  guidance: [],
  muted: false,
};

function captureClockSeconds(): number {
  return performance.now() / 1000;
}

/**
 * Owns the camera stream, the WebSocket, and the frame/audio capture loops.
 * Attach `videoRef` to a <video> element; call start()/stop().
 */
export function usePatrolSession() {
  const [state, setState] = useState<PatrolState>(INITIAL);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const frameTimerRef = useRef<number | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const mutedRef = useRef(false);

  const handleEvent = useCallback((event: PatrolEvent) => {
    switch (event.type) {
      case "status":
        setState((prev) => ({
          ...prev,
          backend: event.detail ?? event.state,
          sessionId: event.session_id ?? prev.sessionId,
        }));
        break;
      case "detection":
        setState((prev) => ({ ...prev, boxes: event.boxes }));
        break;
      case "transcript":
        setState((prev) => ({ ...prev, transcript: [...prev.transcript.slice(-30), event] }));
        break;
      case "guidance":
        setState((prev) => ({ ...prev, guidance: [event, ...prev.guidance].slice(0, 20) }));
        break;
      case "speech": {
        if (mutedRef.current || !event.audio_b64) break;
        if (!ttsAudioRef.current) ttsAudioRef.current = new Audio();
        ttsAudioRef.current.src = `data:audio/mp3;base64,${event.audio_b64}`;
        void ttsAudioRef.current.play().catch(() => {});
        break;
      }
    }
  }, []);

  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const socket = socketRef.current;
    if (!video || !socket || socket.readyState !== WebSocket.OPEN || video.videoWidth === 0) return;

    if (!canvasRef.current) canvasRef.current = document.createElement("canvas");
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      async (blob) => {
        if (!blob || socket.readyState !== WebSocket.OPEN) return;
        const bytes = new Uint8Array(await blob.arrayBuffer());
        socket.send(encodeVideo(captureClockSeconds(), canvas.width, canvas.height, bytes));
        setState((prev) => ({ ...prev, framesSent: prev.framesSent + 1 }));
      },
      "image/jpeg",
      JPEG_QUALITY,
    );
  }, []);

  const stop = useCallback(() => {
    if (frameTimerRef.current) window.clearInterval(frameTimerRef.current);
    frameTimerRef.current = null;
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    recorderRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setState((prev) => ({ ...prev, conn: "closed" }));
  }, []);

  const startAudioRecorder = useCallback((stream: MediaStream, socket: WebSocket) => {
    const audioTracks = stream.getAudioTracks();
    if (audioTracks.length === 0 || typeof MediaRecorder === "undefined") return;

    const recorder = new MediaRecorder(new MediaStream(audioTracks), {
      mimeType: "audio/webm",
    });
    recorder.ondataavailable = async (recorderEvent) => {
      if (recorderEvent.data.size === 0 || socket.readyState !== WebSocket.OPEN) return;
      const bytes = new Uint8Array(await recorderEvent.data.arrayBuffer());
      socket.send(encodeAudio(captureClockSeconds(), bytes));
    };
    recorder.start(AUDIO_CHUNK_MS);
    recorderRef.current = recorder;
  }, []);

  const start = useCallback(async () => {
    setState({ ...INITIAL, conn: "connecting" });

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    } catch (error) {
      const name = error instanceof DOMException ? error.name : "";
      const message =
        name === "NotAllowedError"
          ? "Camera and microphone access was denied. Enable permissions and try again."
          : name === "NotFoundError"
            ? "No camera or microphone found on this device."
            : "Could not access the camera.";
      setState((prev) => ({ ...prev, conn: "error", error: message }));
      return;
    }

    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play().catch(() => {});
    }

    const socket = new WebSocket(`${WS_BASE}/ws/patrol`);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;
    socket.onopen = () => {
      setState((prev) => ({ ...prev, conn: "live", error: null }));
      frameTimerRef.current = window.setInterval(sendFrame, FRAME_INTERVAL_MS);
      startAudioRecorder(stream, socket);
    };
    socket.onmessage = (messageEvent) =>
      handleEvent(JSON.parse(messageEvent.data) as PatrolEvent);
    socket.onerror = () =>
      setState((prev) => ({ ...prev, conn: "error", error: "Connection to the server failed." }));
    socket.onclose = () =>
      setState((prev) => (prev.conn === "error" ? prev : { ...prev, conn: "closed" }));
  }, [handleEvent, sendFrame, startAudioRecorder]);

  const toggleMute = useCallback(() => {
    mutedRef.current = !mutedRef.current;
    setState((prev) => ({ ...prev, muted: mutedRef.current }));
  }, []);

  const dismissError = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  useEffect(() => () => stop(), [stop]);

  return { state, videoRef, start, stop, toggleMute, dismissError };
}
