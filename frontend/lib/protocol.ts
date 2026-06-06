// Encodes outbound binary messages matching backend/app/realtime/session.py.
//   byte 0       : messageKind (0 = video JPEG, 1 = audio)
//   bytes 1..8   : float64 BE capture timestamp (seconds)
//   bytes 9..10  : uint16 width   (video only)
//   bytes 11..12 : uint16 height  (video only)
//   bytes 13..   : payload

export const MESSAGE_KIND_VIDEO = 0x00;
export const MESSAGE_KIND_AUDIO = 0x01;
const HEADER_LENGTH = 1 + 8 + 2 + 2;

function encodeMessage(
  messageKind: number,
  timestamp: number,
  width: number,
  height: number,
  payload: Uint8Array,
): ArrayBuffer {
  const buffer = new ArrayBuffer(HEADER_LENGTH + payload.byteLength);
  const header = new DataView(buffer);
  header.setUint8(0, messageKind);
  header.setFloat64(1, timestamp, false);
  header.setUint16(9, width, false);
  header.setUint16(11, height, false);
  new Uint8Array(buffer, HEADER_LENGTH).set(payload);
  return buffer;
}

export function encodeVideo(
  timestamp: number,
  width: number,
  height: number,
  jpeg: Uint8Array,
): ArrayBuffer {
  return encodeMessage(MESSAGE_KIND_VIDEO, timestamp, width, height, jpeg);
}

export function encodeAudio(timestamp: number, pcm: Uint8Array): ArrayBuffer {
  return encodeMessage(MESSAGE_KIND_AUDIO, timestamp, 0, 0, pcm);
}
