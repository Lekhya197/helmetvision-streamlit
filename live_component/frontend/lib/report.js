// Owner-mode alert reports: the event live.js sends to Python with Streamlit's setComponentValue. Only
// used when Python passes report_alerts=true (owner mode, alerts toggle on); visitors never send any.
// Pure functions (no DOM), so node can test them.

export const THUMB_LONG_SIDE = 320;
export const THUMB_MAX_BYTES = 40000;
const QUALITIES = [0.8, 0.65, 0.5, 0.35];

let counter = 0;

/** Decoded size of a base64 string, in bytes. */
export function base64Bytes(b64) {
  const pad = b64.endsWith("==") ? 2 : b64.endsWith("=") ? 1 : 0;
  return Math.floor((b64.length * 3) / 4) - pad;
}

/** [width, height] fitted into longSide, never upscaled. */
export function thumbSize(w, h, longSide = THUMB_LONG_SIDE) {
  const r = Math.min(1, longSide / Math.max(w, h));
  return [Math.max(1, Math.round(w * r)), Math.max(1, Math.round(h * r))];
}

/**
 * Plain base64 of a JPEG of at most maxBytes, trying lower qualities in turn, or null.
 * encode(quality) returns a data URL, as canvas.toDataURL("image/jpeg", quality) does; a browser that
 * cannot make JPEGs falls back to PNG, which is refused here.
 */
export function smallJpegBase64(encode, maxBytes = THUMB_MAX_BYTES) {
  for (const q of QUALITIES) {
    const url = String(encode(q));
    if (!url.startsWith("data:image/jpeg")) return null;
    const b64 = url.slice(url.indexOf(",") + 1);
    if (base64Bytes(b64) <= maxBytes) return b64;
  }
  return null;
}

/** A unique id for one alert event (Python de-duplicates by it). */
export function eventId(now = Date.now(), rand = Math.random) {
  counter += 1;
  return `${now.toString(36)}-${rand().toString(36).slice(2, 8)}-${counter}`;
}

export function alertEvent({ id, conf, ts, thumb }) {
  return { id, conf: Math.round(conf * 1000) / 1000, ts, thumb_jpeg_base64: thumb ?? null };
}

/** The message Streamlit's component API expects for setComponentValue (JSON data). */
export function componentValueMessage(value) {
  return { isStreamlitMessage: true, type: "streamlit:setComponentValue", dataType: "json", value };
}
