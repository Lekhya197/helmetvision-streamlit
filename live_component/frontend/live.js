// Live camera tab: getUserMedia -> in-browser detector -> tracker -> voter/suppressor -> canvas, banner,
// beep and an alert log. Nothing is uploaded or stored. Talks to Streamlit with a hand-written minimal
// component handshake (componentReady + setFrameHeight); render messages give the theme and
// args.report_alerts and args.include_thumb. Only when report_alerts is true (owner mode) does an alert
// go to Python with setComponentValue: {id, conf, ts, thumb_jpeg_base64}; the thumbnail (320 px,
// <= ~40 KB) only when include_thumb is true (Telegram toggle and Include photo on), otherwise null.
import { Detector, ORT_VERSION } from "./lib/detector.js";
import { alertEvent, componentValueMessage, eventId, smallJpegBase64, thumbSize } from "./lib/report.js";
import { IoUTracker } from "./lib/tracker.js";
import { AlertSuppressor, ViolationVoter, decide } from "./lib/voter.js";

const FRAME_HEIGHT = 760;
const MAX_LOG = 50;
const BANNER_MS = 3000;
const COLORS = { helmet_on: "#00c800", no_helmet: "#e60000" }; // by class name, never by id
const OTHER_COLOR = "#f5a623";

const $ = (id) => document.getElementById(id);
const els = {
  start: $("start"), stop: $("stop"), flip: $("flip"), sound: $("sound"), stats: $("stats"), msg: $("msg"),
  progress: $("progress"), progressFill: $("progressFill"), progressText: $("progressText"),
  view: $("view"), banner: $("banner"), log: $("log"), video: $("video"),
};
const ctx = els.view.getContext("2d");

// ---- Streamlit handshake ----
let reportAlerts = false; // false for visitors: then nothing is ever sent to Python
let includeThumb = false;
function post(type, extra = {}) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...extra }, "*");
}
function applyTheme(t) {
  const root = document.documentElement.style;
  if (t.backgroundColor) root.setProperty("--bg", t.backgroundColor);
  if (t.secondaryBackgroundColor) root.setProperty("--panel", t.secondaryBackgroundColor);
  if (t.textColor) root.setProperty("--fg", t.textColor);
  if (t.primaryColor) root.setProperty("--primary", t.primaryColor);
  if (t.font) root.setProperty("--font", t.font);
}
window.addEventListener("message", (ev) => {
  const d = ev.data;
  if (!d || d.type !== "streamlit:render") return; // everything else is ignored
  if (d.theme) applyTheme(d.theme);
  reportAlerts = d.args?.report_alerts === true;
  includeThumb = reportAlerts && d.args?.include_thumb === true;
  post("streamlit:setFrameHeight", { height: FRAME_HEIGHT });
});
post("streamlit:componentReady", { apiVersion: 1 });
post("streamlit:setFrameHeight", { height: FRAME_HEIGHT });

// ---- messages, progress, placeholder ----
function say(text, kind = "") {
  els.msg.className = kind;
  els.msg.textContent = text;
}
function progress(fraction, text) {
  els.progress.hidden = false;
  els.progress.classList.toggle("indeterminate", fraction === null);
  els.progressFill.style.width = fraction === null ? "" : `${Math.round(fraction * 100)}%`;
  els.progressText.textContent = text;
}
function hideProgress() {
  els.progress.hidden = true;
}
function placeholder(text) {
  const { width: w, height: h } = els.view;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#a3a8b8";
  ctx.font = "20px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  text.split("\n").forEach((line, i, all) => ctx.fillText(line, w / 2, h / 2 + (i - (all.length - 1) / 2) * 28));
}

// ---- state ----
let detector = null;
let loading = null;
let stream = null;
let running = false;
let facing = "user";
let tracker, voter, suppressor;
let lastTracked = [];
let bannerTimer = null;
let audio = null;
let runError = null;

function resetLogic() {
  tracker = new IoUTracker();
  voter = new ViolationVoter();
  suppressor = new AlertSuppressor();
  lastTracked = [];
}

function environmentProblem() {
  if (!window.isSecureContext) {
    return "The camera only works on a secure page. Open this app over https:// (or on localhost).";
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return "This browser does not give pages access to a camera.";
  }
  return null;
}

function cameraError(e) {
  const name = e && e.name;
  if (name === "NotAllowedError" || name === "SecurityError" || name === "PermissionDeniedError") {
    return "Camera permission was denied. Allow camera access for this site (the camera icon in the address bar or the browser settings), then press Start again.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError" || name === "OverconstrainedError") {
    return "No camera was found on this device.";
  }
  if (name === "NotReadableError" || name === "TrackStartError" || name === "AbortError") {
    return "The camera could not be started. It may be in use by another app or browser tab.";
  }
  return `The camera could not be started (${name || e}).`;
}

async function loadDetector() {
  if (detector) return detector;
  if (!loading) {
    loading = Detector.create({ progress, status: (t) => say(t) })
      .then((d) => { detector = d; return d; })
      .catch((e) => { loading = null; throw e; })
      .finally(hideProgress);
  }
  return loading;
}

async function openCamera() {
  const s = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: { ideal: facing } },
  });
  stream = s;
  els.video.srcObject = s;
  await els.video.play();
  els.view.width = els.video.videoWidth || 640;
  els.view.height = els.video.videoHeight || 480;
  try {
    const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    els.flip.hidden = cams.length < 2;
  } catch {
    els.flip.hidden = true;
  }
}

function closeCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  els.video.srcObject = null;
}

async function start() {
  const problem = environmentProblem();
  if (problem) { say(problem, "error"); return; }
  els.start.disabled = true;
  try {
    if (!audio && (window.AudioContext || window.webkitAudioContext)) {
      audio = new (window.AudioContext || window.webkitAudioContext)(); // created on this click
    }
    audio?.resume?.();
  } catch { audio = null; }
  try {
    say("Asking for camera access...");
    await openCamera();
  } catch (e) {
    say(cameraError(e), "error");
    els.start.disabled = false;
    return;
  }
  resetLogic();
  runError = null;
  running = true;
  els.stop.disabled = false;
  requestAnimationFrame(loop);
  try {
    if (!detector) say(`Loading the model and ONNX Runtime ${ORT_VERSION}. The camera preview runs meanwhile.`);
    await loadDetector();
    detector.resetTiming();
    const notes = detector.provider === "wasm" && detector.threads === 1 && !self.crossOriginIsolated
      ? " WASM runs on 1 thread here (the page is not cross-origin isolated)." : "";
    say(`Running in this browser with ${detector.provider.toUpperCase()}.${notes} Nothing is uploaded.`);
  } catch (e) {
    say(`The model could not be loaded: ${e?.message || e}. Check your connection and press Start again.`, "error");
    stop(false);
  }
}

function stop(clearMessage = true) {
  running = false;
  closeCamera();
  els.start.disabled = false;
  els.stop.disabled = true;
  els.flip.hidden = true;
  els.banner.hidden = true;
  placeholder("Camera stopped.\nPress Start camera to run again.");
  if (clearMessage) say("Stopped. The camera is off.");
}

async function flip() {
  facing = facing === "user" ? "environment" : "user";
  closeCamera();
  try {
    await openCamera();
    resetLogic();
  } catch (e) {
    say(cameraError(e), "error");
    stop(false);
  }
}

// ---- drawing ----
function drawBoxes(tracked) {
  const lw = Math.max(2, Math.round(els.view.height / 240));
  const fontPx = Math.max(14, Math.round(els.view.height / 30));
  ctx.lineWidth = lw;
  ctx.font = `${fontPx}px system-ui, sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  for (const t of tracked) {
    const [x1, y1, x2, y2] = t.box;
    const color = COLORS[t.name] || OTHER_COLOR;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `#${t.trackId} ${t.name} ${t.conf.toFixed(2)}`;
    const tw = ctx.measureText(label).width + 8;
    const ty = y1 - fontPx - 4 >= 0 ? y1 : y2 + fontPx + 4;
    ctx.fillStyle = color;
    ctx.fillRect(x1, ty - fontPx - 4, tw, fontPx + 4);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 4, ty - 2);
  }
}

function thumbnail(box) {
  const [x1, y1, x2, y2] = box;
  const side = Math.max(x2 - x1, y2 - y1) * 1.6;
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  const sx = Math.max(0, Math.min(els.view.width - side, cx - side / 2));
  const sy = Math.max(0, Math.min(els.view.height - side, cy - side / 2));
  const c = document.createElement("canvas");
  c.width = c.height = 96;
  c.getContext("2d").drawImage(els.view, sx, sy, Math.min(side, els.view.width), Math.min(side, els.view.height), 0, 0, 96, 96);
  return c.toDataURL("image/jpeg", 0.8); // stays in this page's memory only
}

function beep() {
  if (!audio || !els.sound.checked) return;
  const t0 = audio.currentTime;
  for (const [offset, freq] of [[0, 880], [0.18, 660]]) {
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.0001, t0 + offset);
    gain.gain.exponentialRampToValueAtTime(0.25, t0 + offset + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + offset + 0.15);
    osc.connect(gain).connect(audio.destination);
    osc.start(t0 + offset);
    osc.stop(t0 + offset + 0.16);
  }
}

function addAlert(t) {
  const time = new Date().toLocaleTimeString();
  els.banner.textContent = `ALERT: no helmet (track #${t.trackId})`;
  els.banner.hidden = false;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => { els.banner.hidden = true; }, BANNER_MS);
  beep();
  els.log.querySelector(".empty")?.remove();
  const card = document.createElement("div");
  card.className = "card";
  const img = document.createElement("img");
  img.src = thumbnail(t.box);
  img.alt = `track ${t.trackId}`;
  const cap = document.createElement("div");
  cap.textContent = `${time} #${t.trackId} ${t.conf.toFixed(2)}`;
  card.append(img, cap);
  els.log.prepend(card);
  while (els.log.children.length > MAX_LOG) els.log.lastChild.remove();
  reportAlert(t);
}

// Owner mode only: send the alert to Python, which shows its Telegram status and, with the toggle on,
// forwards it to Telegram under its rate limiter.
function makeThumb() {
  const [w, h] = thumbSize(els.view.width, els.view.height);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d").drawImage(els.view, 0, 0, w, h); // the frame with its boxes, drawn in handle()
  return smallJpegBase64((q) => c.toDataURL("image/jpeg", q));
}

function reportAlert(t) {
  if (!reportAlerts) return;
  try {
    const thumb = includeThumb ? makeThumb() : null;
    const value = alertEvent({ id: eventId(), conf: t.conf, ts: Date.now(), thumb });
    window.parent.postMessage(componentValueMessage(value), "*");
  } catch (e) {
    console.warn("alert report failed", e);
  }
}

// ---- main loop ----
function handle(dets) {
  if (!dets || !running) return; // null: a frame was skipped while the detector was busy
  const now = performance.now() / 1000;
  const ids = tracker.update(dets.map((d) => d.box));
  lastTracked = dets.map((d, i) => ({ trackId: ids[i], box: d.box, conf: d.conf, name: d.name }));
  const { fired } = decide(lastTracked, voter, suppressor, now);
  if (fired.length) {
    ctx.drawImage(els.video, 0, 0, els.view.width, els.view.height); // fresh frame for the thumbnails
    drawBoxes(lastTracked);
    fired.forEach(addAlert);
  }
  els.stats.textContent = `${detector.provider} | ${detector.ms.toFixed(0)} ms per frame | ${detector.fps.toFixed(1)} fps`;
}

function loop() {
  if (!running) return;
  const v = els.video;
  if (v.readyState >= 2 && v.videoWidth) {
    if (els.view.width !== v.videoWidth || els.view.height !== v.videoHeight) {
      els.view.width = v.videoWidth;
      els.view.height = v.videoHeight;
    }
    ctx.drawImage(v, 0, 0, els.view.width, els.view.height);
    drawBoxes(lastTracked);
    if (detector && !detector.busy && !runError) {
      detector.detect(v, v.videoWidth, v.videoHeight).then(handle).catch((e) => {
        runError = e;
        say(`Detection failed: ${e?.message || e}. Press Stop, then Start to retry.`, "error");
      });
    }
  }
  requestAnimationFrame(loop);
}

els.start.addEventListener("click", start);
els.stop.addEventListener("click", () => stop());
els.flip.addEventListener("click", flip);

placeholder("Press Start camera.\nDetection runs in this browser; video is not uploaded.");
const problem = environmentProblem();
if (problem) {
  say(problem, "error");
  els.start.disabled = true;
}
