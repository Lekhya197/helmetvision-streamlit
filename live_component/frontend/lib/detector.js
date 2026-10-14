// In-browser helmet detector: onnxruntime-web + the pre/postprocessing of scripts/onnx_ref.py.
//
// The pure functions (roundHalfEven .. postprocess) mirror onnx_ref.py line by line and have no DOM, so
// node --test can check them. Detector (bottom) is the browser part: it loads ONNX Runtime from jsDelivr,
// tries the WebGPU provider and then WASM, and runs at most one inference at a time.
//
// Model contract (scripts/export_onnx.py): input "images" [1, 3, 416, 416] float32 RGB 0..1, letterboxed
// with gray 114; output raw [1, 4 + nc, N] (cx, cy, w, h in letterbox pixels, then one score per class),
// NMS done here. An end-to-end [1, N, 6] output is also understood, as in onnx_ref.py.

export const ORT_VERSION = "1.30.0"; // pinned; checked on jsDelivr (HTTP 200) 2026-10-14
export const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
export const ORT_URL = `${ORT_BASE}ort.webgpu.min.mjs`; // has both the webgpu and the wasm providers
export const PROVIDERS = ["webgpu", "wasm"];

export const IMGSZ = 416;
export const PAD_VALUE = 114;
export const DETECT_CONF = 0.25;
export const NMS_IOU = 0.7;

// Python's round(): halves go to the even neighbor (round(0.5) == 0, round(1.5) == 2).
export function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// onnx_ref.letterbox_params: scale, resized size and padding, as Ultralytics LetterBox.
export function letterboxParams(h, w, size = IMGSZ) {
  const r = Math.min(size / h, size / w);
  const newW = roundHalfEven(w * r);
  const newH = roundHalfEven(h * r);
  const dw = (size - newW) / 2;
  const dh = (size - newH) / 2;
  return {
    r, newW, newH,
    left: roundHalfEven(dw - 0.1), right: roundHalfEven(dw + 0.1),
    top: roundHalfEven(dh - 0.1), bottom: roundHalfEven(dh + 0.1),
  };
}

const INV255 = Float32Array.from({ length: 256 }, (_, v) => v / 255);

// RGBA pixels of the size x size letterboxed canvas -> NCHW float32 RGB 0..1 (onnx_ref.letterbox's last
// step; the canvas already holds the resized frame on gray 114).
export function rgbaToNCHW(rgba, size = IMGSZ) {
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    out[i] = INV255[rgba[p]];
    out[n + i] = INV255[rgba[p + 1]];
    out[2 * n + i] = INV255[rgba[p + 2]];
  }
  return out;
}

// onnx_ref.boxes_to_original: letterbox-pixel xyxy boxes -> original pixels, clipped.
export function boxesToOriginal(boxes, scale, left, top, origW, origH) {
  const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
  return boxes.map(([x1, y1, x2, y2]) => [
    clip((x1 - left) / scale, origW), clip((y1 - top) / scale, origH),
    clip((x2 - left) / scale, origW), clip((y2 - top) / scale, origH),
  ]);
}

// onnx_ref.detect_layout
export function detectLayout(dims, nc) {
  if (dims.length === 3 && dims[0] === 1 && dims[2] === 6) return "end2end";
  if (dims.length === 3 && dims[0] === 1 && dims[1] === 4 + nc) return "raw";
  throw new Error(`unknown output shape [${dims}] for ${nc} classes`);
}

function area(b) {
  return (b[2] - b[0]) * (b[3] - b[1]);
}

// onnx_ref.nms: indices kept by greedy NMS, highest score first.
export function nms(boxes, scores, iouThres) {
  let order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const keep = [];
  while (order.length) {
    const i = order[0];
    keep.push(i);
    order = order.slice(1).filter((j) => {
      const iw = Math.max(0, Math.min(boxes[i][2], boxes[j][2]) - Math.max(boxes[i][0], boxes[j][0]));
      const ih = Math.max(0, Math.min(boxes[i][3], boxes[j][3]) - Math.max(boxes[i][1], boxes[j][1]));
      const inter = iw * ih;
      const union = area(boxes[i]) + area(boxes[j]) - inter;
      const iou = union > 0 ? inter / union : 0;
      return iou <= iouThres;
    });
  }
  return keep;
}

// onnx_ref.postprocess: flat output data + dims -> {boxes (letterbox px), scores, cls}, best first.
export function postprocess(data, dims, nc, conf = DETECT_CONF, iouThres = NMS_IOU) {
  const layout = detectLayout(dims, nc);
  let boxes = [], scores = [], cls = [];
  if (layout === "end2end") {
    const n = dims[1];
    for (let i = 0; i < n; i++) {
      const o = i * 6;
      boxes.push([data[o], data[o + 1], data[o + 2], data[o + 3]]);
      scores.push(data[o + 4]);
      cls.push(Math.round(data[o + 5]));
    }
  } else {
    const n = dims[2]; // data is [4 + nc][n], row-major
    for (let i = 0; i < n; i++) {
      let best = 0, bestScore = data[4 * n + i];
      for (let c = 1; c < nc; c++) {
        const s = data[(4 + c) * n + i];
        if (s > bestScore) { best = c; bestScore = s; } // first max wins, as numpy argmax
      }
      const cx = data[i], cy = data[n + i], w = data[2 * n + i], h = data[3 * n + i];
      boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
      scores.push(bestScore);
      cls.push(best);
    }
  }
  const m = scores.map((s) => s >= conf);
  boxes = boxes.filter((_, i) => m[i]);
  scores = scores.filter((_, i) => m[i]);
  cls = cls.filter((_, i) => m[i]);
  if (layout === "raw" && boxes.length) {
    const keep = [];
    for (const c of [...new Set(cls)].sort((a, b) => a - b)) {
      const idx = cls.map((k, i) => (k === c ? i : -1)).filter((i) => i >= 0);
      for (const k of nms(idx.map((i) => boxes[i]), idx.map((i) => scores[i]), iouThres)) keep.push(idx[k]);
    }
    boxes = keep.map((i) => boxes[i]);
    scores = keep.map((i) => scores[i]);
    cls = keep.map((i) => cls[i]);
  }
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]); // stable
  return { boxes: order.map((i) => boxes[i]), scores: order.map((i) => scores[i]), cls: order.map((i) => cls[i]) };
}

// classes.json {"0": "helmet_on", ...} -> {0: "helmet_on", ...}
export function parseClasses(json) {
  const names = {};
  for (const [k, v] of Object.entries(json)) names[Number(k)] = String(v);
  return names;
}

// ---- browser part ----

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !res.body.getReader) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress?.(buf.length, total || buf.length);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(got, total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

export class Detector {
  // status(text) gets human-readable progress; progress(fraction or null, text) drives a bar.
  static async create({ modelUrl = "model/helmet.onnx", classesUrl = "model/classes.json", status, progress } = {}) {
    const d = new Detector();
    progress?.(null, "Loading class names...");
    const cres = await fetch(classesUrl);
    if (!cres.ok) throw new Error(`${classesUrl}: HTTP ${cres.status}`);
    d.names = parseClasses(await cres.json());
    d.nc = Object.keys(d.names).length;

    progress?.(null, `Loading ONNX Runtime ${ORT_VERSION} from jsDelivr...`);
    const ort = await import(ORT_URL);
    d.ort = ort;
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.proxy = false;
    d.threads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    ort.env.wasm.numThreads = d.threads; // multithreading needs cross-origin isolation; 1 thread otherwise

    const model = await fetchWithProgress(modelUrl, (got, total) =>
      progress?.(total ? got / total : null,
        `Downloading model: ${(got / 1e6).toFixed(1)}${total ? ` of ${(total / 1e6).toFixed(1)}` : ""} MB`));

    const errors = [];
    for (const ep of PROVIDERS) {
      if (ep === "webgpu" && !navigator.gpu) { errors.push("webgpu: not available in this browser"); continue; }
      try {
        progress?.(null, `Starting the model (${ep})...`);
        const session = await ort.InferenceSession.create(model, { executionProviders: [ep], graphOptimizationLevel: "all" });
        d.session = session;
        d.inputName = session.inputNames[0];
        d.outputName = session.outputNames[0];
        await d.run(new Float32Array(3 * IMGSZ * IMGSZ).fill(PAD_VALUE / 255)); // warm-up; proves the provider works
        d.provider = ep;
        break;
      } catch (e) {
        errors.push(`${ep}: ${e?.message || e}`);
        d.session = null;
      }
    }
    if (!d.session) throw new Error(`no execution provider worked (${errors.join("; ")})`);
    d.fallbackNotes = errors;
    d.canvas = document.createElement("canvas");
    d.canvas.width = d.canvas.height = IMGSZ;
    d.ctx = d.canvas.getContext("2d", { willReadFrequently: true });
    d.busy = false;
    d.ms = 0;
    d.fps = 0;
    d._lastDone = 0;
    status?.(`Model ready (${d.provider}${d.provider === "wasm" ? `, ${d.threads} thread${d.threads > 1 ? "s" : ""}` : ""})`);
    return d;
  }

  async run(tensorData) {
    const t = new this.ort.Tensor("float32", tensorData, [1, 3, IMGSZ, IMGSZ]);
    const out = await this.session.run({ [this.inputName]: t });
    const o = out[this.outputName];
    return { data: o.data, dims: Array.from(o.dims) };
  }

  // One frame from a <video> or <canvas>; returns null if an inference is still running (frame skipped),
  // else [{box: [x1,y1,x2,y2] in source pixels, conf, cls, name}] best first.
  async detect(source, w, h, conf = DETECT_CONF) {
    if (this.busy) return null;
    this.busy = true;
    const t0 = performance.now();
    try {
      const p = letterboxParams(h, w, IMGSZ);
      const ctx = this.ctx;
      ctx.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
      ctx.fillRect(0, 0, IMGSZ, IMGSZ);
      ctx.drawImage(source, 0, 0, w, h, p.left, p.top, p.newW, p.newH);
      const rgba = ctx.getImageData(0, 0, IMGSZ, IMGSZ).data;
      const { data, dims } = await this.run(rgbaToNCHW(rgba, IMGSZ));
      const r = postprocess(data, dims, this.nc, conf);
      const boxes = boxesToOriginal(r.boxes, p.r, p.left, p.top, w, h);
      const now = performance.now();
      this.ms = now - t0;
      if (this._lastDone) {
        const inst = 1000 / (now - this._lastDone);
        this.fps = this.fps ? 0.8 * this.fps + 0.2 * inst : inst;
      }
      this._lastDone = now;
      return boxes.map((box, i) => ({ box, conf: r.scores[i], cls: r.cls[i], name: this.names[r.cls[i]] }));
    } finally {
      this.busy = false;
    }
  }

  resetTiming() {
    this._lastDone = 0;
    this.fps = 0;
  }
}
