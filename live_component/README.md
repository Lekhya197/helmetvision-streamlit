# helmet_live: the Live camera component

A custom Streamlit component that runs helmet detection on the visitor's camera **inside their
browser**. The Streamlit server only serves static files, so it does no inference work, needs no
WebRTC or TURN server, and receives no video.

## How it works

```
getUserMedia (ideal 640x480, front or back camera)
  -> canvas letterbox 416x416, gray 114        lib/detector.js  (mirrors scripts/onnx_ref.py)
  -> onnxruntime-web: WebGPU, else WASM        one inference at a time; frames are skipped while busy
  -> postprocess: per-class NMS 0.7, conf 0.25, boxes mapped back to camera pixels
  -> greedy IoU tracker (iou 0.3, max age 15)  lib/tracker.js   (port of streamlit_app/tracker.py)
  -> k-of-n voter + suppressor                 lib/voter.js     (port of streamlit_app/voting.py)
       no_helmet at conf >= 0.60 in 5 of the last 8 inferences of one track, 60 s cooldown per
       track; repeats at the same spot under a new track id are dropped only if the old id is gone
  -> canvas boxes (green helmet_on, red no_helmet, "#id name conf"), banner, beep, alert log
```

- `__init__.py` declares the component (`components.declare_component("helmet_live", path=frontend/)`)
  and exposes `helmet_live(key=None, report_alerts=False)`. For visitors (`report_alerts=False`) it
  returns None and nothing leaves the browser.
- Owner mode only (`report_alerts=True`: access code entered and the Telegram toggle on): each alert is
  sent to Python with `streamlit:setComponentValue` (dataType json) as
  `{id, conf, ts, thumb_jpeg_base64}`, the thumbnail being the annotated frame at 320 px, JPEG under
  about 40 KB (`lib/report.js`). Python validates it, de-duplicates by `id` (the component returns its
  last value on every rerun) and sends it through the Telegram rate limiter. Two alerts in the same
  frame can collapse into one value; the rate limiter would drop the second anyway.
- `frontend/index.html` and `frontend/live.js` hold the UI. The Streamlit handshake is written by
  hand, with no npm packages: on load the page posts `streamlit:componentReady` (apiVersion 1) and
  `streamlit:setFrameHeight` (760). From `streamlit:render` messages it reads the theme and
  `args.report_alerts`.
- ONNX Runtime Web is loaded from jsDelivr at a pinned version (`ORT_VERSION` in
  `lib/detector.js`, currently 1.30.0, file `ort.webgpu.min.mjs`). The browser downloads it the
  first time Start is pressed, so a visitor who never opens the Live tab downloads nothing extra.
- WASM multithreading needs a cross-origin isolated page, which Streamlit pages are not. The
  component then uses 1 thread and says so.
- Timing: "ms per frame" is the last inference including pre- and postprocessing. "fps" is a
  moving average of completed inferences per second. Votes are counted per inference, so on a slow
  device 5 of 8 takes longer in seconds than on a fast one.

## Model path convention

The page loads `model/helmet.onnx` and `model/classes.json` by **relative URL**, so both files must
sit in `frontend/model/`, next to `index.html`.

- The committed copy is `streamlit_app/model/`; `frontend/model/` is git-ignored.
- For local runs, importing the component calls `model_files.sync()`, which copies any missing or
  changed file. You can also run `python streamlit_app/live_component/model_files.py`.
- `scripts/build_streamlit_repo.py` copies both files into `live_component/frontend/model/` in the
  deployed repo.
- Class names come from `classes.json`. Colors and the violation rule use the class **name**,
  never a fixed id.

## Tests

- `node --test "tests/js/*.test.mjs"` covers the tracker, voter, detector math and the owner-mode
  alert event (`lib/report.js`). It needs no packages and no browser.
  `tests/js/onnx_ref_fixture.json` holds expected results from `scripts/onnx_ref.py`; regenerate it
  with `python tests/js/make_onnx_ref_fixture.py`.
- `tests/test_streamlit_telegram.py` replaces `helmet_live` to check that `report_alerts` is true
  only in owner mode with the toggle on, and that repeated events are sent once.
- `tests/test_streamlit_app.py` checks that the app starts with the Live tab first and that the
  component files exist.

## Known limits

- Speed depends entirely on the visitor's device and browser, and on whether WebGPU is available or
  the slower single-threaded WASM path is used. No FPS figure has been measured yet.
- The browser scales the frame with its own canvas resampling, not OpenCV `INTER_LINEAR`. The
  letterbox geometry, padding, normalization and postprocessing match `onnx_ref.py` exactly, but
  pixel values after resizing can differ slightly, so scores can differ slightly from the server tabs.
- First use downloads the ONNX Runtime WASM file from jsDelivr and the model from the app. On mobile
  data this takes a while; the progress bar shows the model download.
- The camera needs HTTPS or localhost, and camera permission for the page. Streamlit's component
  iframe includes `camera` in its `allow` list.
- No Telegram, buzzer or event log in this version. Alerts exist only on the open page and are
  gone when it is closed. Nothing is uploaded or stored.
- This is a demo, not a safety system.
