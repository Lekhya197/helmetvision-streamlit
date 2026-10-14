# HelmetVision demo

A hosted demo of HelmetVision helmet detection. It finds heads in a live camera feed, a photo, a short
video or a camera snapshot and labels each one `helmet_on` (green) or `no_helmet` (red) with its
confidence.

- Live: continuous detection from your camera, running entirely in your browser with onnxruntime-web
  (WebGPU if available, otherwise WebAssembly) inside a custom component (`live_component/`). No video
  is uploaded. Speed depends on your device. Needs HTTPS or localhost.

- Model: YOLO26n trained on public helmet datasets, exported to ONNX (416x416) and run with ONNX Runtime
  on CPU. No PyTorch and no Ultralytics package are needed at runtime.
- Video: about 3 frames per second go through the detector, a simple IoU tracker and the edge
  device's alert rules with a shorter window for the sampled frames (no_helmet at confidence 0.60 or
  more in 3 of the last 5 sampled frames of one track, about one second; a 60 s cooldown per track;
  suppression of repeats when a head gets a new track id). Live keeps 5 of the last 8 frames.
  Limits: 25 MB and 30 s.
- Camera snapshot: one photo per click, not a continuous live feed.

Privacy: the Live tab processes video in your browser and uploads nothing. Photo, Video and Camera
snapshot are processed in memory on the hosting server and not stored (a video is written to a
temporary file while it is processed and deleted right after).

Telegram alerts are an owner-only feature protected by an access code; visitors cannot trigger messages to my phone.
Visitors' use of the demo sends no alerts anywhere.

**This is a demo, not a safety system.**

## Run locally

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Start it with the venv's Python as above, not another `streamlit` on the PATH: without onnxruntime
the app still starts and Live works, but the server tabs cannot load the model. The first log line
shows the versions in use and a model self-test. `HELMET_DEBUG=1` adds a Diagnostics panel (versions,
model status, which secret names are set) for local runs.

## Telegram alerts (owner only, optional)

`telegram_alerts.py` sends alerts to the owner's Telegram chat. It is off unless `TG_TOKEN`,
`TG_CHAT` and `ALERT_ACCESS_CODE` are set in the app's secrets; then a sidebar section asks for the
access code (5 wrong tries lock it for 60 s). After unlocking, with the toggle on, alerts are sent
automatically: one per new Photo or Camera snapshot that has a no_helmet detection at confidence 0.60
or more (the annotated picture, at most 800 px), one per uploaded video after it is processed (the
highest-confidence alert frame, with every alert's timestamp), and Live alerts (a 320 px thumbnail).
Reruns never resend. Per session: one send per 5 s for Photo and Camera snapshot, and one per
`TG_COOLDOWN_SECONDS` (default 60) for Live and Video; overall at most `TG_MAX_PER_HOUR` (default 10)
per hour. Both are optional secrets. Never commit `.streamlit/secrets.toml`.

## Licences and credits

- ONNX Runtime Web (MIT) is loaded in the browser from jsDelivr.
- Ultralytics code and models (YOLO26n) are licensed AGPL-3.0; the model in `model/` is derived from
  them.
- Training data: Roboflow "Hard Hat Workers"
  (https://universe.roboflow.com/joseph-nelson/hard-hat-workers), Public Domain; Cheese
  "Helmet - No helmet detection" (https://universe.roboflow.com/cheese-ttcmr/helmet-no-helmet-detection),
  MIT.
