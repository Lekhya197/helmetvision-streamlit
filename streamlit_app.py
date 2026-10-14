"""HelmetVision hosted demo (Streamlit Community Cloud, CPU only, about 1 GB RAM).

ONNX Runtime only: no torch, no Ultralytics. The Live tab is a custom component that runs the model in
the visitor's browser (onnxruntime-web), so its video never reaches the server. Photo, Video and Camera
snapshot run on the server: nothing is stored, uploads are processed in memory, and the video tab's
temporary files are deleted in a finally block. Telegram alerts are owner-only: the sidebar section
exists only when TG_TOKEN, TG_CHAT and ALERT_ACCESS_CODE are in st.secrets, and nothing is sent until
the access code is entered in that session.
"""
import hashlib
import os
import pathlib
import shutil
import sys
import tempfile
import time
import traceback
import uuid

import numpy as np
import streamlit as st

import diagnostics
import onnx_utils
from live_component import helmet_live
import processing as proc
import telegram_alerts
from processing import UploadError

APP_DIR = pathlib.Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Loading the model...")
def load_detector():
    return onnx_utils.OnnxDetector(threads=2)  # model/ next to onnx_utils.py, not the working directory


def get_detector():
    try:
        return load_detector()
    except Exception as e:  # the cause goes to the server log; visitors get the short message
        print(f"[app] model load failed: {diagnostics.short_error(e)}", file=sys.stderr)
        raise UploadError("The model could not be loaded. Please try again later.") from None


@st.cache_resource(show_spinner=False)
def startup_check():
    """Once per server process: log the versions and a model self-test (load plus one inference)."""
    v = diagnostics.versions()
    try:
        load_detector().detect(np.full((onnx_utils.IMGSZ, onnx_utils.IMGSZ, 3), 114, np.uint8))
        model = {"ok": True, "error_type": None, "text": "ok"}
    except Exception as e:
        model = {"ok": False, "error_type": type(e).__name__, "text": f"FAILED {diagnostics.short_error(e)}"}
    print(f"[app] python {v['python']}, streamlit {v['streamlit']}, onnxruntime {v['onnxruntime']}; "
          f"model self-test: {model['text']}", file=sys.stderr)
    mismatches = diagnostics.pin_mismatches(APP_DIR / "requirements.txt")
    if mismatches:
        print("[app] not the versions pinned in requirements.txt: " + "; ".join(mismatches), file=sys.stderr)
    return {"versions": v, "model": model}


def friendly(fn):
    """Run one tab's work; show a short message instead of a traceback."""
    try:
        fn()
    except UploadError as e:
        st.error(str(e))
    except Exception:
        traceback.print_exc(file=sys.stderr)  # to the server log only
        st.error("Sorry, something went wrong while processing this file. Please try another one.")


def show_photo(data):
    """Detect and show one picture; returns (annotated RGB image, detections)."""
    detector = get_detector()
    img = proc.decode_image(data)
    annotated, dets = proc.photo(detector, img)
    c = proc.counts(dets)
    cols = st.columns(3)
    cols[0].metric("Heads found", len(dets))
    cols[1].metric("helmet_on", c.get("helmet_on", 0))
    cols[2].metric("no_helmet", c.get("no_helmet", 0))
    st.image(annotated, caption="green = helmet_on, red = no_helmet")
    if dets:
        st.dataframe([{"class": d.name, "confidence": round(d.conf, 3),
                       "x1": round(d.box[0]), "y1": round(d.box[1]), "x2": round(d.box[2]),
                       "y2": round(d.box[3])} for d in dets], hide_index=True)
    else:
        st.info("No heads detected in this picture.")
    return annotated, dets


def photo_tab():
    f = st.file_uploader(f"Upload a photo (jpg, png or webp, up to {proc.mb(proc.PHOTO_MAX_BYTES)})",
                         type=list(proc.PHOTO_TYPES), key="photo")
    if f is None:
        return

    def work():
        proc.check_upload(f.name, f.size, proc.PHOTO_TYPES, proc.PHOTO_MAX_BYTES)
        data = f.getvalue()
        annotated, dets = show_photo(data)
        picture_alert("photo", "photo", data, annotated, dets)
    friendly(work)


def run_video(f):
    """Process one upload; returns the result dict with the annotated video as bytes (files deleted)."""
    detector = get_detector()
    ext = os.path.splitext(f.name)[1].lower()
    fd_in, path_in = tempfile.mkstemp(prefix="hv_in_", suffix=ext)
    fd_out, path_out = tempfile.mkstemp(prefix="hv_out_", suffix=".mp4")
    os.close(fd_out)
    try:
        with os.fdopen(fd_in, "wb") as out:
            f.seek(0)
            shutil.copyfileobj(f, out, 1024 * 1024)
        bar = st.progress(0.0, text="Processing video...")
        result = proc.process_video(path_in, detector, out_path=path_out,
                                    progress=lambda p, text: bar.progress(p, text=text))
        bar.empty()
        result["video_bytes"] = None
        if result["video"] and os.path.getsize(path_out) > 0:
            with open(path_out, "rb") as v:
                result["video_bytes"] = v.read()
        return result
    finally:
        for p in (path_in, path_out):
            try:
                os.remove(p)
            except OSError:
                pass


def show_video_result(r, tg_line=None):
    cols = st.columns(4)
    cols[0].metric("Alerts", len(r["alerts"]))
    cols[1].metric("Tracks", r["tracks"])
    cols[2].metric("Frames checked", r["frames"])
    cols[3].metric("Video length", f"{r['duration']:.1f} s")
    st.caption(f"Sampled at about {r['sample_fps']} frames per second. An alert needs no_helmet "
               f"(confidence {proc.ALERT_CONF:.2f} or more) on the same track in {proc.VIDEO_MIN_HITS} of "
               f"its last {proc.VIDEO_WINDOW} sampled frames (about one second); each track then waits "
               f"60 s of video time before it can alert again.")
    st.subheader("Alert timeline")
    if r["alerts"]:
        st.dataframe([{"time": a["time"], "track": a["track"], "confidence": a["conf"]}
                      for a in r["alerts"]], hide_index=True)
        tg_status(tg_line)
    else:
        st.info("No alerts in this video.")
        if r.get("unconfirmed"):
            st.info(f"no_helmet was seen, but brief detections not confirmed; need {proc.VIDEO_MIN_HITS} "
                    f"hits in {proc.VIDEO_WINDOW} sampled frames.")
        if TG is not None:
            st.caption("No violation confirmed - nothing sent")
    if r["suppressed"]:
        st.caption(f"{len(r['suppressed'])} repeat alert(s) suppressed (same spot, new track id): "
                   + ", ".join(f"{s['time']} #{s['track']}" for s in r["suppressed"]))
    if r["gallery"]:
        st.subheader("Alert frames")
        st.image([g[0] for g in r["gallery"]], caption=[g[1] for g in r["gallery"]], width=320)
    st.subheader("Annotated video")
    if r.get("video_bytes"):
        st.video(r["video_bytes"], format="video/mp4")
    else:
        st.info(r.get("note") or "The annotated video could not be made.")


def video_tab():
    f = st.file_uploader(f"Upload a video (mp4, mov or webm, up to {proc.mb(proc.VIDEO_MAX_BYTES)} "
                         f"and {proc.VIDEO_MAX_SECONDS:.0f} s)", type=list(proc.VIDEO_TYPES), key="video")
    if f is None:
        st.session_state.pop("video_result", None)  # free the last result's frames and video
        return

    def work():
        proc.check_upload(f.name, f.size, proc.VIDEO_TYPES, proc.VIDEO_MAX_BYTES)
        cache = st.session_state.get("video_result")
        digest = hashlib.sha256(f.getvalue()).hexdigest()
        if not cache or cache[0] != digest:
            st.session_state.pop("video_result", None)
            st.session_state["video_result"] = (digest, run_video(f))
        result = st.session_state["video_result"][1]
        show_video_result(result, video_alert(digest, result))
    friendly(work)


def camera_tab():
    st.info("This takes a single snapshot from your camera and checks it once. It is not a "
            "continuous live feed.")
    shot = st.camera_input("Take a snapshot")
    if shot is None:
        return

    def work():
        data = shot.getvalue()
        annotated, dets = show_photo(data)
        picture_alert("camera", "camera snapshot", data, annotated, dets)
    friendly(work)


# ---- Telegram alerts (owner mode) ----
TG_KEYS = ("TG_TOKEN", "TG_CHAT", "ALERT_ACCESS_CODE")


def secret_value(name):
    return st.secrets.get(name)  # raises when there is no secrets file at all


def tg_config():
    """The three Telegram secrets from st.secrets, or None when any is missing; the whole section is
    then hidden and only the missing NAMES are logged. Values are never shown, logged or printed."""
    present = diagnostics.presence(TG_KEYS, secret_value)
    missing = tuple(k for k in TG_KEYS if not present[k])
    if missing:
        log_missing_secrets(missing)
        return None
    return {k: str(secret_value(k)).strip() for k in TG_KEYS}


@st.cache_resource(show_spinner=False)
def log_missing_secrets(names):
    """Once per process and set of names."""
    print(f"[app] Telegram alerts off: missing secret(s) {', '.join(names)}", file=sys.stderr)


# Optional secrets: (name, default, lowest, highest). Values out of range fall back to the default.
TG_LIMIT_KEYS = (("TG_MAX_PER_HOUR", 10, 1, 1000), ("TG_COOLDOWN_SECONDS", 60, 5, 3600))
PICTURE_COOLDOWN = 5.0  # Photo and Camera snapshot; Live, Video and the test message use TG_COOLDOWN_SECONDS


def tg_limits():
    """{name: int} for TG_LIMIT_KEYS from st.secrets; an invalid value is logged by NAME only."""
    out = {}
    for name, default, lo, hi in TG_LIMIT_KEYS:
        out[name], ok = telegram_alerts.limit_setting(secret_value(name), default, lo, hi)
        if not ok:
            log_bad_limit(name, default, lo, hi)
    return out


@st.cache_resource(show_spinner=False)
def log_bad_limit(name, default, lo, hi):
    print(f"[app] {name} is not a whole number from {lo} to {hi}; using {default}", file=sys.stderr)


@st.cache_resource(show_spinner=False)
def tg_limiter(hourly_cap, cooldown):
    """One RateLimiter shared by all sessions of this app process (a new one if the settings change)."""
    return telegram_alerts.RateLimiter(cooldown=float(cooldown), hourly_cap=hourly_cap)


def tg_owner():
    """True only in a session that entered the right access code (and while the secrets exist)."""
    return TG is not None and bool(st.session_state.get("tg_unlocked"))


def tg_alerts_on():
    return tg_owner() and bool(st.session_state.get("tg_on"))


def alert_caption(where, conf):
    return f"No helmet - hosted demo - {where} conf {conf:.2f}"


def tg_decided(key, decide):
    """The status stored for `key` in this session, calling decide() only the first time, so Streamlit
    reruns never send twice."""
    done = st.session_state.setdefault("tg_done", {})
    if key not in done:
        done[key] = decide()
        while len(done) > 200:
            done.pop(next(iter(done)))
    return done[key]


def auto_send(caption, jpeg, channel):
    """An automatic alert: sent only in owner mode with the toggle on; otherwise the reason."""
    if not tg_owner():
        return "Telegram: locked"
    if not tg_alerts_on():
        return "Telegram: toggle is off"
    return tg_send(caption, jpeg, channel=channel)


def picture_alert(kind, where, data, annotated, dets):
    """Photo and Camera snapshot: one automatic alert per new picture (SHA-256 of its bytes) when it
    has a no_helmet detection at ALERT_CONF or more, then its status and, in owner mode, "Send again"."""
    if TG is None:
        return
    bad = proc.violations(dets)
    caption = proc.violation_caption(where, bad) if bad else None
    status = tg_decided(f"{kind}:" + hashlib.sha256(data).hexdigest(),
                        lambda: auto_send(caption, proc.small_jpeg(annotated[:, :, ::-1]), kind) if bad else None)
    if bad:
        tg_status(status)
    else:
        st.info("No violation detected - nothing sent")
    if tg_owner() and st.button("Send again", key=f"{kind}_send"):
        tg_status(tg_send(caption or f"Hosted demo - {where}: no no_helmet alert ({len(dets)} heads)",
                          proc.small_jpeg(annotated[:, :, ::-1]), channel=kind))


def tg_send(caption, jpeg=None, sent_text="Telegram: sent", channel="test"):
    """Send through the shared rate limiter in owner mode; returns the status line for tg_status.
    Each session has its own cooldown per channel: PICTURE_COOLDOWN for "photo" and "camera",
    TG_COOLDOWN_SECONDS for "live", "video" and "test"; TG_MAX_PER_HOUR is global."""
    if not tg_owner():
        return None
    limits = tg_limits()
    cooldown = PICTURE_COOLDOWN if channel in ("photo", "camera") else limits["TG_COOLDOWN_SECONDS"]
    sid = st.session_state.setdefault("tg_sid", uuid.uuid4().hex)
    ok, why = tg_limiter(limits["TG_MAX_PER_HOUR"], limits["TG_COOLDOWN_SECONDS"]).acquire(
        f"{sid}:{channel}", cooldown=cooldown)
    if not ok:
        return f"Telegram: rate limited ({why})"
    if not st.session_state.get("tg_photo", True):
        jpeg = None
    with st.spinner("Sending to Telegram..."):
        sent, msg = telegram_alerts.send_alert(TG["TG_TOKEN"], TG["TG_CHAT"], caption, jpeg)
    return sent_text if sent else f"Telegram: send failed ({msg})"


def tg_status(line):
    if not line:
        return
    if line.endswith("sent"):
        st.success(line)
    elif "rate limited" in line:
        st.warning(line)
    elif line.endswith(("toggle is off", "locked")):
        st.info(line)
    else:
        st.error(line)


def tg_state_caption():
    """Above the tabs, only when Telegram is configured: locked, ON or OFF for this session."""
    if TG is None:
        return
    if not tg_owner():
        st.caption("Telegram alerts: locked (owner access code in the sidebar)")
    else:
        st.caption(f"Telegram alerts: {'ON' if tg_alerts_on() else 'OFF'}")


def diagnostics_panel():
    """Local debugging only (HELMET_DEBUG=1): versions, model status and which secret names exist."""
    if os.environ.get("HELMET_DEBUG") != "1":
        return
    v, model = STARTUP["versions"], STARTUP["model"]
    with st.expander("Diagnostics"):
        st.markdown("\n".join([
            f"- python {v['python']}, streamlit {v['streamlit']}, onnxruntime {v['onnxruntime']}",
            "- model self-test at startup: " + ("ok" if model["ok"] else f"FAILED ({model['error_type']})"),
            *(f"- secret {k}: {'present' if ok else 'missing'}"
              for k, ok in diagnostics.presence(TG_KEYS, secret_value).items()),
        ]))


def video_alert(digest, r):
    """ONE automatic alert per uploaded video (SHA-256 of the file), after the whole video is
    processed: the frame of the highest-confidence alert, captioned with every alert's timestamp."""
    if TG is None or not r["alerts"]:
        return None
    return tg_decided("video:" + digest, lambda: auto_send(
        proc.video_caption(r["alerts"]), proc.small_jpeg(proc.decode_image(r["best"]["jpeg"])), "video"))


def check_code():
    """on_change of the access-code field: unlock on a match, count wrong tries, clear the field."""
    entered = st.session_state.get("tg_code", "")
    st.session_state["tg_code"] = ""
    cfg = tg_config()
    if not entered or cfg is None:
        return
    guard = st.session_state.setdefault("tg_guard", telegram_alerts.AttemptGuard())
    result = guard.check(entered, cfg["ALERT_ACCESS_CODE"])
    st.session_state["tg_unlocked"] = result == "ok"
    st.session_state["tg_code_msg"] = result


def lock_owner_mode():
    for k in ("tg_unlocked", "tg_on", "tg_photo"):
        st.session_state.pop(k, None)


def owner_section():
    st.sidebar.subheader("Telegram alerts (owner mode)")
    if tg_owner():
        st.sidebar.toggle("Send alerts to Telegram", value=False, key="tg_on")
        st.sidebar.checkbox("Include photo", value=True, key="tg_photo")
        if st.sidebar.button("Send test message", key="tg_test"):
            line = tg_send("HelmetVision hosted demo: test message", None, sent_text="Telegram: sent")
            with st.sidebar:
                tg_status(line)
        st.sidebar.button("Lock", key="tg_lock", on_click=lock_owner_mode)
        return
    guard = st.session_state.get("tg_guard")
    left = guard.locked_for() if guard else 0.0
    st.sidebar.text_input("Access code", type="password", key="tg_code", on_change=check_code,
                          disabled=left > 0)
    msg = st.session_state.pop("tg_code_msg", None)
    if left > 0:
        st.sidebar.warning(f"Too many wrong codes. Try again in {left:.0f} s.")
    elif msg == "wrong":
        st.sidebar.error("Wrong code.")
    st.sidebar.caption("Owner only. Visitors cannot trigger any Telegram message.")


def live_tab():
    st.caption("Continuous detection from your camera, running entirely in your browser with "
               "onnxruntime-web (WebGPU if available, otherwise WebAssembly). No video is uploaded and "
               "nothing is stored. Speed depends on your device. Needs HTTPS or localhost and camera "
               "permission.")
    # Alerts leave the browser only for the owner: events once unlocked (so "toggle is off" can be
    # shown), thumbnails only with the toggle and Include photo on. Visitors' pages send nothing.
    value = helmet_live(key="live", report_alerts=tg_owner(),
                        include_thumb=tg_alerts_on() and bool(st.session_state.get("tg_photo", True)))
    if TG is None:
        return
    if not tg_owner():
        tg_status("Telegram: locked")
        st.caption("Live alerts are not reported until the owner enters the access code in the sidebar.")
    handle_live_event(value)


def handle_live_event(value):
    """Send a Live alert once per event id; the component returns its last value on every rerun."""
    ev = telegram_alerts.parse_live_event(value)
    if ev is None:
        return
    seen = st.session_state.setdefault("tg_live_seen", {})  # event id -> status line
    when = time.strftime("%H:%M:%S UTC", time.gmtime(ev["ts"] / 1000)) if ev["ts"] else "time unknown"
    if ev["id"] not in seen:
        if not tg_owner():
            line = "Telegram: locked"
        elif not tg_alerts_on():
            line = "Telegram: toggle is off"
        else:
            line = tg_send(f"{alert_caption('live camera', ev['conf'])} at {when}", ev["jpeg"],
                           channel="live")
        seen[ev["id"]] = line
        while len(seen) > 200:
            seen.pop(next(iter(seen)))
    tg_status(seen[ev["id"]])
    st.caption(f"Last Live alert: {when}, conf {ev['conf']:.2f}")


def about_tab():
    st.markdown("""
**HelmetVision** finds heads in a picture or video and labels each one `helmet_on` or `no_helmet`.
It uses a small YOLO26n detector trained on public helmet datasets, exported to ONNX.

- **Live**: continuous detection from your camera. The model runs in your browser with
  onnxruntime-web, with the same tracker as the Video tab and the edge device's alert rule (5 of the
  last 8 frames), and a banner and a beep when an alert fires.
- **Photo** and **Camera snapshot**: one detection pass, boxes drawn green (`helmet_on`) or red
  (`no_helmet`) with their confidence.
- **Video**: about 3 frames per second go through the detector, a simple box tracker and the
  edge device's alert rules with a shorter window for the sampled frames: an alert needs `no_helmet`
  (confidence 0.60 or more) on one tracked head in 3 of its last 5 sampled frames (about one second),
  with a 60 s cooldown per track and suppression of repeats when a head gets a new track id.

**Privacy.** The **Live** tab processes the camera video in your browser and uploads nothing; its
alert log lives only in the open page. The **Photo**, **Video** and **Camera snapshot** tabs are
processed in memory on the hosting server and are not stored (a video upload is written to a
temporary file while it is processed and deleted right after).

**Alerts.** Telegram alerts are an owner-only feature protected by an access code; visitors cannot trigger messages to my phone. Your use of this demo sends no alerts anywhere.

**Licences and credits.** Ultralytics code and models (YOLO26n) are AGPL-3.0. Training data:
Roboflow "Hard Hat Workers" (Public Domain) and Cheese "Helmet - No helmet detection" (MIT).

**This is a demo, not a safety system.** It makes mistakes and must not be used to make safety
decisions.
""")


TG = tg_config()
# expanded so the owner finds the access code; without Telegram secrets there is no sidebar at all
st.set_page_config(page_title="HelmetVision demo", page_icon="⛑️", layout="wide",
                   initial_sidebar_state="expanded" if TG is not None else "auto")
STARTUP = startup_check()
if TG is not None:
    owner_section()
st.title("⛑️ HelmetVision demo")
st.caption("Helmet detection with ONNX Runtime, live in your browser or on uploads. A demo, not a safety system.")
tg_state_caption()
diagnostics_panel()
live, photo, video, camera, about = st.tabs(["Live", "Photo", "Video", "Camera snapshot", "About"])
with live:
    live_tab()
with photo:
    photo_tab()
with video:
    video_tab()
with camera:
    camera_tab()
with about:
    about_tab()
