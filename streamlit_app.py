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
import shutil
import sys
import tempfile
import time
import traceback
import uuid

import streamlit as st

import onnx_utils
from live_component import helmet_live
import processing as proc
import telegram_alerts
from processing import UploadError

st.set_page_config(page_title="HelmetVision demo", page_icon="⛑️", layout="wide")


@st.cache_resource(show_spinner="Loading the model...")
def load_detector():
    return onnx_utils.OnnxDetector(threads=2)


def get_detector():
    try:
        return load_detector()
    except Exception as e:
        print(f"[app] model load failed: {type(e).__name__}: {e}", file=sys.stderr)
        raise UploadError("The model could not be loaded. Please try again later.") from None


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
        annotated, dets = show_photo(f.getvalue())
        if tg_owner() and st.button("Send this result to Telegram", key="photo_send"):
            bad = proc.violations(dets)
            caption = (alert_caption("photo", max(d.conf for d in bad)) if bad
                       else f"Hosted demo - photo: no no_helmet alert ({len(dets)} heads)")
            tg_status(tg_send(caption, proc.small_jpeg(annotated[:, :, ::-1])))
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


def show_video_result(r):
    cols = st.columns(4)
    cols[0].metric("Alerts", len(r["alerts"]))
    cols[1].metric("Tracks", r["tracks"])
    cols[2].metric("Frames checked", r["frames"])
    cols[3].metric("Video length", f"{r['duration']:.1f} s")
    st.caption(f"Sampled at about {r['sample_fps']} frames per second. An alert needs no_helmet "
               f"(confidence 0.60 or more) on the same track in 5 of its last 8 sampled frames; each "
               f"track then waits 60 s of video time before it can alert again.")
    st.subheader("Alert timeline")
    if r["alerts"]:
        st.dataframe([{"time": a["time"], "track": a["track"], "confidence": a["conf"]}
                      for a in r["alerts"]], hide_index=True)
    else:
        st.info("No alerts in this video.")
    if r["suppressed"]:
        st.caption(f"{len(r['suppressed'])} repeat alert(s) suppressed (same spot, new track id): "
                   + ", ".join(f"{s['time']} #{s['track']}" for s in r["suppressed"]))
    if r["gallery"]:
        st.subheader("Alert frames")
        for line in r.get("tg") or []:
            tg_status(line)
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
        file_key = getattr(f, "file_id", None) or (f.name, f.size)
        if not cache or cache[0] != file_key:
            st.session_state.pop("video_result", None)
            result = run_video(f)
            if tg_alerts_on():  # only when the alerts fire, i.e. once per processed upload
                result["tg"] = send_video_alerts(result)
            st.session_state["video_result"] = (file_key, result)
        show_video_result(st.session_state["video_result"][1])
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
        bad = proc.violations(dets)
        if not bad or TG is None:
            return
        # each snapshot is decided once, when it is first shown; reruns only repeat the status
        key = "camera:" + hashlib.sha256(data).hexdigest()
        done = st.session_state.setdefault("tg_done", {})
        if key not in done:
            done[key] = (tg_send(alert_caption("camera snapshot", max(d.conf for d in bad)),
                                 proc.small_jpeg(annotated[:, :, ::-1])) if tg_alerts_on() else None)
        tg_status(done[key])
    friendly(work)


# ---- Telegram alerts (owner mode) ----
TG_KEYS = ("TG_TOKEN", "TG_CHAT", "ALERT_ACCESS_CODE")


def tg_config():
    """The three Telegram secrets from st.secrets, or None when any is missing; the whole section is
    then hidden. The values are never shown, logged or printed."""
    try:
        cfg = {k: st.secrets.get(k) for k in TG_KEYS}
    except Exception:  # no secrets file at all, or an unreadable one
        return None
    if not all(isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip()
               for v in cfg.values()):
        return None
    return {k: str(v).strip() for k, v in cfg.items()}


@st.cache_resource
def tg_limiter():
    """One RateLimiter shared by all sessions of this app process."""
    return telegram_alerts.RateLimiter()


def tg_owner():
    """True only in a session that entered the right access code (and while the secrets exist)."""
    return TG is not None and bool(st.session_state.get("tg_unlocked"))


def tg_alerts_on():
    return tg_owner() and bool(st.session_state.get("tg_on"))


def alert_caption(where, conf):
    return f"No helmet - hosted demo - {where} conf {conf:.2f}"


def tg_send(caption, jpeg=None):
    """Send through the shared rate limiter in owner mode; returns the status line for tg_status."""
    if not tg_owner():
        return None
    ok, why = tg_limiter().acquire(st.session_state.setdefault("tg_sid", uuid.uuid4().hex))
    if not ok:
        return f"Telegram: rate limited ({why})"
    if not st.session_state.get("tg_photo", True):
        jpeg = None
    with st.spinner("Sending to Telegram..."):
        sent, msg = telegram_alerts.send_alert(TG["TG_TOKEN"], TG["TG_CHAT"], caption, jpeg)
    return "Telegram: alert sent" if sent else f"Telegram: send failed ({msg})"


def tg_status(line):
    if not line:
        return
    if line.endswith("alert sent"):
        st.success(line)
    elif "rate limited" in line:
        st.warning(line)
    else:
        st.error(line)


def send_video_alerts(r):
    """One Telegram attempt per alert frame; returns the status lines (most hit the rate limiter)."""
    lines = []
    for (jpeg, _), meta in zip(r["gallery"], r["gallery_meta"]):
        status = tg_send(alert_caption(f"video t={meta['t']:.1f}s", meta["conf"]),
                         proc.small_jpeg(proc.decode_image(jpeg)))
        lines.append(f"{meta['time']} {status}")
    return lines


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
    value = helmet_live(key="live", report_alerts=tg_alerts_on())
    if TG is not None:
        handle_live_event(value)


def handle_live_event(value):
    """Send a Live alert once per event id; the component returns its last value on every rerun."""
    ev = telegram_alerts.parse_live_event(value)
    if ev is None:
        return
    seen = st.session_state.setdefault("tg_live_seen", {})  # event id -> status line
    if ev["id"] not in seen:
        when = time.strftime(" at %H:%M:%S UTC", time.gmtime(ev["ts"] / 1000)) if ev["ts"] else ""
        seen[ev["id"]] = (tg_send(alert_caption("live camera", ev["conf"]) + when, ev["jpeg"])
                          if tg_alerts_on() else None)
        while len(seen) > 200:
            seen.pop(next(iter(seen)))
    tg_status(seen[ev["id"]])


def about_tab():
    st.markdown("""
**HelmetVision** finds heads in a picture or video and labels each one `helmet_on` or `no_helmet`.
It uses a small YOLO26n detector trained on public helmet datasets, exported to ONNX.

- **Live**: continuous detection from your camera. The model runs in your browser with
  onnxruntime-web, using the same tracker and alert rules as the Video tab, with a banner and a beep
  when an alert fires.
- **Photo** and **Camera snapshot**: one detection pass, boxes drawn green (`helmet_on`) or red
  (`no_helmet`) with their confidence.
- **Video**: about 3 frames per second go through the detector, a simple box tracker and the same
  alert rules as the edge device: an alert needs `no_helmet` (confidence 0.60 or more) on one tracked
  head in 5 of its last 8 sampled frames, with a 60 s cooldown per track and suppression of repeats
  when a head gets a new track id.

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
if TG is not None:
    owner_section()
st.title("⛑️ HelmetVision demo")
st.caption("Helmet detection with ONNX Runtime, live in your browser or on uploads. A demo, not a safety system.")
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
