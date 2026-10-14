"""Upload checks, drawing and the video loop for the Streamlit demo (pure Python, no Streamlit here).

The detector is anything with detect(img_bgr) -> [onnx_utils.Detection] (the app passes OnnxDetector,
tests pass a stub). Colors and the violation rule go by class name, never by class id.
"""
import os

import cv2
import numpy as np

from tracker import IoUTracker
from voting import AlertSuppressor, Tracked, ViolationVoter, decide, is_violation

PHOTO_TYPES = ("jpg", "jpeg", "png", "webp")
PHOTO_MAX_BYTES = 8 * 1024 * 1024
VIDEO_TYPES = ("mp4", "mov", "webm")
VIDEO_MAX_BYTES = 25 * 1024 * 1024
VIDEO_MAX_SECONDS = 30.0
SAMPLE_FPS = 3.0
VIDEO_LONG_SIDE = 960  # frames are downscaled to this before drawing and encoding
MAX_GALLERY = 20  # annotated alert frames kept in memory
TELEGRAM_LONG_SIDE = 800

COLORS = {"helmet_on": (0, 200, 0), "no_helmet": (0, 0, 230)}  # BGR by class name
OTHER_COLOR = (0, 200, 230)


class UploadError(Exception):
    """A problem with the user's file; the message is shown as is."""


def mb(n):
    return f"{n / 1024 / 1024:.0f} MB"


def check_upload(name, size, types, max_bytes):
    ext = os.path.splitext(name or "")[1].lower().lstrip(".")
    if ext not in types:
        raise UploadError(f"Unsupported file type '.{ext}'. Use one of: {', '.join(types)}.")
    if size > max_bytes:
        raise UploadError(f"The file is {size / 1024 / 1024:.1f} MB; the limit is {mb(max_bytes)}.")


def decode_image(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise UploadError("Could not read this image. Try a JPG or PNG file.")
    return img


def fmt_time(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def draw(img, dets, track_ids=None):
    """Boxes with "name conf" labels (and "#id" when track_ids is given), in place."""
    h = img.shape[0]
    scale = max(0.5, h / 900)
    thick = max(1, round(h / 400))
    for i, d in enumerate(dets):
        color = COLORS.get(d.name, OTHER_COLOR)
        x1, y1, x2, y2 = (int(round(v)) for v in d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        label = f"{d.name} {d.conf:.2f}"
        if track_ids is not None:
            label = f"#{track_ids[i]} {label}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y1 - 4 if y1 - th - base - 4 >= 0 else y2 + th + 4
        cv2.rectangle(img, (x1, ty - th - base), (x1 + tw + 4, ty + base), color, -1)
        cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick,
                    cv2.LINE_AA)
    return img


def banner(img, text, color=(0, 0, 230)):
    scale = max(0.6, img.shape[0] / 700)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (0, 0), (tw + 16, th + base + 16), color, -1)
    cv2.putText(img, text, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def counts(dets):
    out = {}
    for d in dets:
        out[d.name] = out.get(d.name, 0) + 1
    return out


def violations(dets):
    """Detections that count as a violation on their own (no_helmet at the voter's confidence)."""
    return [d for d in dets if is_violation(d.name, d.conf)]


def small_jpeg(img_bgr, long_side=TELEGRAM_LONG_SIDE, quality=85):
    """JPEG bytes of img_bgr fitted into long_side (never upscaled)."""
    h, w = img_bgr.shape[:2]
    r = min(1.0, long_side / max(h, w))
    if r < 1.0:
        img_bgr = cv2.resize(img_bgr, (max(1, round(w * r)), max(1, round(h * r))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def photo(detector, img_bgr):
    """(annotated RGB image, detections) for one picture."""
    dets = detector.detect(img_bgr)
    return draw(img_bgr.copy(), dets)[:, :, ::-1], dets


def downscale(frame, long_side=VIDEO_LONG_SIDE):
    """Fit into long_side, with even width and height (H.264 yuv420p needs them)."""
    h, w = frame.shape[:2]
    r = min(1.0, long_side / max(h, w))
    nw, nh = max(2, int(w * r) // 2 * 2), max(2, int(h * r) // 2 * 2)
    return frame if (nw, nh) == (w, h) else cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def open_h264_writer(path, size, fps):
    """An imageio-ffmpeg H.264 writer generator, or (None, reason) when imageio-ffmpeg is missing."""
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None, "imageio-ffmpeg is not available on this server, so no annotated video was made."
    gen = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264", pix_fmt_in="rgb24",
                                      pix_fmt_out="yuv420p", macro_block_size=2, quality=None,
                                      output_params=["-crf", "26", "-preset", "veryfast",
                                                     "-movflags", "+faststart"])
    gen.send(None)
    return gen, None


def video_duration(cap):
    """Seconds from the container's frame count and fps, or None if it does not say."""
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return n / fps if fps and fps > 0 and n and n > 0 else None


def process_video(path, detector, out_path=None, progress=None):
    """Sample about SAMPLE_FPS frames per second through detector, tracker, voter and suppressor.

    Frames are read one at a time from `path`. Video time drives the voter cooldown and the suppressor.
    Returns a dict: alerts [{t, time, track, conf}], suppressed [{t, time, track}], gallery
    [(JPEG bytes, caption)] of up to MAX_GALLERY annotated alert frames, gallery_meta [{t, time, conf}]
    (one per gallery frame; conf is the highest of the alerts fired in it), frames, tracks, duration,
    sample_fps, video (out_path or None) and note (why there is no video, or None).
    progress(fraction, text) is called after each sampled frame.
    """
    cap = cv2.VideoCapture(path)
    gen = None
    try:
        if not cap.isOpened():
            raise UploadError("Could not open this video. Try an MP4 (H.264) file.")
        dur = video_duration(cap)
        if dur is not None and dur > VIDEO_MAX_SECONDS + 0.5:
            raise UploadError(f"The video is {dur:.0f} s long; the limit is {VIDEO_MAX_SECONDS:.0f} s.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if 1 <= fps <= 240 else 30.0
        step = max(1, round(fps / SAMPLE_FPS))
        tracker, voter, suppressor = IoUTracker(), ViolationVoter(), AlertSuppressor()
        alerts, suppressed, gallery, gallery_meta, track_ids = [], [], [], [], set()
        note = None if out_path else "No annotated video was requested."
        idx = frames = 0
        while True:
            if idx % step:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            if t > VIDEO_MAX_SECONDS + 0.5:
                raise UploadError(f"The video is longer than {VIDEO_MAX_SECONDS:.0f} s.")
            frame = downscale(frame)
            dets = detector.detect(frame)
            ids = tracker.update([d.box for d in dets])
            track_ids.update(ids)
            tracked = [Tracked(i, d.box, d.conf, d.name) for i, d in zip(ids, dets)]
            decision = decide(tracked, voter, suppressor, now=t)
            draw(frame, dets, ids)
            for a in decision.fired:
                alerts.append({"t": round(t, 2), "time": fmt_time(t), "track": a.track_id,
                               "conf": round(a.conf, 3)})
            for s in decision.suppressed:
                suppressed.append({"t": round(t, 2), "time": fmt_time(t), "track": s.track_id})
            if decision.fired:
                banner(frame, f"ALERT {fmt_time(t)}  #" + " #".join(str(a.track_id) for a in decision.fired))
                if len(gallery) < MAX_GALLERY:
                    gallery.append((cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(),
                                    f"{fmt_time(t)} track " + ", ".join(f"#{a.track_id}" for a in decision.fired)))
                    gallery_meta.append({"t": round(t, 2), "time": fmt_time(t),
                                         "conf": round(max(a.conf for a in decision.fired), 3)})
            if out_path and gen is None and note is None:
                h, w = frame.shape[:2]
                gen, note = open_h264_writer(out_path, (w, h), fps / step)
            if gen is not None:
                gen.send(np.ascontiguousarray(frame[:, :, ::-1]).tobytes())
            frames += 1
            idx += 1
            if progress:
                progress(min(1.0, t / dur) if dur else min(1.0, t / VIDEO_MAX_SECONDS),
                         f"{fmt_time(t)} processed")
        if not frames:
            raise UploadError("No frames could be read from this video.")
    finally:
        cap.release()
        if gen is not None:
            gen.close()
    return {"alerts": alerts, "suppressed": suppressed, "gallery": gallery, "gallery_meta": gallery_meta,
            "frames": frames,
            "tracks": len(track_ids), "duration": round(idx / fps, 2), "sample_fps": round(fps / step, 2),
            "video": out_path if gen is not None else None, "note": note}
