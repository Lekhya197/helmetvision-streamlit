"""Owner-only Telegram alerts for the hosted demo (pure Python, no Streamlit here).

send_alert() posts one message (sendPhoto with a JPEG, or sendMessage without) and never raises. The
bot token is part of the API URL, so every message this module returns or logs is masked first, and
exceptions from requests are never re-raised or logged as is. RateLimiter (per-session cooldown plus a
global hourly cap) and AttemptGuard (wrong access codes) take an injectable clock for tests.
parse_live_event() validates the alert events the Live component sends in owner mode.
"""
import base64
import binascii
import hmac
import logging
import re
import threading
import time
from collections import deque

import requests

API = "https://api.telegram.org"
TIMEOUT = (5, 15)  # (connect, read) seconds
RETRY_DELAY = 1.0
CAPTION_MAX = 1024  # Telegram's limit for photo captions
LIVE_THUMB_MAX = 60_000  # bytes after base64 decoding; live.js aims for about 40 KB
_TOKEN_LIKE = re.compile(r"bot\d+:[\w-]+")  # the shape of a bot token inside an API URL

log = logging.getLogger("telegram_alerts")


def mask(text, token):
    """`text` with the token (and anything shaped like bot<token>) replaced by ***."""
    text = str(text)
    if token:
        text = text.replace(str(token), "***")
    return _TOKEN_LIKE.sub("bot***", text)


def send_alert(token, chat, caption, jpeg_bytes=None, *, post=None, sleep=time.sleep):
    """Send `caption` (with the photo if jpeg_bytes) to `chat`. Returns (ok, message); never raises.

    Network errors (connection failure, timeout) are retried once; HTTP errors are not. `post` and
    `sleep` are for tests (default requests.post, time.sleep).
    """
    try:
        if not token or not chat:
            return False, "not configured"
        post = post or requests.post
        caption = str(caption)[:CAPTION_MAX]
        method = "sendPhoto" if jpeg_bytes else "sendMessage"
        url = f"{API}/bot{token}/{method}"
        for attempt in (1, 2):
            try:
                if jpeg_bytes:
                    resp = post(url, data={"chat_id": str(chat), "caption": caption},
                                files={"photo": ("alert.jpg", jpeg_bytes, "image/jpeg")}, timeout=TIMEOUT)
                else:
                    resp = post(url, data={"chat_id": str(chat), "text": caption}, timeout=TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 1:
                    log.warning("telegram network error (%s); retrying once", type(e).__name__)
                    sleep(RETRY_DELAY)
                    continue
                msg = f"network error after retry ({type(e).__name__})"
                log.warning("telegram %s", msg)
                return False, msg
            if resp.status_code == 200:
                return True, "sent"
            msg = mask(f"HTTP {resp.status_code}: {str(resp.text)[:200]}", token)
            log.warning("telegram returned %s", msg)
            return False, msg
    except Exception as e:  # anything else: only the type name and masked text, never a traceback
        msg = mask(f"failed ({type(e).__name__}: {e})", token)
        log.warning("telegram %s", msg)
        return False, msg
    return False, "failed"


class RateLimiter:
    """At most one send per `cooldown` seconds per session and `hourly_cap` sends per hour overall.

    Thread-safe; one instance is shared by all sessions. acquire() claims a slot before the send, so a
    failed send still counts (no retry floods).
    """

    def __init__(self, cooldown=60.0, hourly_cap=10, clock=time.monotonic):
        self.cooldown, self.hourly_cap, self.clock = cooldown, hourly_cap, clock
        self._lock = threading.Lock()
        self._last = {}  # session id -> time of its last send
        self._sends = deque()  # times of all sends in the last hour

    def acquire(self, session_id):
        """(True, "ok") and record a send, or (False, reason) when rate limited."""
        with self._lock:
            now = self.clock()
            while self._sends and now - self._sends[0] >= 3600:
                self._sends.popleft()
            for sid in [s for s, t in self._last.items() if now - t >= self.cooldown]:
                del self._last[sid]  # keeps the dict small
            if session_id in self._last:
                wait = self.cooldown - (now - self._last[session_id])
                return False, f"wait {wait:.0f} s between alerts"
            if len(self._sends) >= self.hourly_cap:
                return False, f"hourly limit of {self.hourly_cap} reached"
            self._last[session_id] = now
            self._sends.append(now)
            return True, "ok"



def code_matches(entered, expected):
    """Constant-time comparison; False when either is empty."""
    if not entered or not expected:
        return False
    return hmac.compare_digest(str(entered).encode("utf-8"), str(expected).encode("utf-8"))


class AttemptGuard:
    """Access-code checks for one session: after `max_tries` wrong codes, refuse for `lockout` seconds."""

    def __init__(self, max_tries=5, lockout=60.0, clock=time.monotonic):
        self.max_tries, self.lockout, self.clock = max_tries, lockout, clock
        self.wrong = 0
        self.locked_until = None

    def locked_for(self):
        """Seconds left in the lockout, or 0."""
        if self.locked_until is None:
            return 0.0
        left = self.locked_until - self.clock()
        if left <= 0:
            self.locked_until, self.wrong = None, 0
            return 0.0
        return left

    def check(self, entered, expected):
        """"ok", "wrong" or "locked". A locked guard never compares the code."""
        if self.locked_for() > 0:
            return "locked"
        if code_matches(entered, expected):
            self.wrong = 0
            return "ok"
        self.wrong += 1
        if self.wrong >= self.max_tries:
            self.locked_until = self.clock() + self.lockout
        return "wrong"


def parse_live_event(value):
    """{"id", "conf", "ts", "jpeg"} from a Live component alert event, or None if it is not one.

    The value comes from a browser, so everything is checked. A thumbnail that is not a JPEG, cannot be
    decoded or is over LIVE_THUMB_MAX is dropped (jpeg None) and the alert is kept.
    """
    if not isinstance(value, dict):
        return None
    eid, conf, ts = value.get("id"), value.get("conf"), value.get("ts")
    if not isinstance(eid, str) or not 0 < len(eid) <= 64:
        return None
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return None
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not 0 < ts < 1e14:
        ts = None  # milliseconds since 1970, as Date.now() gives
    jpeg = None
    b64 = value.get("thumb_jpeg_base64")
    if isinstance(b64, str) and len(b64) <= LIVE_THUMB_MAX * 4 // 3 + 4:
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            data = b""
        if data.startswith(b"\xff\xd8") and len(data) <= LIVE_THUMB_MAX:
            jpeg = data
    return {"id": eid, "conf": float(conf), "ts": ts, "jpeg": jpeg}
