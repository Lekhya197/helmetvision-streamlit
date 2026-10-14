"""Violation rule, k-of-n voting and duplicate suppression for the Streamlit demo (no Streamlit here).

ViolationVoter and AlertSuppressor are copied unchanged from edge/voting.py (same rules as the edge
app; tests/test_streamlit_core.py checks the source is identical). decide() mirrors edge/pipeline.py:
every track in the frame votes, then fired alerts go through the suppressor with the set of track ids
active in this frame. `now` is video time in seconds, so cooldowns follow the video, not the server clock.
"""
import math
import time
from collections import deque, namedtuple

VIOLATION_CLASS = "no_helmet"
VIOLATION_CONF = 0.60  # a frame only votes "violation" for no_helmet at or above this

# box = xyxy in pixels; track_id from tracker.IoUTracker
Tracked = namedtuple("Tracked", "track_id box conf name")
FrameDecision = namedtuple("FrameDecision", "fired suppressed")  # lists of Tracked


def is_violation(name, conf):
    return name == VIOLATION_CLASS and conf >= VIOLATION_CONF


def decide(tracked, voter, suppressor, now):
    """Vote every tracked box, then drop fired alerts that are re-IDs of a recent alert."""
    active_ids = {t.track_id for t in tracked}
    fired = [t for t in tracked if voter.update(t.track_id, is_violation(t.name, t.conf), now=now)]
    kept, suppressed = [], []
    for t in fired:
        x1, y1, x2, y2 = t.box
        center, width = ((x1 + x2) / 2, (y1 + y2) / 2), x2 - x1
        if suppressor.should_suppress(t.track_id, center, width, active_ids, now=now):
            suppressed.append(t)
        else:
            kept.append(t)
    return FrameDecision(kept, suppressed)

class ViolationVoter:
    def __init__(self, window=8, min_hits=5, cooldown=60, forget_after=30):
        if not 1 <= min_hits <= window:
            raise ValueError("need 1 <= min_hits <= window")
        self.window = window
        self.min_hits = min_hits
        self.cooldown = cooldown
        self.forget_after = forget_after
        # track_id -> {"votes": deque[bool], "last_seen": float, "last_fired": float | None}
        self._tracks = {}

    def __len__(self):
        return len(self._tracks)

    def update(self, track_id, is_violation, now=None):
        """Record one frame's vote for `track_id`; return True if an alert should fire now."""
        now = time.time() if now is None else now
        self._prune(now)

        t = self._tracks.get(track_id)
        if t is None:
            t = {"votes": deque(maxlen=self.window), "last_seen": now, "last_fired": None}
            self._tracks[track_id] = t
        t["votes"].append(bool(is_violation))
        t["last_seen"] = now

        if sum(t["votes"]) < self.min_hits:
            return False
        if t["last_fired"] is not None and now - t["last_fired"] < self.cooldown:
            return False
        t["last_fired"] = now
        return True

    def _prune(self, now):
        stale = [tid for tid, t in self._tracks.items() if now - t["last_seen"] > self.forget_after]
        for tid in stale:
            del self._tracks[tid]


class AlertSuppressor:
    """Drops an alert that repeats a recent one for the same person under a new track ID.

    ByteTrack can give one person several IDs in a row (occlusion, missed frames), and each new ID
    gets its own voter cooldown. Every alert that actually fires is remembered as
    (time, track id, center, width). A new alert is suppressed only if, for some remembered alert,
    all of these hold:
      - it fired less than `seconds` ago,
      - the new center lies within `radius_factor` x the earlier box width of the earlier center,
      - the earlier track id is no longer active in the current frame.
    The last condition is what a re-ID looks like; two people standing side by side both stay
    tracked, so a neighbor's real alert is never dropped. Suppressed alerts are not remembered, so a
    person who stays in violation is re-alerted once `seconds` have passed.
    """

    def __init__(self, seconds=30, radius_factor=1.0):
        self.seconds = seconds
        self.radius_factor = radius_factor
        self._recent = deque()  # (ts, track_id, cx, cy, w), oldest first

    def should_suppress(self, track_id, center, width, active_ids, now=None):
        """True if this alert is a re-ID duplicate; otherwise remember it and return False."""
        now = time.time() if now is None else now
        while self._recent and now - self._recent[0][0] >= self.seconds:
            self._recent.popleft()
        cx, cy = center
        for _, prev_id, px, py, pw in self._recent:
            if (prev_id not in active_ids
                    and math.hypot(cx - px, cy - py) <= self.radius_factor * pw):
                return True
        self._recent.append((now, track_id, cx, cy, width))
        return False
