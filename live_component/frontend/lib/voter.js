// Port of streamlit_app/voting.py (itself a copy of edge/voting.py): violation rule, k-of-n voting per
// track with a per-track cooldown, and duplicate suppression of re-IDs. No DOM. `now` is in seconds.

export const VIOLATION_CLASS = "no_helmet";
export const VIOLATION_CONF = 0.60; // a frame only votes "violation" for no_helmet at or above this

export function isViolation(name, conf) {
  return name === VIOLATION_CLASS && conf >= VIOLATION_CONF;
}

// A frame's vote only fires when min_hits of the last `window` votes of the same track are violations,
// and not again within `cooldown` seconds for that track. Tracks unseen for forgetAfter s are forgotten.
export class ViolationVoter {
  constructor({ window = 8, minHits = 5, cooldown = 60, forgetAfter = 30 } = {}) {
    if (!(minHits >= 1 && minHits <= window)) throw new Error("need 1 <= minHits <= window");
    this.window = window;
    this.minHits = minHits;
    this.cooldown = cooldown;
    this.forgetAfter = forgetAfter;
    this.tracks = new Map(); // id -> {votes: bool[], lastSeen, lastFired}
  }

  get size() {
    return this.tracks.size;
  }

  // Record one frame's vote for trackId; true if an alert should fire now.
  update(trackId, violation, now) {
    this._prune(now);
    let t = this.tracks.get(trackId);
    if (!t) {
      t = { votes: [], lastSeen: now, lastFired: null };
      this.tracks.set(trackId, t);
    }
    t.votes.push(Boolean(violation));
    if (t.votes.length > this.window) t.votes.shift();
    t.lastSeen = now;
    if (t.votes.filter(Boolean).length < this.minHits) return false;
    if (t.lastFired !== null && now - t.lastFired < this.cooldown) return false;
    t.lastFired = now;
    return true;
  }

  _prune(now) {
    for (const [tid, t] of [...this.tracks]) {
      if (now - t.lastSeen > this.forgetAfter) this.tracks.delete(tid);
    }
  }
}

// Drops an alert that repeats a recent one for the same person under a new track id: suppressed only if
// some remembered alert fired less than `seconds` ago, the new center is within radiusFactor x its width
// of that alert's center, AND that alert's track id is no longer active in the current frame (a re-ID;
// two people side by side both stay tracked, so a neighbor's alert is never dropped). Suppressed alerts
// are not remembered.
export class AlertSuppressor {
  constructor({ seconds = 30, radiusFactor = 1.0 } = {}) {
    this.seconds = seconds;
    this.radiusFactor = radiusFactor;
    this.recent = []; // [ts, trackId, cx, cy, w], oldest first
  }

  shouldSuppress(trackId, center, width, activeIds, now) {
    while (this.recent.length && now - this.recent[0][0] >= this.seconds) this.recent.shift();
    const [cx, cy] = center;
    for (const [, prevId, px, py, pw] of this.recent) {
      if (!activeIds.has(prevId) && Math.hypot(cx - px, cy - py) <= this.radiusFactor * pw) return true;
    }
    this.recent.push([now, trackId, cx, cy, width]);
    return false;
  }
}

// voting.decide: every tracked box votes, then fired alerts go through the suppressor with the set of
// track ids active in this frame. tracked: [{trackId, box, conf, name}]. Returns {fired, suppressed}.
export function decide(tracked, voter, suppressor, now) {
  const activeIds = new Set(tracked.map((t) => t.trackId));
  const fired = tracked.filter((t) => voter.update(t.trackId, isViolation(t.name, t.conf), now));
  const kept = [], suppressed = [];
  for (const t of fired) {
    const [x1, y1, x2, y2] = t.box;
    const center = [(x1 + x2) / 2, (y1 + y2) / 2];
    (suppressor.shouldSuppress(t.trackId, center, x2 - x1, activeIds, now) ? suppressed : kept).push(t);
  }
  return { fired: kept, suppressed };
}
