// Port of streamlit_app/tracker.py: a small greedy IoU tracker (no DOM).
//
// Each update() matches this frame's boxes to the existing tracks by IoU, best pairs first, one box per
// track; a pair needs IoU >= iouThres. Matching ignores the class (a head can flip between helmet_on and
// no_helmet and the voter needs all of its votes). Unmatched boxes start new tracks; ids count up from 1
// and are never reused. A track unmatched for more than maxAge updates is dropped.

export function iou(a, b) {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

export class IoUTracker {
  constructor({ iouThres = 0.3, maxAge = 15 } = {}) {
    this.iouThres = iouThres;
    this.maxAge = maxAge;
    this.nextId = 1;
    this.tracks = new Map(); // id -> {box, missed}
  }

  get activeIds() {
    return new Set(this.tracks.keys());
  }

  // Track ids for boxes ([x1, y1, x2, y2]), in the same order.
  update(boxes) {
    const pairs = [];
    for (const [tid, t] of this.tracks) {
      boxes.forEach((b, i) => pairs.push([iou(t.box, b), tid, i]));
    }
    pairs.sort((p, q) => q[0] - p[0] || p[1] - q[1] || p[2] - q[2]);
    const ids = new Array(boxes.length).fill(null);
    const used = new Set();
    for (const [score, tid, i] of pairs) {
      if (score < this.iouThres) break;
      if (used.has(tid) || ids[i] !== null) continue;
      ids[i] = tid;
      used.add(tid);
    }
    for (const [tid, t] of [...this.tracks]) {
      if (!used.has(tid)) {
        t.missed += 1;
        if (t.missed > this.maxAge) this.tracks.delete(tid);
      }
    }
    boxes.forEach((b, i) => {
      if (ids[i] === null) ids[i] = this.nextId++;
      this.tracks.set(ids[i], { box: [...b], missed: 0 });
    });
    return ids;
  }
}
