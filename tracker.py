"""A small greedy IoU tracker for the Streamlit video tab (no Ultralytics, no ByteTrack).

Each update() matches this frame's boxes to the existing tracks by IoU, best pairs first, one box per
track. A pair needs IoU >= iou_thres. Matching ignores the class, because one person's head can flip
between helmet_on and no_helmet from frame to frame and the voter needs all of that person's votes.
Unmatched boxes start new tracks; ids count up from 1 and are never reused. A track that goes
unmatched for more than max_age updates is dropped.
"""


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    def __init__(self, iou_thres=0.3, max_age=15):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self._next_id = 1
        self._tracks = {}  # id -> {"box": xyxy, "missed": updates since the last match}

    @property
    def active_ids(self):
        return set(self._tracks)

    def update(self, boxes):
        """Track ids for `boxes` (xyxy), in the same order."""
        pairs = sorted(((iou(t["box"], b), tid, i)
                        for tid, t in self._tracks.items() for i, b in enumerate(boxes)),
                       key=lambda p: (-p[0], p[1], p[2]))
        ids = [None] * len(boxes)
        used = set()
        for score, tid, i in pairs:
            if score < self.iou_thres:
                break
            if tid in used or ids[i] is not None:
                continue
            ids[i] = tid
            used.add(tid)
        for tid, t in list(self._tracks.items()):
            if tid not in used:
                t["missed"] += 1
                if t["missed"] > self.max_age:
                    del self._tracks[tid]
        for i, b in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self._next_id
                self._next_id += 1
            self._tracks[ids[i]] = {"box": tuple(b), "missed": 0}
        return ids
