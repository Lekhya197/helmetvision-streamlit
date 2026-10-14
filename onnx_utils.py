"""ONNX helmet detector for the Streamlit demo: onnxruntime only, no torch, no Ultralytics, no Streamlit.

The math below (letterbox_params .. detect) is copied unchanged from scripts/onnx_ref.py, the reference
that was checked against the .pt model (scripts/check_onnx_parity.py); tests/test_streamlit_core.py
checks that both give the same output. Output layout of helmet.onnx: raw [1, 4 + nc, N] (cx, cy, w, h,
then one score per class), NMS done here. Class names come from model/classes.json, never fixed ids.
"""
import json
import pathlib
from collections import namedtuple

import cv2
import numpy as np

PAD_VALUE = 114


def letterbox_params(h, w, size):
    """(scale, (new_w, new_h), (left, top, right, bottom)) for an h x w image, as Ultralytics LetterBox."""
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    return r, (new_w, new_h), (left, top, right, bottom)


def letterbox(img_bgr, size):
    """(NCHW float32 RGB 0..1 tensor, scale, (left, top)) for one BGR uint8 image."""
    h, w = img_bgr.shape[:2]
    r, (new_w, new_h), (left, top, right, bottom) = letterbox_params(h, w, size)
    if (new_w, new_h) != (w, h):
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))
    x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, (left, top)


def boxes_to_original(boxes, scale, pad, orig_shape):
    """Letterbox-pixel xyxy boxes (N, 4) -> original-image pixels, clipped to the image."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    left, top = pad
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / scale).clip(0, w)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / scale).clip(0, h)
    return boxes


def detect_layout(shape, nc):
    """"end2end" for [1, N, 6], "raw" for [1, 4 + nc, N]; raises on anything else."""
    shape = tuple(shape)
    if len(shape) == 3 and shape[0] == 1 and shape[2] == 6:
        return "end2end"
    if len(shape) == 3 and shape[0] == 1 and shape[1] == 4 + nc:
        return "raw"
    raise ValueError(f"unknown output shape {shape} for {nc} classes")


def nms(boxes, scores, iou_thres):
    """Indices kept by greedy NMS on xyxy boxes, highest score first."""
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        union = area(boxes[i]) + area(boxes[rest]) - inter
        iou = np.where(union > 0, inter / np.where(union > 0, union, 1), 0)
        order = rest[iou <= iou_thres]
    return keep


def postprocess(out, nc, conf=0.25, iou_thres=0.7):
    """(boxes xyxy in letterbox pixels (N, 4), scores (N,), class ids (N,)), highest score first."""
    out = np.asarray(out, dtype=np.float32)
    layout = detect_layout(out.shape, nc)
    if layout == "end2end":
        rows = out[0]
        boxes, scores, cls = rows[:, :4], rows[:, 4], rows[:, 5].round().astype(int)
    else:
        rows = out[0].T  # (N, 4 + nc)
        cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        cls_scores = rows[:, 4:]
        cls = cls_scores.argmax(axis=1) if len(rows) else np.zeros(0, dtype=int)
        scores = cls_scores.max(axis=1) if len(rows) else np.zeros(0, dtype=np.float32)
    m = scores >= conf
    boxes, scores, cls = boxes[m], scores[m], cls[m]
    if layout == "raw" and len(boxes):
        keep = []
        for c in np.unique(cls):
            idx = np.flatnonzero(cls == c)
            keep += [idx[k] for k in nms(boxes[idx], scores[idx], iou_thres)]
        keep = np.array(keep, dtype=int)
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    order = np.argsort(-scores, kind="stable")
    return boxes[order].reshape(-1, 4), scores[order], cls[order]


def detect(session, img_bgr, size, nc, conf=0.25, iou_thres=0.7):
    """Run one onnxruntime session on one BGR image: (boxes in original pixels, scores, class ids)."""
    x, scale, pad = letterbox(img_bgr, size)
    out = session.run(None, {session.get_inputs()[0].name: x})[0]
    boxes, scores, cls = postprocess(out, nc, conf, iou_thres)
    return boxes_to_original(boxes, scale, pad, img_bgr.shape), scores, cls


IMGSZ = 416
DETECT_CONF = 0.25  # boxes below this are not shown, tracked or voted
MODEL_DIR = pathlib.Path(__file__).resolve().parent / "model"

# box = (x1, y1, x2, y2) in original image pixels
Detection = namedtuple("Detection", "box conf cls name")


def load_classes(path):
    """{class id: name} from classes.json ({"0": "helmet_on", ...})."""
    return {int(k): str(v) for k, v in json.loads(pathlib.Path(path).read_text()).items()}


class OnnxDetector:
    """One onnxruntime CPU session; detect() returns Detections sorted by confidence."""

    def __init__(self, model_path=None, classes_path=None, threads=2):
        """Defaults: model/helmet.onnx and model/classes.json next to this file, whatever the working
        directory. onnxruntime is imported here, so a missing install fails at load, not at import."""
        model_path = pathlib.Path(model_path or MODEL_DIR / "helmet.onnx")
        classes_path = pathlib.Path(classes_path or MODEL_DIR / "classes.json")
        for p in (model_path, classes_path):
            if not p.is_file():
                raise FileNotFoundError(f"missing model file {p.name} in {p.parent}")
        import onnxruntime as ort
        self.names = load_classes(classes_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])

    def detect(self, img_bgr, conf=DETECT_CONF):
        boxes, scores, cls = detect(self.session, img_bgr, IMGSZ, len(self.names), conf=conf)
        return [Detection(tuple(float(v) for v in b), float(s), int(c), self.names[int(c)])
                for b, s, c in zip(boxes, scores, cls)]
