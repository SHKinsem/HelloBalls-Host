"""Tennis ball detection.

Only the inference step is platform specific. Letterboxing and box decoding are
shared so that the PC (onnxruntime) and RDK X5 (BPU) backends give the same results.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Letterbox:
    """Mapping between the original frame and the square network input."""

    scale: float
    pad_x: int
    pad_y: int

    def to_original(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        boxes = boxes_xyxy.copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - self.pad_x) / self.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - self.pad_y) / self.scale
        return boxes


def letterbox(image: np.ndarray, size: int, pad_value: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping aspect ratio and pad to `size` x `size` (Ultralytics convention)."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, Letterbox(scale, pad_x, pad_y)


def decode_yolo_output(
    output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    class_ids: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a standard Ultralytics YOLOv8/11 detect output.

    `output` is (1, 4 + nc, N) or (1, N, 4 + nc): cx, cy, w, h in input pixels followed
    by per-class scores that are *already* sigmoid-activated.
    Returns boxes (K, 4) xyxy in input pixels, scores (K,), class ids (K,).
    """
    pred = output[0]
    if pred.shape[0] > pred.shape[1]:  # (N, 4 + nc) layout
        pred = pred.T
    boxes_cxcywh = pred[:4].T
    scores_all = pred[4:].T

    if class_ids is not None:
        keep_cols = np.asarray(class_ids)
        cls_local = scores_all[:, keep_cols].argmax(axis=1)
        cls = keep_cols[cls_local]
    else:
        cls = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(cls)), cls]

    mask = scores >= conf_threshold
    boxes_cxcywh, scores, cls = boxes_cxcywh[mask], scores[mask], cls[mask]
    if len(scores) == 0:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)

    boxes = np.empty_like(boxes_cxcywh)
    boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    # Class-agnostic NMS: every kept class is "a ball", duplicates across labels are one object.
    xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]])
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    keep = np.asarray(keep, dtype=np.int64).reshape(-1)
    return boxes[keep], scores[keep], cls[keep]


class OnnxDetector:
    """onnxruntime backend for development on PC."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        class_names: Optional[Sequence[str]] = None,
        ball_class_ids: Optional[Sequence[int]] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), providers=list(providers or ["CPUExecutionProvider"])
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[2])
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.class_names = list(class_names) if class_names else self._names_from_metadata()
        if ball_class_ids is None and self.class_names:
            ball_class_ids = [i for i, n in enumerate(self.class_names) if "ball" in n.lower()]
        self.ball_class_ids = list(ball_class_ids) if ball_class_ids else None

    def _names_from_metadata(self) -> list[str]:
        # Ultralytics stores {0: 'name', ...} as a string in the ONNX metadata.
        raw = self.session.get_modelmeta().custom_metadata_map.get("names")
        if not raw:
            return []
        names = ast.literal_eval(raw)
        return [names[i] for i in sorted(names)]

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        canvas, lb = letterbox(frame_bgr, self.input_size)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
        output = self.session.run(None, {self.input_name: blob})[0]

        boxes, scores, cls = decode_yolo_output(
            output, self.conf_threshold, self.iou_threshold, self.ball_class_ids
        )
        boxes = lb.to_original(boxes)
        h, w = frame_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        return [
            Detection(*map(float, b), float(s), int(c), self._name(int(c)))
            for b, s, c in zip(boxes, scores, cls)
        ]

    def _name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)
