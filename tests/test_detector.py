from pathlib import Path

import numpy as np
import pytest

from helloballs.vision.detector import Letterbox, decode_yolo_output, letterbox

MODEL = Path(__file__).resolve().parents[1] / "scripts" / "yolov11_roboflow_ir9.onnx"


def make_output(dets, nc=3, n=100):
    """(1, 4 + nc, n) tensor with the given (cx, cy, w, h, cls, score) rows, rest zeros."""
    out = np.zeros((1, 4 + nc, n), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(dets):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


def test_decode_threshold_and_nms():
    out = make_output([
        (100, 100, 20, 20, 0, 0.9),
        (102, 101, 20, 20, 2, 0.8),  # same ball, other label -> suppressed (class agnostic)
        (300, 300, 30, 30, 0, 0.2),  # below threshold
    ])
    boxes, scores, cls = decode_yolo_output(out, 0.4, 0.5)
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0], [90, 90, 110, 110])
    assert scores[0] == pytest.approx(0.9)


def test_decode_class_filter_and_transposed_layout():
    out = make_output([(100, 100, 20, 20, 1, 0.9), (300, 300, 20, 20, 0, 0.7)])
    boxes, _, cls = decode_yolo_output(out.transpose(0, 2, 1), 0.4, 0.5, class_ids=[0])
    assert cls.tolist() == [0]
    np.testing.assert_allclose(boxes[0], [290, 290, 310, 310])


def test_letterbox_roundtrip():
    img = np.zeros((720, 1280, 3), np.uint8)
    canvas, lb = letterbox(img, 640)
    assert canvas.shape == (640, 640, 3)
    assert lb == Letterbox(scale=0.5, pad_x=0, pad_y=140)
    back = lb.to_original(np.array([[0, 140, 640, 500]], np.float32))
    np.testing.assert_allclose(back[0], [0, 0, 1280, 720])


@pytest.mark.skipif(not MODEL.exists(), reason="model not present")
def test_onnx_model_runs():
    from helloballs.vision.detector import OnnxDetector

    det = OnnxDetector(MODEL)
    assert det.ball_class_ids == [0, 1, 3, 5]
    assert det(np.zeros((480, 640, 3), np.uint8)) == []
