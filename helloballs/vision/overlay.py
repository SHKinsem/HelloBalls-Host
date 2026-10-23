"""Drawing helpers shared by the CLI demo and the web console."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .detector import Detection
from .geometry import BallPosition

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    positions: Sequence[Optional[BallPosition]],
    show_ranges: bool = False,
) -> None:
    for i, (det, pos) in enumerate(zip(detections, positions)):
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, GREEN, 2)
        label = f"#{i + 1} {det.score:.2f}"
        if pos is not None:
            label += f"  {pos.range:.2f}m  az {pos.azimuth_deg:+.1f}"
            if show_ranges:
                detail = f"size {pos.range_size or 0:.2f}  gnd {pos.range_ground or 0:.2f}"
                cv2.putText(frame, detail, (p1[0], p2[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1)
        cv2.putText(frame, label, (p1[0], max(p1[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)


def draw_hud(frame: np.ndarray, text: str, warning: Optional[str] = None) -> None:
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
    if warning:
        cv2.putText(frame, warning, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
