"""Checkerboard intrinsic calibration shared by the CLI tool and the web console."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .geometry import CameraModel

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
MIN_VIEWS = 10


def find_corners(gray: np.ndarray, pattern: tuple[int, int]) -> Optional[np.ndarray]:
    found, corners = cv2.findChessboardCorners(gray, pattern, FIND_FLAGS)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)


@dataclass
class CalibrationResult:
    camera: CameraModel
    rms: float
    per_view: list[float]

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.camera.width / (2 * self.camera.K[0, 0]))))

    def summary(self) -> dict:
        K = self.camera.K
        return {
            "rms_px": round(self.rms, 3),
            "views": len(self.per_view),
            "fx": round(float(K[0, 0]), 1),
            "fy": round(float(K[1, 1]), 1),
            "cx": round(float(K[0, 2]), 1),
            "cy": round(float(K[1, 2]), 1),
            "hfov_deg": round(self.hfov_deg, 1),
            "worst_views": [int(i) + 1 for i in np.argsort(self.per_view)[::-1][:3]],
        }


def calibrate(
    image_points: list[np.ndarray], image_size: tuple[int, int], pattern: tuple[int, int], square_m: float
) -> CalibrationResult:
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square_m
    object_points = [objp] * len(image_points)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

    per_view = []
    for imgp, rvec, tvec in zip(image_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj - imgp) ** 2, axis=2)))))
    return CalibrationResult(CameraModel(K, dist.reshape(-1), image_size[0], image_size[1]), float(rms), per_view)


@dataclass
class CalibrationSession:
    """Accumulates checkerboard views captured from a live stream."""

    pattern: tuple[int, int] = (9, 6)
    square_m: float = 0.025
    views: list[np.ndarray] = field(default_factory=list)
    image_size: Optional[tuple[int, int]] = None

    def add(self, corners: np.ndarray, image_size: tuple[int, int]) -> None:
        if self.image_size is not None and image_size != self.image_size:
            self.views.clear()  # resolution changed, old views are meaningless
        self.image_size = image_size
        self.views.append(corners)

    def clear(self) -> None:
        self.views.clear()
        self.image_size = None

    def run(self) -> CalibrationResult:
        if len(self.views) < MIN_VIEWS:
            raise ValueError(f"need at least {MIN_VIEWS} views, have {len(self.views)}")
        return calibrate(self.views, self.image_size, self.pattern, self.square_m)
