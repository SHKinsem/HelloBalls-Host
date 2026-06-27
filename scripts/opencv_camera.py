from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from importlib import import_module
from typing import Optional

cv2 = None
_cv2_import_error: Optional[Exception] = None


@dataclass(frozen=True)
class CameraConfig:
    camera_id: Optional[int] = None
    width: int = 1280
    height: int = 720
    buffer_size: int = 1
    search_count: int = 10


@dataclass(frozen=True)
class CameraFrame:
    image: object
    camera_id: int
    captured_at: float
    width: int
    height: int


class OpenCVCamera:
    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self.camera_id: Optional[int] = None
        self._capture = None

    @staticmethod
    def find_available_camera(search_count: int = 10) -> Optional[int]:
        _require_cv2()

        for camera_id in range(search_count):
            if _can_read_camera(camera_id):
                return camera_id

        if os.path.exists("/dev/"):
            for device in sorted(glob.glob("/dev/video*")):
                try:
                    camera_id = int(device.removeprefix("/dev/video"))
                except ValueError:
                    continue
                if _can_read_camera(camera_id):
                    return camera_id

        return None

    def open(self) -> int:
        cv = _require_cv2()

        camera_id = self.config.camera_id
        if camera_id is None:
            camera_id = self.find_available_camera(self.config.search_count)
        if camera_id is None:
            raise RuntimeError("No readable OpenCV camera was found.")

        capture = cv.VideoCapture(camera_id)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open OpenCV camera {camera_id}.")

        capture.set(cv.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"Camera {camera_id} opened but did not return a frame.")

        self.camera_id = camera_id
        self._capture = capture
        return camera_id

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self.camera_id = None

    def read(self) -> Optional[CameraFrame]:
        if self._capture is None or self.camera_id is None:
            self.open()

        ok, image = self._capture.read()
        if not ok or image is None:
            return None

        height, width = image.shape[:2]
        return CameraFrame(
            image=image,
            camera_id=self.camera_id,
            captured_at=time.time(),
            width=width,
            height=height,
        )

    def actual_resolution(self) -> tuple[int, int]:
        if self._capture is None:
            return 0, 0
        cv = _require_cv2()
        width = int(self._capture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def __enter__(self) -> OpenCVCamera:
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def get_cv2():
    return _require_cv2()


def _require_cv2():
    global cv2, _cv2_import_error

    if cv2 is None:
        try:
            cv2 = import_module("cv2")
        except Exception as exc:
            _cv2_import_error = exc
            raise RuntimeError(
                "OpenCV could not be imported. Install or repair opencv-python for the active Python environment."
            ) from exc
    return cv2


def _can_read_camera(camera_id: int) -> bool:
    cv = _require_cv2()
    capture = cv.VideoCapture(camera_id)
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None)
    finally:
        capture.release()
