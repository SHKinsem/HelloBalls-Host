from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import import_module
from typing import Optional

cv2 = None
_cv2_import_error: Optional[Exception] = None


@dataclass(frozen=True)
class CameraConfig:
    camera_device: str = "/dev/video0"
    width: int = 2560
    height: int = 720
    buffer_size: int = 1
    fourcc: Optional[str] = "MJPG"
    fps: Optional[float] = None


@dataclass(frozen=True)
class CameraFrame:
    image: object
    camera_device: str
    captured_at: float
    width: int
    height: int


class OpenCVCamera:
    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self.camera_device: Optional[str] = None
        self._capture = None

    def open(self) -> str:
        camera_device = self.config.camera_device
        capture = _open_capture(camera_device)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open OpenCV camera {camera_device}.")

        _configure_capture(capture, self.config)

        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"Camera {camera_device} opened but did not return a frame.")

        self.camera_device = camera_device
        self._capture = capture
        return camera_device

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self.camera_device = None

    def read(self) -> Optional[CameraFrame]:
        if self._capture is None or self.camera_device is None:
            self.open()

        ok, image = self._capture.read()
        if not ok or image is None:
            return None

        height, width = image.shape[:2]
        return CameraFrame(
            image=image,
            camera_device=self.camera_device,
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


def _open_capture(camera_device: str):
    cv = _require_cv2()
    return cv.VideoCapture(camera_device, cv.CAP_V4L2)


def _configure_capture(capture, config: CameraConfig) -> None:
    cv = _require_cv2()
    if config.fourcc:
        if len(config.fourcc) != 4:
            raise RuntimeError(f"Camera FOURCC must be four characters, got {config.fourcc!r}.")
        capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*config.fourcc))
    capture.set(cv.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, config.height)
    if config.fps is not None:
        capture.set(cv.CAP_PROP_FPS, config.fps)
    capture.set(cv.CAP_PROP_BUFFERSIZE, config.buffer_size)
