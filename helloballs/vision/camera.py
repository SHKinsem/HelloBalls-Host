"""Frame sources: USB camera, video file, or a single image."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def open_usb_camera(
    device: Union[int, str] = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    mjpg: bool = True,
) -> cv2.VideoCapture:
    """Open a USB (UVC) camera with the backend that fits the platform.

    On Windows DirectShow honours resolution requests far more reliably than MSMF;
    on Linux (RDK X5) V4L2 is used and `device` may also be a path like /dev/video0.
    """
    if platform.system() == "Windows":
        backend = cv2.CAP_DSHOW
    elif platform.system() == "Linux":
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(device, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {device!r}")

    # Most UVC cameras only reach full resolution / frame rate in MJPG mode.
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class FrameSource:
    """Endless frame reader: cameras stream, video files loop, images repeat."""

    def __init__(self, source: str, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.source = source
        self._image: Optional[np.ndarray] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self.is_live = False

        if Path(source).suffix.lower() in IMAGE_SUFFIXES:
            self._image = cv2.imread(source)
            if self._image is None:
                raise RuntimeError(f"Cannot read image {source}")
        elif source.isdigit() or source.startswith("/dev/video"):
            self._cap = open_usb_camera(int(source) if source.isdigit() else source, width, height)
            self.is_live = True
        else:
            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video {source}")

    def read(self) -> Optional[np.ndarray]:
        if self._image is not None:
            return self._image.copy()
        ok, frame = self._cap.read()
        if not ok and not self.is_live:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


def frames(
    source: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Yield BGR frames from a camera index ("0"), device path, video file or image."""
    path = Path(source)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Cannot read image {source}")
        yield image
        return

    if source.isdigit() or source.startswith("/dev/video"):
        cap = open_usb_camera(int(source) if source.isdigit() else source, width, height)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()
