"""Browser console: live view, detection tuning, calibration and snapshots.

    python -m helloballs.ui.server                 # http://127.0.0.1:8000, opens the browser
    python -m helloballs.ui.server --host 0.0.0.0  # on the RDK X5, open http://<board-ip>:8000 from a laptop

Standard library HTTP server only, so it runs on the board without extra packages.
The page streams annotated frames as MJPEG and polls /api/state for numbers.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..vision.calibration import MIN_VIEWS, CalibrationResult, CalibrationSession, find_corners
from ..vision.camera import FrameSource
from ..vision.detector import OnnxDetector
from ..vision.geometry import CameraModel, Mount, floor_xy, locate_ball
from ..vision.overlay import draw_detections, draw_hud

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
CAMERA_DIR = ROOT / "configs" / "cameras"
CAPTURE_DIR = ROOT / "captures"
CALIB_IMG_DIR = ROOT / "calib_imgs"
MODEL_DIRS = [ROOT / "models", ROOT / "scripts"]

DEFAULT_SETTINGS: dict[str, Any] = {
    "source": "0",
    "width": 1280,
    "height": 720,
    "model": "scripts/yolov11_roboflow_ir9.onnx",
    "conf": 0.4,
    "camera_yaml": "",
    "hfov": 70.0,
    "mount_enabled": False,
    # Robot camera mount from the localization branch's mechanical extrinsics.
    "mount_height": 0.142,
    "mount_pitch": 0.0,
    "mode": "detect",  # detect | calibrate
    "calib_cols": 9,
    "calib_rows": 6,
    "calib_square": 0.025,
}

# Changing these requires reopening the frame source / reloading the model.
SOURCE_KEYS = {"source", "width", "height"}
MODEL_KEYS = {"model"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


class Console:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = dict(settings)
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition()
        self.jpeg: Optional[bytes] = None
        self.frame_id = 0
        self.raw: Optional[np.ndarray] = None
        self.corners: Optional[np.ndarray] = None
        self.status: dict[str, Any] = {}
        self.error: Optional[str] = None

        self.source: Optional[FrameSource] = None
        self.detector: Optional[OnnxDetector] = None
        self.base_camera: Optional[CameraModel] = None
        self.camera: Optional[CameraModel] = None
        self.calib = CalibrationSession()
        self.calib_result: Optional[CalibrationResult] = None
        self._reload = {"source", "model", "camera"}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="console", daemon=True)

    # ---------- lifecycle ----------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        if self.source:
            self.source.release()

    def update_settings(self, changes: dict[str, Any]) -> None:
        with self.lock:
            for key, value in changes.items():
                if key not in DEFAULT_SETTINGS:
                    continue
                kind = type(DEFAULT_SETTINGS[key])
                self.settings[key] = value if kind is str else kind(value)
                if key in SOURCE_KEYS:
                    self._reload.add("source")
                if key in MODEL_KEYS:
                    self._reload.add("model")
                if key in {"camera_yaml", "hfov"}:
                    self._reload.add("camera")
            pattern = (self.settings["calib_cols"], self.settings["calib_rows"])
            if pattern != self.calib.pattern:
                self.calib.clear()
            self.calib.pattern = pattern
            self.calib.square_m = self.settings["calib_square"]

    def _apply_reloads(self) -> None:
        with self.lock:
            pending, self._reload = self._reload, set()
            s = dict(self.settings)
        try:
            if "source" in pending:
                if self.source:
                    self.source.release()
                self.source = None
                self.source = FrameSource(s["source"], s["width"], s["height"])
            if "model" in pending:
                self.detector = OnnxDetector(ROOT / s["model"])
            if "camera" in pending:
                self.base_camera = CameraModel.load(ROOT / s["camera_yaml"]) if s["camera_yaml"] else None
                self.camera = None
            self.error = None
        except Exception as e:  # surface to the page instead of killing the thread
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    # ---------- processing loop ----------

    def _loop(self) -> None:
        fps = 0.0
        last = time.perf_counter()
        while not self._stop.is_set():
            if self._reload:
                self._apply_reloads()
            if self.source is None or self.detector is None:
                self._publish_placeholder(self.error or "no source")
                time.sleep(0.5)
                continue

            frame = self.source.read()
            if frame is None:
                self.error = "frame read failed"
                time.sleep(0.2)
                continue

            with self.lock:
                s = dict(self.settings)
            h, w = frame.shape[:2]
            if self.camera is None or (self.camera.width, self.camera.height) != (w, h):
                self.camera = (
                    self.base_camera.scaled_to(w, h) if self.base_camera else CameraModel.from_fov(w, h, s["hfov"])
                )

            raw = frame.copy()
            t0 = time.perf_counter()
            if s["mode"] == "calibrate":
                balls, infer_ms = [], 0.0
                self._process_calibration(frame)
            else:
                self.detector.conf_threshold = s["conf"]
                balls, infer_ms = self._process_detection(frame, s), (time.perf_counter() - t0) * 1000

            now = time.perf_counter()
            dt, last = now - last, now
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3) if fps else 1 / max(dt, 1e-3)

            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self.frame_ready:
                self.raw = raw
                self.jpeg = jpg.tobytes() if ok else self.jpeg
                self.frame_id += 1
                self.status = {
                    "fps": round(fps, 1),
                    "infer_ms": round(infer_ms, 1),
                    "frame_size": [w, h],
                    "calibrated": self.camera.calibrated,
                    "hfov_deg": round(float(np.degrees(2 * np.arctan(w / (2 * self.camera.K[0, 0])))), 1),
                    "balls": balls,
                }
                self.frame_ready.notify_all()

            if not self.source.is_live:
                time.sleep(1 / 30)  # don't spin on images / play videos at ~real time

    def _process_detection(self, frame: np.ndarray, s: dict[str, Any]) -> list[dict]:
        mount = Mount(s["mount_height"], s["mount_pitch"]) if s["mount_enabled"] else None
        detections = self.detector(frame)
        positions = [locate_ball(self.camera, (d.x1, d.y1, d.x2, d.y2), mount) for d in detections]

        draw_detections(frame, detections, positions, show_ranges=mount is not None)
        if not self.camera.calibrated:
            draw_hud(frame, "", "UNCALIBRATED")

        balls = []
        for d, p in zip(detections, positions):
            ball = {"score": round(d.score, 3), "label": d.class_name, "box": [round(v, 1) for v in (d.x1, d.y1, d.x2, d.y2)]}
            if p is not None:
                right, forward = floor_xy(p.xyz, mount)
                ball.update(
                    range=round(p.range, 3),
                    range_size=None if p.range_size is None else round(p.range_size, 3),
                    range_ground=None if p.range_ground is None else round(p.range_ground, 3),
                    azimuth=round(p.azimuth_deg, 2),
                    elevation=round(p.elevation_deg, 2),
                    xyz=[round(float(v), 3) for v in p.xyz],
                    floor=[round(right, 3), round(forward, 3)],
                )
            balls.append(ball)
        return balls

    def _process_calibration(self, frame: np.ndarray) -> None:
        # A full-resolution search is slow when no board is visible, so gate it on a downscaled copy.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, 640 / gray.shape[1])
        small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1 else gray
        board_visible = find_corners(small, self.calib.pattern) is not None
        self.corners = find_corners(gray, self.calib.pattern) if board_visible else None

        for view in self.calib.views:  # coverage of captured views
            for c in view[:: max(1, len(view) // 12)]:
                cv2.circle(frame, tuple(int(x) for x in c.ravel()), 4, (255, 128, 0), -1)
        if self.corners is not None:
            cv2.drawChessboardCorners(frame, self.calib.pattern, self.corners, True)
        text = f"views {len(self.calib.views)}/{MIN_VIEWS}+  board {'OK' if self.corners is not None else '--'}"
        draw_hud(frame, text)

    def _publish_placeholder(self, message: str) -> None:
        img = np.zeros((360, 640, 3), np.uint8)
        cv2.putText(img, message[:70], (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        with self.frame_ready:
            self.jpeg = cv2.imencode(".jpg", img)[1].tobytes()
            self.frame_id += 1
            self.frame_ready.notify_all()

    # ---------- actions ----------

    def snapshot(self, folder: Path) -> str:
        with self.frame_ready:
            raw = None if self.raw is None else self.raw.copy()
        if raw is None:
            raise RuntimeError("no frame yet")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{datetime.now():%Y%m%d-%H%M%S-%f}"[:-3]
        path = path.with_suffix(".jpg")
        cv2.imwrite(str(path), raw)
        return rel(path)

    def calib_capture(self) -> int:
        corners = self.corners
        if corners is None:
            raise RuntimeError("checkerboard not detected in the current frame")
        with self.frame_ready:
            raw = self.raw.copy()
        self.calib.add(corners, (raw.shape[1], raw.shape[0]))
        self.snapshot(CALIB_IMG_DIR)
        return len(self.calib.views)

    def calib_run(self, name: str) -> dict:
        self.calib_result = self.calib.run()
        name = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "usb_cam"
        path = CAMERA_DIR / f"{name}.yaml"
        self.calib_result.camera.save(
            path,
            extra={"rms_px": round(self.calib_result.rms, 4), "pattern": list(self.calib.pattern), "square_m": self.calib.square_m},
        )
        self.update_settings({"camera_yaml": rel(path)})
        return {**self.calib_result.summary(), "saved": rel(path)}

    def state(self) -> dict:
        with self.lock:
            settings = dict(self.settings)
        models = sorted({rel(p) for d in MODEL_DIRS if d.exists() for p in d.glob("*.onnx")})
        cameras = sorted(rel(p) for p in CAMERA_DIR.glob("*.yaml")) if CAMERA_DIR.exists() else []
        return {
            **self.status,
            "error": self.error,
            "settings": settings,
            "models": models,
            "camera_yamls": cameras,
            "calib": {
                "views": len(self.calib.views),
                "min_views": MIN_VIEWS,
                "board_found": self.corners is not None,
                "result": self.calib_result.summary() if self.calib_result else None,
            },
        }


def make_handler(console: Console):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = (STATIC / "index.html").read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                self._json(console.state())
            elif self.path.startswith("/stream.mjpg"):
                self._stream()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                if self.path == "/api/settings":
                    console.update_settings(self._body())
                    self._json({"ok": True})
                elif self.path == "/api/snapshot":
                    self._json({"ok": True, "saved": console.snapshot(CAPTURE_DIR)})
                elif self.path == "/api/calib/capture":
                    self._json({"ok": True, "views": console.calib_capture()})
                elif self.path == "/api/calib/clear":
                    console.calib.clear()
                    console.calib_result = None
                    self._json({"ok": True})
                elif self.path == "/api/calib/run":
                    self._json({"ok": True, "result": console.calib_run(self._body().get("name", "usb_cam"))})
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as e:
                self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, HTTPStatus.BAD_REQUEST)

        def _stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last_id = -1
            try:
                while True:
                    with console.frame_ready:
                        console.frame_ready.wait_for(lambda: console.frame_id != last_id, timeout=2)
                        last_id, jpeg = console.frame_id, console.jpeg
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                    self.wfile.write(f"{len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--source", default=DEFAULT_SETTINGS["source"])
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    console = Console({**DEFAULT_SETTINGS, "source": args.source})
    console.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(console))
    server.daemon_threads = True

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}"
    print(f"HelloBalls console on {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        console.stop()


if __name__ == "__main__":
    main()
