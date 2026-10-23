"""Live tennis ball detection with distance / angle overlay.

    python tools/detect_demo.py --source 0                                    # USB camera
    python tools/detect_demo.py --source clip.mp4 --camera-yaml configs/cameras/usb_cam.yaml
    python tools/detect_demo.py --source 0 --mount-height 0.25 --mount-pitch 20

Without --camera-yaml a rough pinhole model from --hfov is used and distances are only
indicative ("UNCALIBRATED" is shown). Q quits, SPACE pauses.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helloballs.vision.camera import frames  # noqa: E402
from helloballs.vision.detector import OnnxDetector  # noqa: E402
from helloballs.vision.geometry import CameraModel, Mount, locate_ball  # noqa: E402
from helloballs.vision.overlay import draw_detections, draw_hud  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "scripts" / "yolov11_roboflow_ir9.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="camera index, /dev/videoN, video or image path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-yaml", type=Path)
    parser.add_argument("--hfov", type=float, default=70.0, help="assumed horizontal FOV when uncalibrated")
    parser.add_argument("--mount-height", type=float, help="camera height above floor in metres")
    parser.add_argument("--mount-pitch", type=float, default=0.0, help="downward tilt in degrees")
    args = parser.parse_args()

    detector = OnnxDetector(args.model, conf_threshold=args.conf)
    print(f"model classes: {detector.class_names}  ball ids: {detector.ball_class_ids}")
    base_camera = CameraModel.load(args.camera_yaml) if args.camera_yaml else None
    mount = Mount(args.mount_height, args.mount_pitch) if args.mount_height else None
    is_image = Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

    camera = None
    fps = 0.0
    for frame in frames(args.source, args.width, args.height):
        h, w = frame.shape[:2]
        if camera is None or (camera.width, camera.height) != (w, h):
            camera = base_camera.scaled_to(w, h) if base_camera else CameraModel.from_fov(w, h, args.hfov)

        t0 = time.perf_counter()
        detections = detector(frame)
        infer_ms = (time.perf_counter() - t0) * 1000
        fps = 0.9 * fps + 0.1 * (1000 / max(infer_ms, 1e-3)) if fps else 1000 / max(infer_ms, 1e-3)

        positions = [locate_ball(camera, (d.x1, d.y1, d.x2, d.y2), mount) for d in detections]
        draw_detections(frame, detections, positions, show_ranges=mount is not None)
        hud = f"{w}x{h}  infer {infer_ms:.0f}ms  {fps:.1f} FPS  balls {len(detections)}"
        draw_hud(frame, hud, None if camera.calibrated else "UNCALIBRATED")

        cv2.imshow("HelloBalls detect", frame)
        key = cv2.waitKey(0 if is_image else 1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
