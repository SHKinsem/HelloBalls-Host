"""Checkerboard intrinsic calibration for the USB camera (the web console has the same feature).

Live capture (SPACE = grab the current view when the board is detected, C = calibrate, Q = quit):
    python tools/calibrate_camera.py --camera 0 --width 1280 --height 720 --out configs/cameras/usb_cam.yaml

From a folder of images taken earlier:
    python tools/calibrate_camera.py --images calib_imgs/ --out configs/cameras/usb_cam.yaml

`--cols/--rows` count *inner* corners (a board of 10x7 squares has 9x6 inner corners).
Calibrate at the resolution you will run detection at; aim for 20+ views covering the
whole frame, including corners, at different tilts and distances.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helloballs.vision.calibration import MIN_VIEWS, CalibrationSession, find_corners  # noqa: E402
from helloballs.vision.camera import IMAGE_SUFFIXES, open_usb_camera  # noqa: E402


def collect_live(args, session: CalibrationSession) -> None:
    cap = open_usb_camera(args.camera, args.width, args.height)
    save_dir = Path(args.save_images) if args.save_images else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print("SPACE: capture   C: calibrate   Q: quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            size = (frame.shape[1], frame.shape[0])
            corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), session.pattern)

            view = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(view, session.pattern, corners, True)
            for p in session.views:  # show coverage so far
                for c in p[:: max(1, len(p) // 8)]:
                    cv2.circle(view, tuple(int(x) for x in c.ravel()), 3, (255, 128, 0), -1)
            status = f"{size[0]}x{size[1]}  views: {len(session.views)}  board: {'OK' if corners is not None else '--'}"
            cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("calibrate", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and corners is not None:
                session.add(corners, size)
                if save_dir:
                    cv2.imwrite(str(save_dir / f"calib_{len(session.views):03d}.png"), frame)
            elif key == ord("c") and len(session.views) >= MIN_VIEWS:
                break
            elif key == ord("q"):
                sys.exit(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def collect_images(folder: Path, session: CalibrationSession) -> None:
    for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        size = (gray.shape[1], gray.shape[0])
        if session.image_size is not None and size != session.image_size:
            raise SystemExit(f"{path.name}: all images must share one resolution")
        corners = find_corners(gray, session.pattern)
        print(f"{path.name}: {'ok' if corners is not None else 'board not found'}")
        if corners is not None:
            session.add(corners, size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--images", type=Path, help="calibrate from a folder instead of live capture")
    parser.add_argument("--save-images", help="also save captured frames to this folder")
    parser.add_argument("--cols", type=int, default=9, help="inner corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument("--square", type=float, default=0.025, help="square size in metres")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    session = CalibrationSession(pattern=(args.cols, args.rows), square_m=args.square)
    if args.images:
        collect_images(args.images, session)
    else:
        collect_live(args, session)

    try:
        result = session.run()
    except ValueError as e:
        raise SystemExit(str(e))

    s = result.summary()
    print(f"\nviews: {s['views']}   RMS reprojection error: {s['rms_px']} px")
    print(f"worst views: {s['worst_views']}")
    print(f"fx={s['fx']} fy={s['fy']} cx={s['cx']} cy={s['cy']}   horizontal FOV: {s['hfov_deg']} deg")
    if result.rms > 1.0:
        print("WARNING: RMS > 1 px, consider recapturing (blurry views, board not flat, wrong --square?)")

    result.camera.save(args.out, extra={"rms_px": s["rms_px"], "pattern": list(session.pattern), "square_m": session.square_m})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
