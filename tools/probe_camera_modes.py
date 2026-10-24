"""List a USB camera's video modes and grab one frame per resolution.

The robot's dual-lens module (wide + tele) selects the lens by resolution, and some
modes may deliver both lenses side by side, so looking at real frames is the only way
to tell which mode is which.

    python tools/probe_camera_modes.py                      # first non-integrated camera
    python tools/probe_camera_modes.py --name "USB Camera" --out captures/camera_modes

Writes one JPEG per resolution plus contact_sheet.jpg, and prints a table of
requested vs. actual resolution and measured frame rate.
Windows: formats come from DirectShow via pygrabber. Linux: from `v4l2-ctl`.
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helloballs.vision.camera import open_usb_camera  # noqa: E402


def list_modes_windows(name: str | None) -> tuple[int, str, list[dict]]:
    from pygrabber.dshow_graph import FilterGraph

    graph = FilterGraph()
    devices = graph.get_input_devices()
    if name is None:
        candidates = [i for i, d in enumerate(devices) if "integrated" not in d.lower()]
        if not candidates:
            raise SystemExit(f"No external camera found, devices: {devices}")
        index = candidates[0]
    else:
        if name not in devices:
            raise SystemExit(f"Camera {name!r} not found, devices: {devices}")
        index = devices.index(name)

    graph.add_video_input_device(index)
    modes = [
        {"fourcc": f["media_type_str"], "width": f["width"], "height": f["height"],
         "fps": max(f["min_framerate"], f["max_framerate"])}
        for f in graph.get_input_device().get_formats()
    ]
    return index, devices[index], modes


def list_modes_linux(device: str) -> list[dict]:
    out = subprocess.run(["v4l2-ctl", "-d", device, "--list-formats-ext"], capture_output=True, text=True).stdout
    modes, fourcc, size = [], None, None
    for line in out.splitlines():
        if m := re.search(r"'(\w{4})'", line):
            fourcc = m.group(1)
        elif m := re.search(r"Size: Discrete (\d+)x(\d+)", line):
            size = (int(m.group(1)), int(m.group(2)))
        elif (m := re.search(r"\(([\d.]+) fps\)", line)) and size:
            modes.append({"fourcc": fourcc, "width": size[0], "height": size[1], "fps": float(m.group(1))})
    return modes


def grab(device, width: int, height: int, warmup: int = 15, timed: int = 20):
    """Open at a resolution, let exposure settle, return (frame, actual size, measured fps)."""
    cap = open_usb_camera(device, width, height)
    try:
        for _ in range(warmup):
            cap.read()
        t0, frame = time.perf_counter(), None
        n = 0
        for _ in range(timed):
            ok, f = cap.read()
            if ok:
                frame, n = f, n + 1
        fps = n / (time.perf_counter() - t0) if n else 0.0
    finally:
        cap.release()
    if frame is None:
        return None, (0, 0), 0.0
    return frame, (frame.shape[1], frame.shape[0]), fps


def contact_sheet(tiles: list[tuple[str, np.ndarray]], tile_w: int = 480, cols: int = 3) -> np.ndarray:
    resized = []
    for label, img in tiles:
        h = int(img.shape[0] * tile_w / img.shape[1])
        t = cv2.resize(img, (tile_w, h))
        cv2.rectangle(t, (0, 0), (tile_w, 26), (0, 0, 0), -1)
        cv2.putText(t, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        resized.append(t)
    row_h = max(t.shape[0] for t in resized)
    rows = []
    for i in range(0, len(resized), cols):
        row = [np.pad(t, ((0, row_h - t.shape[0]), (0, 0), (0, 0))) for t in resized[i : i + cols]]
        row += [np.zeros((row_h, tile_w, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", help="DirectShow device name (Windows), default: first non-integrated camera")
    parser.add_argument("--device", default="/dev/video0", help="V4L2 device (Linux)")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "captures" / "camera_modes")
    args = parser.parse_args()

    if platform.system() == "Windows":
        device, label, modes = list_modes_windows(args.name)
    else:
        device, label, modes = args.device, args.device, list_modes_linux(args.device)

    print(f"camera: {label}\n\nreported formats:")
    for m in modes:
        print(f"  {m['fourcc']:5s} {m['width']:>5}x{m['height']:<5} {m['fps']:6.1f} fps")

    # One grab per distinct resolution; OpenCV picks the format (MJPG requested).
    sizes = sorted({(m["width"], m["height"]) for m in modes}, key=lambda s: -s[0] * s[1])
    args.out.mkdir(parents=True, exist_ok=True)
    tiles = []
    print(f"\n{'requested':>12}  {'actual':>12}  {'fps':>5}  file")
    for w, h in sizes:
        frame, actual, fps = grab(device, w, h)
        if frame is None:
            print(f"{w:>5}x{h:<6}  {'(no frame)':>12}")
            continue
        path = args.out / f"{w}x{h}.jpg"
        cv2.imwrite(str(path), frame)
        mismatch = "" if actual == (w, h) else "  <- camera gave a different size"
        print(f"{w:>5}x{h:<6}  {actual[0]:>5}x{actual[1]:<6}  {fps:5.1f}  {path.name}{mismatch}")
        tiles.append((f"{w}x{h} -> {actual[0]}x{actual[1]}", frame))

    if tiles:
        sheet = args.out / "contact_sheet.jpg"
        cv2.imwrite(str(sheet), contact_sheet(tiles))
        print(f"\ncontact sheet: {sheet}")


if __name__ == "__main__":
    main()
