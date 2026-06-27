import argparse
import time

from scripts import HelloBalls_Serial
from scripts.opencv_camera import CameraConfig, OpenCVCamera, get_cv2


def main():
    parser = argparse.ArgumentParser(description="HelloBalls RDK host entry point.")
    parser.add_argument("--no-serial", action="store_true", help="Run without opening the MCU UART.")
    parser.add_argument("--serial-port", default="/dev/ttyS1", help="UART device path for the MCU.")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--print-imu", action="store_true", help="Print received IMU frames.")
    parser.add_argument("--enable-camera", action="store_true", help="Read frames from an OpenCV camera.")
    parser.add_argument("--camera-device", default="/dev/video0", help="V4L2 camera device path.")
    parser.add_argument("--camera-width", type=int, default=2560)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-buffer-size", type=int, default=1)
    parser.add_argument("--camera-fourcc", default="MJPG", help="Request a pixel format such as MJPG or YUYV.")
    parser.add_argument("--camera-fps", type=float, default=None, help="Request camera FPS.")
    parser.add_argument("--camera-preview", action="store_true", help="Show an OpenCV preview window.")
    parser.add_argument("--print-camera-fps", action="store_true", help="Print camera read FPS once per second.")
    args = parser.parse_args()

    if args.no_serial and not args.enable_camera:
        parser.error("Nothing to run: remove --no-serial or add --enable-camera.")

    receiver = None
    camera = None
    preview_cv2 = None
    frame_count = 0
    fps_started_at = time.time()

    if not args.no_serial:
        receiver = HelloBalls_Serial.SerialReceiver(port=args.serial_port, baudrate=args.baudrate)
        receiver.start()
        print(f"Serial receiver started on {args.serial_port}.")

    if args.enable_camera:
        camera = OpenCVCamera(
            CameraConfig(
                camera_device=args.camera_device,
                width=args.camera_width,
                height=args.camera_height,
                buffer_size=args.camera_buffer_size,
                fourcc=args.camera_fourcc,
                fps=args.camera_fps,
            )
        )
        camera_id = camera.open()
        actual_width, actual_height = camera.actual_resolution()
        print(f"OpenCV camera {camera_id} opened at {actual_width}x{actual_height}.")

        if args.camera_preview:
            preview_cv2 = get_cv2()

    print("HelloBalls host running. Press Ctrl+C to stop.")

    try:
        while True:
            if receiver is not None:
                state = receiver.get_latest_state()
            else:
                state = None
            if args.print_imu and state is not None:
                print(HelloBalls_Serial.format_state(state))

            if camera is not None:
                frame = camera.read()
                if frame is None:
                    print("Warning: failed to read camera frame.")
                    time.sleep(0.05)
                    continue

                frame_count += 1
                now = time.time()
                if args.print_camera_fps and now - fps_started_at >= 1.0:
                    fps = frame_count / (now - fps_started_at)
                    print(f"Camera FPS: {fps:.1f} ({frame.width}x{frame.height})")
                    frame_count = 0
                    fps_started_at = now

                if args.camera_preview:
                    preview_cv2.imshow("HelloBalls Camera", frame.image)
                    if preview_cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if camera is None:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping HelloBalls host.")
    finally:
        if camera is not None:
            camera.close()
        if preview_cv2 is not None:
            preview_cv2.destroyAllWindows()
        if receiver is not None:
            receiver.close()


if __name__ == "__main__":
    main()
