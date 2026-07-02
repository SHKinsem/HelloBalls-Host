import argparse
from pathlib import Path
import time

from scripts import HelloBalls_Serial
from scripts.keyboard_control import KeyboardController


def main():
    parser = argparse.ArgumentParser(description="HelloBalls RDK host entry point.")
    parser.add_argument("--no-serial", action="store_true", help="Run without opening the MCU UART.")
    parser.add_argument("--serial-port", default="/dev/ttyS1", help="UART device path for the MCU.")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--print-imu", action="store_true", help="Print received IMU frames.")
    parser.add_argument("--ros-publish", action="store_true", help="Publish IMU data as a ROS2 topic.")
    parser.add_argument(
        "--ros-sensors",
        action="store_true",
        help="Publish IMU data and start the ROS2 camera bringup launch.",
    )
    parser.add_argument("--ros-node-name", default="helloballs_sensor_publisher")
    parser.add_argument("--imu-topic", default="/imu/data_raw")
    parser.add_argument("--imu-frame-id", default="imu_link")
    parser.add_argument(
        "--imu-angular-velocity-variance",
        type=float,
        default=0.0,
        help="Diagonal angular velocity covariance variance in (rad/s)^2.",
    )
    parser.add_argument(
        "--imu-linear-acceleration-variance",
        type=float,
        default=0.0,
        help="Diagonal linear acceleration covariance variance in (m/s^2)^2.",
    )
    parser.add_argument("--keyboard-control", action="store_true", help="Enable keyboard driving over the MCU UART.")
    parser.add_argument("--keyboard-speed", type=int, default=100, help="Motor speed used by keyboard control.")
    parser.add_argument("--keyboard-state", type=int, default=1, help="MCU state value used for keyboard motor commands.")
    parser.add_argument(
        "--ros-camera",
        action="store_true",
        help="Start helloballs_bringup camera.launch.py alongside this host process.",
    )
    parser.add_argument("--ros-workspace", default="ros2_ws", help="ROS2 workspace containing install/setup.bash.")
    parser.add_argument("--camera-device", default="/dev/video0")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--camera-frame-id", default="camera_link")
    parser.add_argument("--camera-grayscale", action="store_true")
    parser.add_argument("--camera-buffer-size", type=int, default=4)
    parser.add_argument("--no-v4l2-ctl", action="store_true", help="Do not let camera launch run v4l2-ctl.")
    parser.add_argument("--camera-info-rate-hz", type=float, default=1.0)
    args = parser.parse_args()

    if args.ros_sensors:
        args.ros_publish = True
        args.ros_camera = True

    if args.no_serial and not args.ros_camera:
        parser.error("Nothing to run: remove --no-serial.")
    if args.keyboard_control and args.no_serial:
        parser.error("--keyboard-control requires serial; remove --no-serial.")

    receiver = None
    ros_publisher = None
    keyboard_controller = None
    camera_launch = None

    try:
        if args.ros_camera:
            from scripts.ros_launch import RosLaunchConfig, RosLaunchProcess

            camera_launch = RosLaunchProcess(
                RosLaunchConfig(
                    workspace=Path(args.ros_workspace),
                    package="helloballs_bringup",
                    launch_file="camera.launch.py",
                    launch_arguments={
                        "camera_device": args.camera_device,
                        "camera_width": str(args.camera_width),
                        "camera_height": str(args.camera_height),
                        "camera_fps": str(args.camera_fps),
                        "camera_fourcc": args.camera_fourcc,
                        "camera_frame_id": args.camera_frame_id,
                        "camera_grayscale": str(args.camera_grayscale).lower(),
                        "camera_buffer_size": str(args.camera_buffer_size),
                        "use_v4l2_ctl": str(not args.no_v4l2_ctl).lower(),
                        "camera_info_rate_hz": str(args.camera_info_rate_hz),
                    },
                )
            )
            camera_launch.start()
            print("ROS2 camera bringup started: helloballs_bringup camera.launch.py.")

        if args.ros_publish:
            from scripts.ros_publishers import RosPublisherConfig, RosSensorPublisher

            ros_publisher = RosSensorPublisher(
                RosPublisherConfig(
                    node_name=args.ros_node_name,
                    imu_topic=args.imu_topic,
                    imu_frame_id=args.imu_frame_id,
                    imu_angular_velocity_variance=args.imu_angular_velocity_variance,
                    imu_linear_acceleration_variance=args.imu_linear_acceleration_variance,
                )
            )
            print(f"ROS2 IMU publisher started: {args.imu_topic}.")

        if not args.no_serial:
            imu_callback = ros_publisher.publish_imu if ros_publisher is not None else None
            receiver = HelloBalls_Serial.SerialReceiver(
                port=args.serial_port,
                baudrate=args.baudrate,
                state_callback=imu_callback,
            )
            receiver.start()
            print(f"Serial receiver started on {args.serial_port}.")

            if args.keyboard_control:
                keyboard_controller = KeyboardController(
                    serial_receiver=receiver,
                    speed=args.keyboard_speed,
                    state=args.keyboard_state,
                )
                keyboard_controller.start()

        print("HelloBalls host running. Press Ctrl+C to stop.")
        while True:
            if camera_launch is not None:
                camera_returncode = camera_launch.poll()
                if camera_returncode is not None:
                    raise RuntimeError(f"ROS2 camera bringup exited with code {camera_returncode}.")

            if receiver is not None:
                state = receiver.get_latest_state()
            else:
                state = None
            if args.print_imu and state is not None:
                print(HelloBalls_Serial.format_state(state))

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping HelloBalls host.")
    except RuntimeError as error:
        print(f"\nStopping HelloBalls host: {error}")
        raise SystemExit(1) from error
    finally:
        if keyboard_controller is not None:
            keyboard_controller.close()
        if receiver is not None:
            receiver.close()
        if ros_publisher is not None:
            ros_publisher.close()
        if camera_launch is not None:
            camera_launch.close()


if __name__ == "__main__":
    main()
