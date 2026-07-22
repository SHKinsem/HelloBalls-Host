import argparse
from pathlib import Path
import time

from scripts import HelloBalls_Serial
from scripts.keyboard_control import KeyboardController


def default_vins_config_file(ros_workspace: str) -> str:
    workspace = Path(ros_workspace)
    workspace_local = Path("src/helloballs_bringup/config/vins_mono_imu.yaml")
    repo_local = Path("ros2_ws/src/helloballs_bringup/config/vins_mono_imu.yaml")
    if (workspace / workspace_local).exists():
        return str(workspace_local)
    if (workspace / repo_local).exists():
        return str(repo_local)
    return "vins_mono_imu.yaml"


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
    parser.add_argument(
        "--ros-localization",
        action="store_true",
        help="Start IMU, camera, mono image conversion, and VINS odometry.",
    )
    parser.add_argument("--ros-node-name", default="helloballs_sensor_publisher")
    parser.add_argument("--imu-topic", default="/imu/data_raw")
    parser.add_argument("--imu-frame-id", default="imu_link")
    parser.add_argument(
        "--allow-legacy-imu-timestamps",
        action="store_true",
        help=(
            "Temporarily publish legacy IMU v1 frames using UART receive time. "
            "This produces incorrect bursty timing and is unsuitable for VINS."
        ),
    )
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
    parser.add_argument(
        "--imu-accel-scale",
        type=float,
        default=0.907942,
        help=(
            "Uniform accelerometer scale applied after conversion to m/s^2. "
            "The default corrects the measured stationary norm 10.80097 to 9.80665."
        ),
    )
    parser.add_argument("--keyboard-control", action="store_true", help="Enable keyboard driving over the MCU UART.")
    parser.add_argument("--keyboard-speed", type=int, default=100, help="Motor speed used by keyboard control.")
    parser.add_argument("--keyboard-state", type=int, default=1, help="MCU state value used for keyboard motor commands.")
    parser.add_argument(
        "--keyboard-timeout",
        type=float,
        default=0.15,
        help="Stop unless another movement key arrives within this many seconds.",
    )
    parser.add_argument(
        "--ros-camera",
        action="store_true",
        help="Start helloballs_bringup camera.launch.py alongside this host process.",
    )
    parser.add_argument("--ros-workspace", default="ros2_ws", help="ROS2 workspace containing install/setup.bash.")
    parser.add_argument("--camera-device", default="/dev/video0")
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=592)
    parser.add_argument("--camera-fps", type=float, default=15.0)
    parser.add_argument("--camera-fourcc", default="MJPG")
    parser.add_argument("--camera-frame-id", default="camera_optical_frame")
    parser.add_argument(
        "--camera-mono-converter",
        action="store_true",
        help="Start a backend ROS2 node that converts /camera/image_raw to mono8.",
    )
    parser.add_argument("--camera-mono-topic", default="/camera/image_mono")
    parser.add_argument("--camera-buffer-size", type=int, default=1)
    parser.add_argument("--no-v4l2-ctl", action="store_true", help="Do not let camera launch run v4l2-ctl.")
    parser.add_argument("--camera-info-rate-hz", type=float, default=1.0)
    parser.add_argument(
        "--ros-vins",
        action="store_true",
        help="Start helloballs_bringup vins.launch.py alongside this host process.",
    )
    parser.add_argument(
        "--vins-config-file",
        default=None,
        help="VINS config file. Defaults to the repository's vins_mono_imu.yaml.",
    )
    parser.add_argument("--vins-image-topic", default=None)
    parser.add_argument("--vins-imu-topic", default=None)
    parser.add_argument("--vins-package", default="vins")
    parser.add_argument("--vins-executable", default="vins_node")
    args = parser.parse_args()

    if args.ros_localization:
        args.ros_publish = True
        args.ros_camera = True
        args.camera_mono_converter = True
        args.ros_vins = True
    if args.ros_sensors:
        args.ros_publish = True
        args.ros_camera = True

    if args.vins_image_topic is None:
        args.vins_image_topic = args.camera_mono_topic if args.camera_mono_converter else "/camera/image_raw"
    if args.vins_imu_topic is None:
        args.vins_imu_topic = args.imu_topic
    if args.vins_config_file is None:
        args.vins_config_file = default_vins_config_file(args.ros_workspace)

    if args.no_serial and not (args.ros_camera or args.ros_vins):
        parser.error("Nothing to run: remove --no-serial.")
    if args.keyboard_control and args.no_serial:
        parser.error("--keyboard-control requires serial; remove --no-serial.")

    receiver = None
    ros_publisher = None
    keyboard_controller = None
    managed_launches = []

    try:
        from scripts.ros_launch import RosLaunchConfig, RosLaunchProcess

        if args.ros_camera:
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
                        "camera_buffer_size": str(args.camera_buffer_size),
                        "use_v4l2_ctl": str(not args.no_v4l2_ctl).lower(),
                        "camera_info_rate_hz": str(args.camera_info_rate_hz),
                        "start_mono_converter": str(args.camera_mono_converter).lower(),
                        "mono_image_topic": args.camera_mono_topic,
                    },
                )
            )
            camera_launch.start()
            managed_launches.append(("camera bringup", camera_launch))
            print("ROS2 camera bringup started: helloballs_bringup camera.launch.py.")

        if args.ros_vins:
            vins_launch = RosLaunchProcess(
                RosLaunchConfig(
                    workspace=Path(args.ros_workspace),
                    package="helloballs_bringup",
                    launch_file="vins.launch.py",
                    launch_arguments={
                        "config_file": args.vins_config_file,
                        "image_topic": args.vins_image_topic,
                        "imu_topic": args.vins_imu_topic,
                        "vins_package": args.vins_package,
                        "estimator_executable": args.vins_executable,
                    },
                )
            )
            vins_launch.start()
            managed_launches.append(("VINS odometry", vins_launch))
            print(
                "ROS2 VINS odometry started: "
                f"image={args.vins_image_topic}, imu={args.vins_imu_topic}."
            )

        if args.ros_publish:
            from scripts.ros_publishers import RosPublisherConfig, RosSensorPublisher

            ros_publisher = RosSensorPublisher(
                RosPublisherConfig(
                    node_name=args.ros_node_name,
                    imu_topic=args.imu_topic,
                    imu_frame_id=args.imu_frame_id,
                    imu_accel_scale=args.imu_accel_scale,
                    imu_angular_velocity_variance=args.imu_angular_velocity_variance,
                    imu_linear_acceleration_variance=args.imu_linear_acceleration_variance,
                    allow_legacy_imu_timestamps=args.allow_legacy_imu_timestamps,
                )
            )
            imu_timestamp_mode = (
                "legacy UART receive timestamps allowed (diagnostics only)"
                if args.allow_legacy_imu_timestamps
                else "MCU-timestamped IMU v2 required"
            )
            print(f"ROS2 IMU publisher started: {args.imu_topic} ({imu_timestamp_mode}).")

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
                    key_timeout=args.keyboard_timeout,
                )
                keyboard_controller.start()

        print("HelloBalls host running. Press Ctrl+C to stop.")
        while True:
            for launch_name, launch_process in managed_launches:
                returncode = launch_process.poll()
                if returncode is not None:
                    raise RuntimeError(f"ROS2 {launch_name} exited with code {returncode}.")

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
        for _, launch_process in reversed(managed_launches):
            launch_process.close()


if __name__ == "__main__":
    main()
