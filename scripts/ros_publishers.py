from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from scripts.HelloBalls_Serial import ImuState
from scripts.opencv_camera import CameraFrame

G_TO_MPS2 = 9.80665
DEG_TO_RAD = math.pi / 180.0


@dataclass(frozen=True)
class CameraInfoCalibration:
    width: int | None = None
    height: int | None = None
    distortion_model: str = "plumb_bob"
    d: tuple[float, ...] = ()
    k: tuple[float, ...] = (0.0,) * 9
    r: tuple[float, ...] = (0.0,) * 9
    p: tuple[float, ...] = (0.0,) * 12


@dataclass(frozen=True)
class RosPublisherConfig:
    node_name: str = "helloballs_sensor_publisher"
    camera_topic: str = "/camera/image_raw"
    camera_info_topic: str = "/camera/camera_info"
    imu_topic: str = "/imu/data_raw"
    camera_frame_id: str = "camera_link"
    imu_frame_id: str = "imu_link"
    camera_info_yaml: str | None = None
    imu_angular_velocity_variance: float = 0.0
    imu_linear_acceleration_variance: float = 0.0


class RosSensorPublisher:
    def __init__(self, config: RosPublisherConfig | None = None) -> None:
        self.config = config or RosPublisherConfig()
        self.rclpy = import_module("rclpy")
        self.camera_info_msg = import_module("sensor_msgs.msg").CameraInfo
        self.image_msg = import_module("sensor_msgs.msg").Image
        self.imu_msg = import_module("sensor_msgs.msg").Imu
        self.time_msg = import_module("builtin_interfaces.msg").Time

        self.camera_calibration = (
            load_camera_info_calibration(self.config.camera_info_yaml)
            if self.config.camera_info_yaml
            else CameraInfoCalibration()
        )

        self.rclpy.init(args=None)
        self.node = self.rclpy.create_node(self.config.node_name)
        self.camera_pub = self.node.create_publisher(self.image_msg, self.config.camera_topic, 10)
        self.camera_info_pub = self.node.create_publisher(
            self.camera_info_msg,
            self.config.camera_info_topic,
            10,
        )
        self.imu_pub = self.node.create_publisher(self.imu_msg, self.config.imu_topic, 50)

    def publish_camera(self, frame: CameraFrame) -> None:
        stamp = _stamp_from_unix_time(frame.captured_at, self.time_msg)

        image_msg = self.image_msg()
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.config.camera_frame_id
        image_msg.height = frame.height
        image_msg.width = frame.width
        image_msg.encoding = _image_encoding(frame.image)
        image_msg.is_bigendian = 0
        image_msg.step = _image_step(frame.image, frame.width)
        image_msg.data = frame.image.tobytes()
        self.camera_pub.publish(image_msg)

        camera_info_msg = self.camera_info_msg()
        camera_info_msg.header.stamp = stamp
        camera_info_msg.header.frame_id = self.config.camera_frame_id
        camera_info_msg.height = self.camera_calibration.height or frame.height
        camera_info_msg.width = self.camera_calibration.width or frame.width
        camera_info_msg.distortion_model = self.camera_calibration.distortion_model
        camera_info_msg.d = list(self.camera_calibration.d)
        camera_info_msg.k = list(self.camera_calibration.k)
        camera_info_msg.r = list(self.camera_calibration.r)
        camera_info_msg.p = list(self.camera_calibration.p)
        self.camera_info_pub.publish(camera_info_msg)

    def publish_imu(self, state: ImuState) -> None:
        stamp = _stamp_from_unix_time(state.received_at, self.time_msg)
        acc_x, acc_y, acc_z = _remap_imu_raw_to_base(state.acc_g)
        gyro_x, gyro_y, gyro_z = _remap_imu_raw_to_base(state.gyro_dps)

        imu_msg = self.imu_msg()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = self.config.imu_frame_id
        imu_msg.orientation_covariance[0] = -1.0
        _fill_diagonal_covariance(
            imu_msg.angular_velocity_covariance,
            self.config.imu_angular_velocity_variance,
        )
        _fill_diagonal_covariance(
            imu_msg.linear_acceleration_covariance,
            self.config.imu_linear_acceleration_variance,
        )
        imu_msg.linear_acceleration.x = acc_x * G_TO_MPS2
        imu_msg.linear_acceleration.y = acc_y * G_TO_MPS2
        imu_msg.linear_acceleration.z = acc_z * G_TO_MPS2
        imu_msg.angular_velocity.x = gyro_x * DEG_TO_RAD
        imu_msg.angular_velocity.y = gyro_y * DEG_TO_RAD
        imu_msg.angular_velocity.z = gyro_z * DEG_TO_RAD
        self.imu_pub.publish(imu_msg)

    def spin_once(self) -> None:
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()


def _image_encoding(image) -> str:
    if len(image.shape) == 2:
        return "mono8"
    channels = image.shape[2]
    if channels == 3:
        return "bgr8"
    if channels == 4:
        return "bgra8"
    raise RuntimeError(f"Unsupported camera image shape: {image.shape}")


def _remap_imu_raw_to_base(values: tuple[float, float, float]) -> tuple[float, float, float]:
    raw_x, raw_y, raw_z = values
    return raw_x, -raw_z, raw_y


def _image_step(image, width: int) -> int:
    if len(image.shape) == 2:
        return width
    return width * image.shape[2]


def load_camera_info_calibration(path: str) -> CameraInfoCalibration:
    yaml = import_module("yaml")
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise RuntimeError(f"Camera info YAML must contain a mapping: {yaml_path}")

    return CameraInfoCalibration(
        width=_optional_int(data, "image_width", "width"),
        height=_optional_int(data, "image_height", "height"),
        distortion_model=str(data.get("distortion_model", "plumb_bob")),
        d=_matrix_data(data, "distortion_coefficients", "d", "D"),
        k=_matrix_data(data, "camera_matrix", "k", "K", expected_len=9),
        r=_matrix_data(data, "rectification_matrix", "r", "R", expected_len=9),
        p=_matrix_data(data, "projection_matrix", "p", "P", expected_len=12),
    )


def _stamp_from_unix_time(timestamp: float, time_msg):
    if timestamp <= 0.0:
        timestamp = 0.0
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1_000_000_000)
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    stamp = time_msg()
    stamp.sec = sec
    stamp.nanosec = nanosec
    return stamp


def _fill_diagonal_covariance(covariance, variance: float) -> None:
    covariance[0] = variance
    covariance[4] = variance
    covariance[8] = variance


def _optional_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in data and data[key] is not None:
            return int(data[key])
    return None


def _matrix_data(
    data: dict[str, Any],
    *keys: str,
    expected_len: int | None = None,
) -> tuple[float, ...]:
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, dict) and "data" in value:
            value = value["data"]
        if value is None:
            continue
        values = tuple(float(item) for item in value)
        if expected_len is not None and len(values) != expected_len:
            raise RuntimeError(
                f"Camera info field {key!r} must contain {expected_len} values, got {len(values)}."
            )
        return values

    if expected_len is None:
        return ()
    return (0.0,) * expected_len
