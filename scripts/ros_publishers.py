from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module

from scripts.HelloBalls_Serial import ImuState

G_TO_MPS2 = 9.80665
DEG_TO_RAD = math.pi / 180.0


@dataclass(frozen=True)
class RosPublisherConfig:
    node_name: str = "helloballs_sensor_publisher"
    imu_topic: str = "/imu/data_raw"
    imu_frame_id: str = "imu_link"
    imu_angular_velocity_variance: float = 0.0
    imu_linear_acceleration_variance: float = 0.0


class RosSensorPublisher:
    def __init__(self, config: RosPublisherConfig | None = None) -> None:
        self.config = config or RosPublisherConfig()
        self.rclpy = import_module("rclpy")
        self.imu_msg = import_module("sensor_msgs.msg").Imu
        self.time_msg = import_module("builtin_interfaces.msg").Time
        qos = import_module("rclpy.qos")

        self.rclpy.init(args=None)
        self.node = self.rclpy.create_node(self.config.node_name)
        sensor_qos = qos.qos_profile_sensor_data
        self.imu_pub = self.node.create_publisher(self.imu_msg, self.config.imu_topic, sensor_qos)

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


def _remap_imu_raw_to_base(values: tuple[float, float, float]) -> tuple[float, float, float]:
    raw_x, raw_y, raw_z = values
    return raw_x, -raw_z, raw_y


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
