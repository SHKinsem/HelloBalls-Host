from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module

from scripts.HelloBalls_Serial import ImuState
from scripts.opencv_camera import CameraFrame

G_TO_MPS2 = 9.80665
DEG_TO_RAD = math.pi / 180.0


@dataclass(frozen=True)
class RosPublisherConfig:
    node_name: str = "helloballs_sensor_publisher"
    camera_topic: str = "/camera/image_raw"
    camera_info_topic: str = "/camera/camera_info"
    imu_topic: str = "/imu/data_raw"
    camera_frame_id: str = "camera_link"
    imu_frame_id: str = "imu_link"


class RosSensorPublisher:
    def __init__(self, config: RosPublisherConfig | None = None) -> None:
        self.config = config or RosPublisherConfig()
        self.rclpy = import_module("rclpy")
        self.camera_info_msg = import_module("sensor_msgs.msg").CameraInfo
        self.image_msg = import_module("sensor_msgs.msg").Image
        self.imu_msg = import_module("sensor_msgs.msg").Imu

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
        stamp = self.node.get_clock().now().to_msg()

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
        camera_info_msg.height = frame.height
        camera_info_msg.width = frame.width
        camera_info_msg.distortion_model = "plumb_bob"
        self.camera_info_pub.publish(camera_info_msg)

    def publish_imu(self, state: ImuState) -> None:
        stamp = self.node.get_clock().now().to_msg()
        acc_x, acc_y, acc_z = state.acc_g
        gyro_x, gyro_y, gyro_z = state.gyro_dps

        imu_msg = self.imu_msg()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = self.config.imu_frame_id
        imu_msg.orientation_covariance[0] = -1.0
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


def _image_step(image, width: int) -> int:
    if len(image.shape) == 2:
        return width
    return width * image.shape[2]
