from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_to_camera_tf",
                arguments=[
                    "0.23668",
                    "0.0",
                    "0.14214",
                    "0.0",
                    "0.0",
                    "0.0",
                    "base_link",
                    "camera_link",
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_to_imu_tf",
                arguments=[
                    "0.07326",
                    "-0.063",
                    "0.10659",
                    "0.0",
                    "0.0",
                    "0.0",
                    "base_link",
                    "imu_link",
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="camera_to_optical_tf",
                arguments=[
                    "0.0",
                    "0.0",
                    "0.0",
                    "-1.5707963267948966",
                    "0.0",
                    "-1.5707963267948966",
                    "camera_link",
                    "camera_optical_frame",
                ],
            ),
        ]
    )
