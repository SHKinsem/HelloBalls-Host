from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    camera_device = LaunchConfiguration("camera_device")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_fps = LaunchConfiguration("camera_fps")
    camera_fourcc = LaunchConfiguration("camera_fourcc")
    camera_frame_id = LaunchConfiguration("camera_frame_id")
    camera_buffer_size = LaunchConfiguration("camera_buffer_size")
    use_v4l2_ctl = LaunchConfiguration("use_v4l2_ctl")
    camera_info_rate_hz = LaunchConfiguration("camera_info_rate_hz")
    start_mono_converter = LaunchConfiguration("start_mono_converter")
    mono_image_topic = LaunchConfiguration("mono_image_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_device", default_value="/dev/video0"),
            DeclareLaunchArgument("camera_width", default_value="800"),
            DeclareLaunchArgument("camera_height", default_value="592"),
            DeclareLaunchArgument("camera_fps", default_value="15.0"),
            DeclareLaunchArgument("camera_fourcc", default_value="MJPG"),
            DeclareLaunchArgument("camera_frame_id", default_value="camera_optical_frame"),
            DeclareLaunchArgument("camera_buffer_size", default_value="1"),
            DeclareLaunchArgument("use_v4l2_ctl", default_value="false"),
            DeclareLaunchArgument("camera_info_rate_hz", default_value="1.0"),
            DeclareLaunchArgument("start_mono_converter", default_value="false"),
            DeclareLaunchArgument("mono_image_topic", default_value="/camera/image_mono"),
            Node(
                package="helloballs_bringup",
                executable="camera_publisher_node",
                name="helloballs_camera_publisher",
                output="screen",
                parameters=[
                    {
                        "camera_device": camera_device,
                        "camera_width": camera_width,
                        "camera_height": camera_height,
                        "camera_fps": camera_fps,
                        "camera_fourcc": camera_fourcc,
                        "camera_frame_id": camera_frame_id,
                        "camera_buffer_size": ParameterValue(camera_buffer_size, value_type=int),
                        "use_v4l2_ctl": ParameterValue(use_v4l2_ctl, value_type=bool),
                        "image_topic": "/camera/image_raw",
                        "camera_info_topic": "/camera/camera_info",
                        "camera_info_rate_hz": camera_info_rate_hz,
                    }
                ],
            ),
            Node(
                package="helloballs_bringup",
                executable="grayscale_converter_node",
                name="helloballs_grayscale_converter",
                output="screen",
                condition=IfCondition(start_mono_converter),
                parameters=[
                    {
                        "input_topic": "/camera/image_raw",
                        "output_topic": mono_image_topic,
                        "frame_id": camera_frame_id,
                    }
                ],
            ),
        ]
    )
