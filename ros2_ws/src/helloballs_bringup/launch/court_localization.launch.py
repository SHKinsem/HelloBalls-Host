from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    vio_odom_topic = LaunchConfiguration("vio_odom_topic")
    world_frame = LaunchConfiguration("world_frame")
    base_frame = LaunchConfiguration("base_frame")
    court_frame = LaunchConfiguration("court_frame")
    publish_identity_world_to_court_tf = LaunchConfiguration("publish_identity_world_to_court_tf")
    start_corner = LaunchConfiguration("start_corner")
    camera_height_m = LaunchConfiguration("camera_height_m")
    camera_pitch_rad = LaunchConfiguration("camera_pitch_rad")
    court_length_m = LaunchConfiguration("court_length_m")
    court_width_m = LaunchConfiguration("court_width_m")
    singles_width_m = LaunchConfiguration("singles_width_m")
    service_line_distance_from_net_m = LaunchConfiguration("service_line_distance_from_net_m")

    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/camera/image_mono"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
            DeclareLaunchArgument("vio_odom_topic", default_value="/vio/odometry"),
            DeclareLaunchArgument("world_frame", default_value="world"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("court_frame", default_value="court"),
            DeclareLaunchArgument("publish_identity_world_to_court_tf", default_value="true"),
            DeclareLaunchArgument("start_corner", default_value="unknown"),
            DeclareLaunchArgument("camera_height_m", default_value="0.14214"),
            DeclareLaunchArgument("camera_pitch_rad", default_value="0.0"),
            DeclareLaunchArgument("court_length_m", default_value="23.77"),
            DeclareLaunchArgument("court_width_m", default_value="10.97"),
            DeclareLaunchArgument("singles_width_m", default_value="8.23"),
            DeclareLaunchArgument("service_line_distance_from_net_m", default_value="6.40"),
            Node(
                package="helloballs_bringup",
                executable="court_line_localizer_node",
                name="court_line_localizer",
                output="screen",
                parameters=[
                    {
                        "image_topic": image_topic,
                        "camera_info_topic": camera_info_topic,
                        "vio_odom_topic": vio_odom_topic,
                        "base_frame": base_frame,
                        "court_frame": court_frame,
                        "start_corner": start_corner,
                        "camera_height_m": ParameterValue(camera_height_m, value_type=float),
                        "camera_pitch_rad": ParameterValue(camera_pitch_rad, value_type=float),
                        "court_length_m": ParameterValue(court_length_m, value_type=float),
                        "court_width_m": ParameterValue(court_width_m, value_type=float),
                        "singles_width_m": ParameterValue(singles_width_m, value_type=float),
                        "service_line_distance_from_net_m": ParameterValue(
                            service_line_distance_from_net_m,
                            value_type=float,
                        ),
                    }
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="world_to_court_tf",
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--roll",
                    "0",
                    "--pitch",
                    "0",
                    "--yaw",
                    "0",
                    "--frame-id",
                    world_frame,
                    "--child-frame-id",
                    court_frame,
                ],
                condition=IfCondition(publish_identity_world_to_court_tf),
                output="screen",
            ),
        ]
    )
