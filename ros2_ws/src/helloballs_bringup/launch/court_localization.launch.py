from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _world_to_court_tf(context):
    """Place VIO's initial world origin at the configured court start pose.

    VINS starts its local `world` frame at the vehicle's initial pose.  The
    transform below therefore makes that local origin a child of `court`, so
    RViz can draw `/vio/odometry` on the court map without changing the VIO
    estimator itself.
    """
    enabled = LaunchConfiguration("publish_world_to_court_tf").perform(context)
    legacy_enabled = LaunchConfiguration("publish_identity_world_to_court_tf").perform(context)
    if (
        enabled.lower() not in ("1", "true", "yes", "on") or
        legacy_enabled.lower() not in ("1", "true", "yes", "on")
    ):
        return []

    start_corner = LaunchConfiguration("start_corner").perform(context)
    court_length = float(LaunchConfiguration("court_length_m").perform(context))
    court_width = float(LaunchConfiguration("court_width_m").perform(context))
    start_inset = float(LaunchConfiguration("start_inset_m").perform(context))

    half_length = court_length * 0.5
    half_width = court_width * 0.5
    poses = {
        "near_left": (-half_length + start_inset, half_width - start_inset, 0.0),
        "near_right": (-half_length + start_inset, -half_width + start_inset, 0.0),
        "far_left": (half_length - start_inset, half_width - start_inset, 3.141592653589793),
        "far_right": (half_length - start_inset, -half_width + start_inset, 3.141592653589793),
    }
    if start_corner == "unknown":
        x, y, yaw = 0.0, 0.0, 0.0
    elif start_corner in poses:
        x, y, yaw = poses[start_corner]
    else:
        raise RuntimeError(
            "start_corner must be unknown, near_left, near_right, far_left, or far_right"
        )

    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="court_to_vio_world_tf",
            arguments=[
                "--x", str(x),
                "--y", str(y),
                "--z", "0",
                "--roll", "0",
                "--pitch", "0",
                "--yaw", str(yaw),
                "--frame-id", LaunchConfiguration("court_frame").perform(context),
                "--child-frame-id", LaunchConfiguration("world_frame").perform(context),
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    vio_odom_topic = LaunchConfiguration("vio_odom_topic")
    world_frame = LaunchConfiguration("world_frame")
    base_frame = LaunchConfiguration("base_frame")
    court_frame = LaunchConfiguration("court_frame")
    start_corner = LaunchConfiguration("start_corner")
    start_inset_m = LaunchConfiguration("start_inset_m")
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
            DeclareLaunchArgument("publish_world_to_court_tf", default_value="true"),
            # Retain the previous argument as a disable-only compatibility alias.
            DeclareLaunchArgument("publish_identity_world_to_court_tf", default_value="true"),
            DeclareLaunchArgument("start_corner", default_value="unknown"),
            DeclareLaunchArgument(
                "start_inset_m",
                default_value="0.75",
                description="Distance from the specified court corner to the base_link start pose.",
            ),
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
                        "corner_inset_m": ParameterValue(start_inset_m, value_type=float),
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
            OpaqueFunction(function=_world_to_court_tf),
        ]
    )
