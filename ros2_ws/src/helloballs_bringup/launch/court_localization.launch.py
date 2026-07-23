from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
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
    fusion_enabled = LaunchConfiguration("use_fusion").perform(context)
    if fusion_enabled.lower() in ("1", "true", "yes", "on"):
        return []

    enabled = LaunchConfiguration("publish_world_to_court_tf").perform(context)
    legacy_enabled = LaunchConfiguration("publish_identity_world_to_court_tf").perform(context)
    if (
        enabled.lower() not in ("1", "true", "yes", "on") or
        legacy_enabled.lower() not in ("1", "true", "yes", "on")
    ):
        return []

    start_side = LaunchConfiguration("start_side").perform(context)
    court_length = float(LaunchConfiguration("court_length_m").perform(context))
    court_width = float(LaunchConfiguration("court_width_m").perform(context))

    half_length = court_length * 0.5
    half_doubles_width = court_width * 0.5
    poses = {
        "sideline_left": (
            -half_length,
            half_doubles_width,
            -1.5707963267948966,
        ),
        "sideline_right": (
            -half_length,
            -half_doubles_width,
            1.5707963267948966,
        ),
    }
    if start_side == "unknown":
        x, y, yaw = 0.0, 0.0, 0.0
    elif start_side in poses:
        x, y, yaw = poses[start_side]
    else:
        raise RuntimeError(
            "start_side must be unknown, sideline_left, or sideline_right"
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
    start_side = LaunchConfiguration("start_side")
    camera_height_m = LaunchConfiguration("camera_height_m")
    camera_pitch_rad = LaunchConfiguration("camera_pitch_rad")
    court_length_m = LaunchConfiguration("court_length_m")
    court_width_m = LaunchConfiguration("court_width_m")
    singles_width_m = LaunchConfiguration("singles_width_m")
    service_line_distance_from_net_m = LaunchConfiguration("service_line_distance_from_net_m")
    match_doubles_sidelines = LaunchConfiguration("match_doubles_sidelines")
    full_map_min_vio_translation_m = LaunchConfiguration("full_map_min_vio_translation_m")
    full_map_min_vio_rotation_rad = LaunchConfiguration("full_map_min_vio_rotation_rad")
    use_fusion = LaunchConfiguration("use_fusion")
    fusion_output_topic = LaunchConfiguration("fusion_output_topic")
    fusion_publish_tf = LaunchConfiguration("fusion_publish_tf")
    fusion_confirmation_count = LaunchConfiguration("fusion_confirmation_count")
    fusion_max_sync_error_s = LaunchConfiguration("fusion_max_sync_error_s")
    fusion_correction_gain = LaunchConfiguration("fusion_correction_gain")
    fusion_max_correction_jump_m = LaunchConfiguration("fusion_max_correction_jump_m")
    fusion_max_correction_jump_yaw_rad = LaunchConfiguration(
        "fusion_max_correction_jump_yaw_rad"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/camera/image_mono"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
            DeclareLaunchArgument("vio_odom_topic", default_value="/vio/odometry"),
            DeclareLaunchArgument("world_frame", default_value="world"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("court_frame", default_value="court"),
            DeclareLaunchArgument(
                "use_fusion",
                default_value="true",
                description=(
                    "Start court_vio_fusion_node. When enabled, it is the sole "
                    "publisher of court->world."
                ),
            ),
            DeclareLaunchArgument(
                "fusion_output_topic",
                default_value="/localization/odometry",
            ),
            DeclareLaunchArgument("fusion_publish_tf", default_value="true"),
            DeclareLaunchArgument("fusion_confirmation_count", default_value="3"),
            DeclareLaunchArgument("fusion_max_sync_error_s", default_value="0.15"),
            DeclareLaunchArgument("fusion_correction_gain", default_value="0.2"),
            DeclareLaunchArgument("fusion_max_correction_jump_m", default_value="1.0"),
            DeclareLaunchArgument(
                "fusion_max_correction_jump_yaw_rad",
                default_value="0.35",
            ),
            DeclareLaunchArgument("publish_world_to_court_tf", default_value="true"),
            # Retain the previous argument as a disable-only compatibility alias.
            DeclareLaunchArgument("publish_identity_world_to_court_tf", default_value="true"),
            DeclareLaunchArgument(
                "start_side",
                default_value="unknown",
                description=(
                    "Initial hypothesis: unknown, sideline_left, or sideline_right. "
                    "The vehicle heading is parallel to the short baseline."
                ),
            ),
            DeclareLaunchArgument("camera_height_m", default_value="0.14214"),
            DeclareLaunchArgument("camera_pitch_rad", default_value="0.0"),
            DeclareLaunchArgument("court_length_m", default_value="23.77"),
            DeclareLaunchArgument("court_width_m", default_value="10.97"),
            DeclareLaunchArgument("singles_width_m", default_value="8.23"),
            DeclareLaunchArgument(
                "match_doubles_sidelines",
                default_value="true",
                description=(
                    "Include outer doubles sidelines after switching from the "
                    "startup map to the full court map."
                ),
            ),
            DeclareLaunchArgument("full_map_min_vio_translation_m", default_value="0.5"),
            DeclareLaunchArgument("full_map_min_vio_rotation_rad", default_value="0.35"),
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
                        "start_side": start_side,
                        "camera_height_m": ParameterValue(camera_height_m, value_type=float),
                        "camera_pitch_rad": ParameterValue(camera_pitch_rad, value_type=float),
                        "court_length_m": ParameterValue(court_length_m, value_type=float),
                        "court_width_m": ParameterValue(court_width_m, value_type=float),
                        "singles_width_m": ParameterValue(singles_width_m, value_type=float),
                        "match_doubles_sidelines": ParameterValue(
                            match_doubles_sidelines,
                            value_type=bool,
                        ),
                        "full_map_min_vio_translation_m": ParameterValue(
                            full_map_min_vio_translation_m,
                            value_type=float,
                        ),
                        "full_map_min_vio_rotation_rad": ParameterValue(
                            full_map_min_vio_rotation_rad,
                            value_type=float,
                        ),
                        "service_line_distance_from_net_m": ParameterValue(
                            service_line_distance_from_net_m,
                            value_type=float,
                        ),
                    }
                ],
            ),
            Node(
                package="helloballs_bringup",
                executable="court_vio_fusion_node",
                name="court_vio_fusion",
                output="screen",
                condition=IfCondition(use_fusion),
                parameters=[
                    {
                        "vio_topic": vio_odom_topic,
                        "court_pose_topic": "/court/pose_measurement",
                        "output_topic": fusion_output_topic,
                        "court_frame": court_frame,
                        "world_frame": world_frame,
                        "base_frame": base_frame,
                        "publish_tf": ParameterValue(fusion_publish_tf, value_type=bool),
                        "confirmation_count": ParameterValue(
                            fusion_confirmation_count,
                            value_type=int,
                        ),
                        "max_sync_error_s": ParameterValue(
                            fusion_max_sync_error_s,
                            value_type=float,
                        ),
                        "correction_gain": ParameterValue(
                            fusion_correction_gain,
                            value_type=float,
                        ),
                        "max_correction_jump_m": ParameterValue(
                            fusion_max_correction_jump_m,
                            value_type=float,
                        ),
                        "max_correction_jump_yaw_rad": ParameterValue(
                            fusion_max_correction_jump_yaw_rad,
                            value_type=float,
                        ),
                    }
                ],
            ),
            OpaqueFunction(function=_world_to_court_tf),
        ]
    )
